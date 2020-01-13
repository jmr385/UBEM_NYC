#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:25:27 2019

@author: jonathanroth
"""

import numpy as np
import sys
import multiprocessing as mp
from multiprocessing import Pool, Process
import timeit
import gc
import pickle
import holidays
from bdateutil import isbday
import cvxpy as cvx
from pyswarm import pso
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.local_best import LocalBestPSO
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re
from os.path import isfile, join
from os import listdir
# import seaborn as sns
# import matplotlib.pyplot as plt
import random
from collections import Counter
# os.environ["PATH"] += os.pathsep + '/usr/bin'

# FUNCTIONS
class UBEM_Simulator:

    def __init__(self, sample_buildings=500, modeling_hours=1000):
        # Inputs
        # random.seed(123)
        self.building_classes = 25  # building class
        self.total_buildings = 815032  # building parcels
        self.sample_buildings = sample_buildings
        self.doe_archetypes = 19  # building archetypes
        self.total_hours = 8784  # hours in a year
        self.modeling_hours = modeling_hours  # hours to model

        # Load Files and store as np arrays
        self.m = pd.read_csv(os.getcwd() + '/Data/M.csv').values[:, 1:].astype(float)  # [self.building_classes x K]
        self.a = pd.read_csv(os.getcwd() + '/Data/A.csv').values[:, 2:].astype(float)  # [self.sample_buildings x self.building_classes]
        self.pluto_export = pd.read_csv(os.getcwd() + '/Data/PLUTO_export.csv')
        self.building_energy = self.pluto_export['Energy_kbtu'].values.astype(float)

        self.nyc_8784_electricity= pd.read_csv(os.getcwd() + '/Data/NYC_8784_ElectricityUse.csv')
        self.city_electricity = self.nyc_8784_electricity['Load'].values.astype('float')
        self.city_electricity_scaled = self.city_electricity / np.mean(self.city_electricity[:self.total_hours])

        self.doe_ref_buildings = pd.read_csv(os.getcwd() + '/Data/DOE_RefBuildings.csv')
        self.temperature = self.doe_ref_buildings['Temperature'].values.astype(float).reshape((self.total_hours, 1))
        self.cdd = self.doe_ref_buildings['Cooling_Degree_Hour'].values.astype(float).reshape((self.total_hours, 1))
        self.date_time = self.doe_ref_buildings['Date_Time'].values.astype(str).reshape((self.total_hours, 1))
        self.bday = np.array([isbday(self.date_time[i, 0][0:10], holidays=holidays.US())
                              for i in range(len(self.date_time))]).astype(int)  # TODO

        # MANUAL SCALING and X matrix
        self.sf = 0.5
        self.x = np.zeros([self.doe_archetypes, self.doe_archetypes, self.total_hours]).astype(float)

        for k in range(self.doe_archetypes):
            self.x[k, k, :] = self.doe_ref_buildings.values[:, k + 2]
            self.x[k, k, :] = self.sf * (self.x[k, k, :]) / np.mean(self.x[k, k, :self.total_hours]) + (1 - self.sf)
        print('Shape of X: ', np.shape(self.x))

    def create_a_rand(self, sim_ind):  # Random buildings!
        buildings = self.pluto_export['BldgClass2'].copy()
        buildings[666333] = 'V'  # Need at least two parcels with this building class type
        test_size = float(self.sample_buildings) / len(buildings)
        a_train, a_test, buildings_train, buildings_test = \
            train_test_split(self.a, buildings, test_size=test_size, random_state=sim_ind, stratify=buildings)  # TODO
        # print('All Building Classes: ', self.pluto_export['BldgClass2'].value_counts())
        # print('Types of each Building Class: ', buildings_test.value_counts())
        result = {'a_rand': a_test, 'buildings': buildings_test}
        return result

    def create_training_hours(self, sim_ind):
        np.random.seed(seed=sim_ind)
        rand_starting_hour = np.random.randint(0, self.total_hours)  # TODO
        remaining_hours = self.total_hours - rand_starting_hour - self.modeling_hours
        if remaining_hours >= 0:
            hours_vec = np.arange(rand_starting_hour, rand_starting_hour+self.modeling_hours)
        else:
            end_year = np.arange(rand_starting_hour, self.total_hours)
            beg_year = np.arange(0, -remaining_hours)
            hours_vec = np.concatenate((end_year, beg_year), axis=0)
        return hours_vec.astype(int)

    def create_amx(self, a_rand, training_hours):

        amx = np.zeros([self.sample_buildings, self.doe_archetypes, self.modeling_hours])
        for ind, t in enumerate(training_hours):  # TODO
            amx[:, :, ind] = np.matmul(np.matmul(a_rand, self.m), self.x[:, :, t])  # AMX [self.sample_buildings x K]
        print('Shape of AMX: ', amx.shape)

        # plt.figure(figsize=(15, 4))
        # for k in range(self.doe_archetypes):
        #     plt.plot(training_hours, self.x[k, k, training_hours])
        # plt.show()
        return amx

    def create_prepared_buildings(self, amx, a_rand, buildings, training_hours, sim_ind):
        # Construct y [self.sample_buildings x 6*self.building_classes x self.modeling_hours] and shift time-series
        y = np.zeros([self.sample_buildings, 6 * self.building_classes, self.modeling_hours])
        energy_weight = self.building_energy[buildings.index] / np.sum(self.building_energy[buildings.index])  # *100*len(ind)

        # Plot
        # from matplotlib import rcParams
        # rcParams.update({'font.size': 15})
        # plt.figure(figsize=(10, 5))
        # plt.hist(energy_weight)
        # plt.xlabel('Building energy weight')
        # plt.ylabel('Occurrences')
        # plt.show()
        # plt.savefig('BuildingEnergyDistribution.pdf')
        np.random.seed(seed=sim_ind)
        shift_vec = np.rint(np.random.normal(0, 1.5, self.sample_buildings))  # TODO
        for j in range(self.sample_buildings):
            three_nonzeros = np.nonzero(amx[j, :, 0])  # Get the 3 nonzero entries for each row in AMX (same for all t)
            i_entry = np.nonzero(a_rand[j, :])  # Determine the PLUTO class for row j
            three_entries = np.array([i_entry[0], i_entry[0] + 25, i_entry[0] + 50]).flatten()  # ind in new row to fill
            y[j, three_entries, :] = amx[j, three_nonzeros, :] * energy_weight[j]  # fill y matrix AND add energy_weight
            y[j, three_entries, :] = np.roll(y[j, three_entries, :], int(shift_vec[j]), axis=1)  # shift all self.modeling_hours in row j

            # Add Temperature and BusinessDay vectors
            y[j, (i_entry[0] + 75), :] = self.temperature[training_hours, :].T  # TODO --->
            y[j, (i_entry[0] + 100), :] = self.cdd[training_hours, :].T
            y[j, (i_entry[0] + 125), :] = self.bday[training_hours]
        print('Finished creating buildings...')
        return y

    def optimization_setup(self, prepared_buildings, training_hours, sim_ind):
        start  = timeit.default_timer()
        print(sim_ind, 'Setting up optimization parameters...')
        beta = cvx.Variable(6 * self.building_classes)  # <------------------------- 22
        Ec = np.reshape(self.city_electricity_scaled[training_hours], (self.modeling_hours,))
        Eaj = np.array([np.array([prepared_buildings[j, :, t] * beta
                                  for j in range(self.sample_buildings)]) for t in range(self.modeling_hours)])
        Ea = np.sum(Eaj, axis=1)

        print(sim_ind, 'Shape of Ec, Eaj, Ea', np.shape(Ec), np.shape(Eaj), np.shape(Ea))
        print(sim_ind, 'Starting Hour: ', training_hours[0])
        stop = timeit.default_timer()
        print(sim_ind, 'Finished! Total time to set up parameters: ', stop - start)  # ~100sec
        parameters = {'Ea': Ea, 'Ec': Ec, 'beta': beta}
        return parameters

    def optimization_solver(self, parameters, sim_ind, training_hours):
        # Optimization Solution
        print(sim_ind, 'Creating constraints and objective function...')
        f = parameters['Ea'] - parameters['Ec']
        obj = 0
        for t in range(self.modeling_hours):
            obj = obj + f[t] ** 2
        constraints = [1 >= parameters['beta'][:75], parameters['beta'][:75] >= 0]
        for i in range(self.building_classes):
            constraints += [parameters['beta'][i] + parameters['beta'][i + 25] + parameters['beta'][i + 50] == 1]
        problem = cvx.Problem(cvx.Minimize(obj), constraints)

        print(sim_ind, 'Solving Optimization Function...')
        start = timeit.default_timer()
        problem.solve(solver=cvx.SCS)
        stop = timeit.default_timer()
        print(sim_ind, 'Finished! Total time to solve optimization: ', stop - start)  # ~250sec | 8sec vs.
        optimization_results = {'parameters': parameters, 'problem': problem}
        return optimization_results

    def plot_optimization_results(self, optimization_results, prepared_buildings, sim_ind, training_hours):
        from matplotlib import rcParams
        beta = optimization_results['parameters']['beta']
        Ec = optimization_results['parameters']['Ec']

        # Optimization prediction
        Eaj_hat = np.array([prepared_buildings[j, :, :].T * beta.value for j in range(self.sample_buildings)])
        Ea_hat = np.sum(Eaj_hat, axis=2)
        Ea_hat = np.sum(Ea_hat, axis=0).reshape(self.modeling_hours, )

        # Plots
        # rcParams.update({'font.size': 18})
        # plt.figure(figsize=(15, 5))
        # plt.plot(np.arange(self.modeling_hours), Ec[:self.modeling_hours], linewidth=1.5, label='Actual')
        # plt.plot(np.arange(self.modeling_hours), Ea_hat[:self.modeling_hours], '--', linewidth=2, label='Optimized prediction')
        # plt.axis([0, 1000, 0.65, 1.3])
        # plt.ylabel('Normalized Energy Usage')
        # plt.xlabel('Time (hr)')
        # plt.legend()
        # plt.show()
        # plt.savefig('OptimizationResults.pdf')

        # plt.figure(figsize=(15, 5))
        # plt.plot(np.arange(self.modeling_hours), Ec[:self.modeling_hours], linewidth=2.5, label='Actual')
        # plt.plot(np.arange(self.modeling_hours), Ea_hat[:self.modeling_hours], '--', linewidth=2, label='Optimized prediction')
        # plt.axis([75, 340, 0.65, 1.3])
        # plt.ylabel('Normalized Energy Usage')
        # plt.xlabel('Time (hr)')
        # plt.legend()
        # plt.show()
        # plt.savefig('OptimizationResultsZoom.pdf')

        error = np.abs(Ea_hat - Ec[:self.modeling_hours]) / (Ec[:self.modeling_hours]) * 100
        # plt.figure(figsize=(17, 5))
        # plt.plot(np.arange(self.modeling_hours), error)
        # plt.ylabel('Error (MAPE)')
        # plt.xlabel('Time (hr)')
        # plt.show()
        print(sim_ind, 'Relative error: %.2f %% ' % np.mean(error))
        return error

    def simulation(self, sim_ind):

        # Create Matrices
        start = timeit.default_timer()
        a_rand, buildings = self.create_a_rand(sim_ind=sim_ind).values()

        # training_hours = self.create_training_hours(sim_ind=sim_ind)  # TODO: Want same training hours across all simulations??
        training_hours = np.arange(self.modeling_hours)
        amx = self.create_amx(a_rand=a_rand, training_hours=training_hours)
        prepared_buildings = self.create_prepared_buildings(amx=amx,
                                                            a_rand=a_rand,
                                                            buildings=buildings,
                                                            training_hours=training_hours,
                                                            sim_ind=sim_ind)
        # print('Finished creating prepared_buildings...')

        # Solve Optimization
        parameters = self.optimization_setup(prepared_buildings=prepared_buildings,
                                             training_hours=training_hours,
                                             sim_ind=sim_ind)
        optimization_results = self.optimization_solver(parameters=parameters,
                                                        sim_ind=sim_ind,
                                                        training_hours=training_hours)

        # Create Plots
        error = self.plot_optimization_results(optimization_results=optimization_results,
                                               prepared_buildings=prepared_buildings,
                                               sim_ind=sim_ind,
                                               training_hours=training_hours)

        # Save Results
        results = [buildings.index, optimization_results['parameters']['beta'].value, np.mean(error)]
        print('Clearing Memory...')
        gc.collect()
        stop = timeit.default_timer()
        print('Finished simulation with total time: ', stop - start)  # ~100sec

        return results

    def monte_carlo_simulator(self, num_simulations):

        # Create vectors to save results from each simulation
        building_vec = []
        training_hours_vec = []
        prepared_buildings_vec = []
        betas_vec = []
        error_vec = []

        for sim_ind in range(num_simulations):

            print('---------------- SIMULATION NUMBER ' + str(sim_ind) + '----------------')
            start = timeit.default_timer()
            # Create Matrices
            a_rand, buildings = self.create_a_rand(sim_ind=sim_ind).values()
            # training_hours = self.create_training_hours()  # TODO: Want same training hours across all simulations??
            training_hours = np.arange(self.modeling_hours)
            amx = self.create_amx(a_rand=a_rand, training_hours=training_hours)
            prepared_buildings = self.create_prepared_buildings(amx=amx, a_rand=a_rand, buildings=buildings, training_hours=training_hours)
            print('Finished creating prepared_buildings...')

            # Solve Optimization
            parameters = self.optimization_setup(prepared_buildings=prepared_buildings, training_hours=training_hours)
            optimization_results = self.optimization_solver(parameters=parameters, training_hours=training_hours)

            # Create Plots
            self.plot_beta(optimization_results=optimization_results)
            error = self.plot_optimization_results(optimization_results=optimization_results,
                                                   prepared_buildings=prepared_buildings,
                                                   training_hours=training_hours)

            # Save Results
            building_vec.append(buildings)
            training_hours_vec.append(training_hours)
            prepared_buildings_vec.append(prepared_buildings)
            betas_vec.append(optimization_results['parameters']['beta'].value)
            error_vec.append(error)

            print('Clearing Memory...')
            gc.collect()
            stop = timeit.default_timer()
            print('Finished simulation with total time: ', stop - start)  # ~100sec

        simulation_results = {'building_vec': building_vec,
                              'training_hours_vec': training_hours_vec,
                              'prepared_buildings_vec': prepared_buildings_vec,
                              'betas_vec': betas_vec,
                              'error_vec': error_vec}

        return simulation_results

    def monte_carlo_simulator_light(self, start_num, num_simulations):

        buildings_vec = []
        betas_vec = []
        for i in np.arange(start_num, start_num + num_simulations):
            sim = ubem.simulation(i)
            buildings_vec.append(sim[0])
            betas_vec.append(sim[1])

        simulation_results = {'starting num': [start_num],
                              'buildings_vec': buildings_vec,
                              'betas_vec': betas_vec}
        return simulation_results


def create_prepared_building(ubem, amx, a_rand, training_hours, sim_ind):
    y = np.zeros([6 * ubem.building_classes, training_hours])
    energy_weight = 1
    j = 0

    np.random.seed(seed=sim_ind)
    shift_vec = np.rint(np.random.normal(0, 1.5, ubem.sample_buildings))  # TODO

    three_nonzeros = np.nonzero(amx[j, :, 0])  # Get the 3 nonzero entries for each row in AMX (same for all t)
    i_entry = np.nonzero(a_rand[j, :])  # Determine the PLUTO class for row j
    three_entries = np.array([i_entry[0], i_entry[0] + 25, i_entry[0] + 50]).flatten()  # ind in new row to fill
    y[three_entries, :] = amx[j, three_nonzeros, :] * energy_weight  # fill y matrix AND add energy_weight
    y[three_entries, :] = np.roll(y[three_entries, :], int(shift_vec[j]), axis=1)  # shift all self.modeling_hours in row j

    # Add Temperature and BusinessDay vectors
    y[(i_entry[0] + 75), :] = ubem.temperature[np.arange(training_hours), :].T  # TODO --->
    y[(i_entry[0] + 100), :] = ubem.cdd[np.arange(training_hours), :].T
    y[(i_entry[0] + 125), :] = ubem.bday[np.arange(training_hours)]
    print('Finished creating building...')
    return y


def objective_function(x, all_buildings, betas, Ec, ubem):
    building_beta_assignment = np.rint(x).astype(int)
    print('building_beta_assignment', building_beta_assignment.shape)
    print('betas[building_beta_assignment[:j]]', betas[building_beta_assignment[:, 0]].shape)
    print('betas[building_beta_assignment', betas[building_beta_assignment].shape)
    print('all_buildings.shape', all_buildings.shape)
    Eaj_hat = np.array([np.matmul(all_buildings[j, :, :].T, betas[building_beta_assignment[:, j]].T) for j in range(all_buildings.shape[0])])
    print('Eaj_hat.shape', Eaj_hat.shape)
    # Ea_hat = np.sum(Eaj_hat, axis=2)
    # print('Ea_hat.shape', Ea_hat.shape)
    Ea_hat = np.sum(Eaj_hat, axis=0).T
    print('Ea_hat.shape', Ea_hat.shape)
    adj = all_buildings.shape[0]/ubem.sample_buildings
    obj = np.abs(Ea_hat - adj*Ec[:ubem.modeling_hours]) / (adj*Ec[:ubem.modeling_hours]) * 100  # 100
    print('obj shape', obj.shape)
    obj = np.mean(obj, axis=1)
    print('final obj shape', obj.shape)
    return obj


def pso(ubem, all_buildings, betas, Ec, n_particles=25, iters=300, global_pso=True):
    start = timeit.default_timer()
    num_buildings = all_buildings.shape[0]
    mc_simulations = len(betas)
    lb, ub = np.repeat(0, num_buildings), np.repeat(mc_simulations-1, num_buildings)
    bounds = (lb, ub)

    # Use Global or Local optimizer
    if global_pso:
        options = {'c1': 0.5, 'c2': 0.1, 'w': 0.1}  # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=num_buildings, options=options, bounds=bounds)
    else:
        options = {'c1': 0.5, 'c2': 0.1, 'w': 0.1, 'k': 3, 'p': 1}
        optimizer = LocalBestPSO(n_particles=n_particles, dimensions=num_buildings, options=options, bounds=bounds)

    kwargs = {'all_buildings': all_buildings, 'betas': np.array(betas), 'Ec': Ec, 'ubem': ubem}
    cost, pos = optimizer.optimize(objective_function, iters=iters, **kwargs)

    stop = timeit.default_timer()
    print('--- Total time for PSO with ' + str(n_particles) + 'particles and ' + str(iters) + ' loops: ', stop - start)

    return (cost, pos)


def random_betas_new(all_buildings, betas, Ec, iters=50):
    print('--- Starting to make random betas ---')
    start = timeit.default_timer()
    num_buildings = all_buildings.shape[0]
    mc_simulations = len(betas)
    beta_list = []
    error_list = []
    for i in range(iters):
        building_beta_assignment = np.random.randint(0, mc_simulations-1, size=num_buildings)
        beta_matrix = np.array(betas)[building_beta_assignment]
        Ea_hat = np.einsum('ijk,ij->k', all_buildings, beta_matrix)
        adj = len(betas)
        error = np.mean(np.abs(Ea_hat - adj * Ec[:ubem.modeling_hours]) / (adj * Ec[:ubem.modeling_hours]) * 100)

        beta_list.append(building_beta_assignment)
        error_list.append(error)
        print(i, 'MAPE: ', error)

    results = {'Assignments': beta_list, 'Error': error_list}
    stop = timeit.default_timer()
    print('Finished! Total time when vectorized: ', stop - start)  # ~100sec
    return results


def check_error(pos, betas, Ec, all_buildings, one_mc_simulation=False, sim_num=0, sim_buildings=500):
    # CHECK IF ERROR ALIGNS
    num_buildings = all_buildings.shape[0]
    mc_simulations = len(betas)
    if not one_mc_simulation:
        building_beta_assignment = np.rint(pos).astype(int)
        Eaj_hat = np.array([all_buildings[j, :, :].T * betas[building_beta_assignment[j]] for j in range(num_buildings)])
        adj = len(betas)  # np.sum(Ea_hat) / np.sum(Ec) # TODO: This or 4?
    else:
        building_beta_assignment = np.repeat(sim_num, sim_buildings)
        Eaj_hat = np.array([all_buildings[j, :, :].T * betas[sim_num-1] for j in range(sim_buildings)])
        adj = 1
    Ea_hat = np.sum(Eaj_hat, axis=2)
    Ea_hat = np.sum(Ea_hat, axis=0).reshape(ubem.modeling_hours, )
    error = np.mean(np.abs(Ea_hat - adj * Ec[:ubem.modeling_hours]) / (adj * Ec[:ubem.modeling_hours]) * 100)
    print('Checking Error: ', error)
    return error


def monte_carlo_simulator_parallel(ubem, num_simulations, starting_num):
    start = timeit.default_timer()

    pool = mp.Pool(5)
    simulation_results = pool.map(ubem.simulation, [i for i in np.arange(starting_num, starting_num + num_simulations)])
    pool.close()
    pool.join()

    stop = timeit.default_timer()
    print('ALL SIMULATIONS TIME: ', stop - start)  # ~100sec

    return simulation_results


def create_all_buildings(ubem, total_simulations, sim_num=0):
    all_buildings = []
    for sim_ind in np.arange(total_simulations):

        a_rand, buildings = ubem.create_a_rand(sim_ind=sim_ind + 15 + sim_num).values()
        training_hours = np.arange(ubem.modeling_hours)
        amx = ubem.create_amx(a_rand=a_rand, training_hours=training_hours)
        prepared_buildings = ubem.create_prepared_buildings(amx=amx,
                                                            a_rand=a_rand,
                                                            buildings=buildings,
                                                            training_hours=training_hours,
                                                            sim_ind=sim_ind + 15 + sim_num)
        all_buildings.append(prepared_buildings)
        print('Finished creating prepared_buildings...')

    all_buildings = np.concatenate(all_buildings, axis=0)
    return all_buildings


def get_simulation_errors():
    logs = [1, 2, 65, 115, 215, 265, 315, 365, 415, 465]
    log_files = [os.getcwd() + '/Simulation_output/UBEM_batch' + str(x) + '.log' for x in logs]
    all_errors = []
    for f in log_files:
        with open(f, 'r') as file:
            log_string = file.read()

        error_locations = [m.start() for m in re.finditer('Relative error:', log_string)]
        errors = [np.float(log_string[location + 16: (location + 16 + log_string[location + 17:].find('%'))])
                  for location in error_locations]
        all_errors.append(errors)
    all_errors = np.array(all_errors).T
    return all_errors


class run_pso:

    def __init__(self, ubem, betas, Ec, n_particles, iters, global_pso, num_buildings, rand_intializations):
        self.ubem = ubem
        self.betas = betas
        self.Ec = Ec
        self.n_particles = n_particles
        self.iters = iters
        self.global_pso = global_pso
        self.num_buildings = num_buildings
        self.rand_intializations = rand_intializations

    def run(self, sim):
        pso_dict = {}
        all_buildings = create_all_buildings(ubem=self.ubem, total_simulations=self.num_buildings, sim_num=sim)
        for i in np.arange(self.rand_intializations):
            cost, pos = pso(ubem=self.ubem, all_buildings=all_buildings, betas=self.betas, Ec=self.Ec,
                            n_particles=self.n_particles, iters=self.iters, global_pso=self.global_pso)
            pso_dict[cost] = np.round(pos)
        return pso_dict


def run_pso_parallel(run_pso, sim_num=10):
    start = timeit.default_timer()

    pool = mp.Pool(5)
    sim_results = pool.map(run_pso.run, [i for i in np.arange(sim_num)])
    pool.close()
    pool.join()

    stop = timeit.default_timer()
    print('Total Simulations: ', sim_num*2, ' ALL SIMULATIONS TIME: ', stop - start)  #
    return sim_results


def scale_all_doe_columns(doe_df):
    doe_df['TE'] = doe_df['Electricity:Facility [kW](Hourly)'] + doe_df['Gas:Facility [kW](Hourly)']
    doe_df['TE_scaled'] = doe_df['TE'] / np.sum(doe_df['TE'])
    doe_df['Electricity_scaled'] = doe_df['Electricity:Facility [kW](Hourly)'] / doe_df['TE']
    doe_df['EFans_scaled'] = doe_df['Fans:Electricity [kW](Hourly)'] / doe_df['TE']
    doe_df['ECooling_scaled'] = doe_df['Cooling:Electricity [kW](Hourly)'] / doe_df['TE']
    doe_df['EHeating_scaled'] = doe_df['Heating:Electricity [kW](Hourly)'] / doe_df['TE']
    doe_df['Gas_scaled'] = doe_df['Gas:Facility [kW](Hourly)'] / doe_df['TE']
    doe_df['GHeating_scaled'] = doe_df['Heating:Gas [kW](Hourly)'] / doe_df['TE']
    print('doe shape: ', doe_df.shape[1])
    if doe_df.shape[1] != 22:
        doe_df['ELights_scaled'] = doe_df['InteriorLights:Electricity [kW](Hourly)'] / doe_df['TE']
        doe_df['EEquipment_scaled'] = doe_df['InteriorEquipment:Electricity [kW](Hourly)'] / doe_df['TE']

        if 'Water Heater:WaterSystems:Gas [kW](Hourly)' in doe_df.columns:
            doe_df['GWaterheat_scaled'] = doe_df['Water Heater:WaterSystems:Gas [kW](Hourly)'] / doe_df['TE']
        else:
            doe_df['GWaterheat_scaled'] = 0
    else:
        doe_df['ELights_scaled'] = doe_df['General:InteriorLights:Electricity [kW](Hourly)'] / doe_df['TE'] +\
                                   doe_df['General:ExteriorLights:Electricity [kW](Hourly)'] / doe_df['TE']
        doe_df['EEquipment_scaled'] = doe_df['Appl:InteriorEquipment:Electricity [kW](Hourly)'] / doe_df['TE'] +\
                                      doe_df['Misc:InteriorEquipment:Electricity [kW](Hourly)'] / doe_df['TE']
        doe_df['EWaterheat_scaled'] = doe_df['Water Heater:WaterSystems:Electricity [kW](Hourly) '] / doe_df['TE']
    return doe_df


def scale_all_doe_datasets(calculate=False):
    directory = '/Data/DOE_raw2/' if calculate is True else '/Data/DOE_scaled/'
    doe_directory = os.getcwd() + directory
    doe_file_names = [f for f in listdir(doe_directory) if isfile(join(doe_directory, f))]
    doe_file_number = [f.split('_')[0] for f in doe_file_names]
    doe_file_number_sorted = np.sort(np.array(doe_file_number).astype(int)).astype(str)
    all_doe_datasets = [doe_file_names[doe_file_number.index(n)] for n in doe_file_number_sorted]
    doe_list = np.array([pd.read_csv(os.getcwd() + directory + n) for n in all_doe_datasets])

    if calculate is False:
        return list(doe_list)

    doe_list2 = [pd.concat([d.iloc[0:1416, ], d.iloc[1416:1440, ], d.iloc[1416:, ]]) for d in doe_list]
    doe_list2 = [scale_all_doe_columns(doe_df=d) for d in doe_list2]
    for ind, d in enumerate(doe_list2):
        doe_directory = os.getcwd() + directory
        d.to_csv(os.getcwd() + '/Data/DOE_scaled/' + all_doe_datasets[ind])
    return doe_list2


def create_hourly_load(ubem, beta_vec, A_building, doe_1a, doe_2a, doe_3a, column_name, energy):
    i_entry = np.nonzero(A_building[0, :])[0]  # Determine the PLUTO class for row j
    hours = doe_1a.shape[0]

    d1 = np.array(doe_1a[column_name] * beta_vec[i_entry]).reshape(hours, 1)
    d2 = np.array(doe_2a[column_name] * beta_vec[i_entry + 25]).reshape(hours, 1)
    d3 = np.array(doe_3a[column_name] * beta_vec[i_entry + 50]).reshape(hours, 1)

    # t1 = ubem.temperature * beta_vec[i_entry + 75]
    # c1 = ubem.cdd * beta_vec[i_entry + 100]
    # b1 = ubem.bday * beta_vec[i_entry + 125]

    final_vec = d1 + d2 + d3 # + t1.reshape(hours, 1) + c1.reshape(hours, 1) + b1.reshape(hours, 1)

    return final_vec * np.array(energy).reshape(hours,1)


def create_hour_dataframe():
    return 0


def create_one_building_timeseries(ubem, betas, bbl, doe_list, modeling_hours=8784., ll84=False):

    ubem.pluto_export['BBL'] = ubem.pluto_export['BBL'].astype(str)
    building_pluto = ubem.pluto_export.loc[ubem.pluto_export['BBL'] == bbl]
    A_building = ubem.a[building_pluto.index,]
    doe_datasets_needed = np.matmul(A_building, ubem.m)
    doe_datasets_needed_3 = list(np.nonzero(doe_datasets_needed)[1])
    three_doe_datasets = doe_list[doe_datasets_needed_3]
    energy = ubem.pluto_export.loc[ll84['BBL'] == bbl, 'Energy_kbtu'].values

    if ll84 == True:
        ll84 = pd.read_csv(os.getcwd() + '/Data/LL84.csv')
        ll84['Energy[kBtu]'] = ll84['Site EUI (kBtu/ft²)'] * ll84['DOF Gross Floor Area']
        ll84['BBL'] = ll84['BBL - 10 digits'].astype(str)
        bbl = bbl + '.0'
        energy = ll84.loc[ll84['BBL'] == bbl, 'Energy_kbtu'].values

    doe_1a = pd.read_csv(os.getcwd() + '/Data/DOE_scaled/' + three_doe_datasets[0])
    doe_2a = pd.read_csv(os.getcwd() + '/Data/DOE_scaled/' + three_doe_datasets[1])
    doe_3a = pd.read_csv(os.getcwd() + '/Data/DOE_scaled/' + three_doe_datasets[2])
    print(doe_3a.shape)

    # MANUAL SCALING and X matrix
    sf = 0.5
    new_x = np.zeros([ubem.doe_archetypes, ubem.doe_archetypes, ubem.total_hours]).astype(float)
    doe_ref_buildings_energy = ubem.doe_ref_buildings.copy()
    doe_ref_buildings_energy.iloc[:, 2:] = 0
    doe_ref_buildings_energy.iloc[:, list(doe_datasets_needed_str.astype(int)+1)] = np.array([doe_1a['TE_scaled'],
                                                                                              doe_2a['TE_scaled'],
                                                                                              doe_3a['TE_scaled']]).T
    for k in range(ubem.doe_archetypes):
        new_x[k, k, :] = ubem.doe_ref_buildings.values[:, k + 2]
        new_x[k, k, :] = sf * (new_x[k, k, :]) / np.mean(new_x[k, k, :8784]) + (1 - sf)


    AMX_building = np.zeros([1, ubem.doe_archetypes, modeling_hours])
    for ind, t in enumerate(np.arange(8784)):
        AMX_building[:, :, ind] = np.matmul(doe_datasets_needed, ubem.x[:, :, t])

    building_prepared = create_prepared_building(ubem=ubem, amx=AMX_building, a_rand=A_building,
                                                 training_hours=modeling_hours, sim_ind=1)
    building_timeseries = np.matmul(betas[0, :], building_prepared) * energy / modeling_hours
    building_hourly = pd.DataFrame({'Total Energy': building_timeseries})

    return building_hourly


if __name__ == '__main__':
    total_hours = 1000
    ubem = UBEM_Simulator(sample_buildings=1000, modeling_hours=total_hours)  # 6148
    # starting_num = int(sys.argv[1])
    # simulations = monte_carlo_simulator_parallel(ubem=ubem, num_simulations=50, starting_num=starting_num)
    # pickle.dump(simulations, open(os.getcwd() + '/test005_1000_1000_50_' + str(starting_num) + '.obj', 'wb'))

    # Extract simulations
    list_of_simulations = [pickle.load(open(os.getcwd() + '/Data/test005_1000_1000_50_' + str(15 + num*50) + '.obj', 'rb'))
                           for num in np.arange(10)]

    # PSO
    betas = np.array([sim[1] for simulations in list_of_simulations for sim in simulations])
    indices = np.array([sim[1] for simulations in list_of_simulations for sim in simulations])
    training_hours = np.arange(total_hours)
    Ec = np.reshape(ubem.city_electricity_scaled[training_hours], (ubem.modeling_hours,))

    # RUN PSO
    run_pso_instance = run_pso(ubem=ubem, betas=betas, Ec=Ec, n_particles=20, iters=20, global_pso=True, num_buildings=1, rand_intializations=1)
    run_multiple_pso = run_pso_parallel(run_pso_instance, sim_num=500)
    pickle.dump(run_multiple_pso, open(os.getcwd() + '/Data/sim500_BC1000_HC1000.obj', 'wb'))

    # Extract errors from log files
    # all_errors = get_simulation_errors()
    # all_errors = np.array(all_errors).flatten()

    # ax = sns.distplot(all_errors, kde=False, hist=True, color='#274a5c', hist_kws={"alpha":1})
    # ax.axvline(np.mean(all_errors), color='black', linestyle='-')
    # plt.xlabel('Mean Absolute Percentage Error')
    # plt.ylabel('Count')
    # plt.savefig('/Users/jonathanroth/PycharmProjects/UBEM_NYC/Error_Distribution.pdf')
    # plt.show()

    # # CHECK ERRORS
    # all_buildings = create_all_buildings(ubem=ubem, total_simulations=1, sim_num=495)
    # check_error(np.repeat(1, 1000), betas, Ec, all_buildings, one_mc_simulation=True, sim_num=496, sim_buildings=1000)

    # CHECK PSO
    sim500_BC1000_HC1000 = pickle.load(open('/Users/jonathanroth/PycharmProjects/UBEM_NYC/Data/sim500_BC1000_HC1000.obj', 'rb'))
    costs = np.array([list(k.keys()) for k in sim500_BC1000_HC1000]).flatten()


    # HOURLY LOAD ONE BUILDING
    doe_list = scale_all_doe_datasets(calculate=True)







    ubem = UBEM_Simulator(sample_buildings=1000, modeling_hours=8784)  # 6148
    ll84 = pd.read_csv(os.getcwd() + '/Data/LL84.csv')
    ll84['Energy[kBtu]'] = ll84['Site EUI (kBtu/ft²)'] * ll84['DOF Gross Floor Area']
    ll84['BBL'] = ll84['BBL - 10 digits'].astype(str)
    ubem.pluto_export['BBL'] = ubem.pluto_export['BBL'].astype(str)

    chrystler_building = '1012970023.0'
    chrystler_building_ll84 = ll84.loc[ll84['BBL'] == chrystler_building]

    chrystler_building = '1012970023'
    chrystler_building_pluto = ubem.pluto_export.loc[ubem.pluto_export['BBL'] == chrystler_building]
    A_chrystler = ubem.a[chrystler_building_pluto.index, ]

    # GET DOE CSV FILES
    doe_directory = os.getcwd() + '/Data/DOE_scaled/'
    doe_file_names = [f for f in listdir(doe_directory) if isfile(join(doe_directory, f))]
    doe_file_number = [f.split('_')[0] for f in doe_file_names]
    doe_datasets_needed = np.matmul(A_chrystler, ubem.m)

    doe_datasets_needed_str = np.array(np.nonzero(doe_datasets_needed)[1] + 1).astype(str)
    three_doe_datasets = [doe_file_names[doe_file_number.index(n)] for n in doe_datasets_needed_str]

    doe_1a = pd.read_csv(os.getcwd() + '/Data/DOE_scaled/' + three_doe_datasets[0])
    doe_2a = pd.read_csv(os.getcwd() + '/Data/DOE_scaled/' + three_doe_datasets[1])
    doe_3a = pd.read_csv(os.getcwd() + '/Data/DOE_scaled/' + three_doe_datasets[2])

    # MANUAL SCALING and X matrix
    sf = 0.5
    new_x = np.zeros([ubem.doe_archetypes, ubem.doe_archetypes, ubem.total_hours]).astype(float)
    doe_ref_buildings_energy = ubem.doe_ref_buildings.copy()
    doe_ref_buildings_energy.iloc[:, 2:] = 0
    doe_ref_buildings_energy.iloc[:, list(doe_datasets_needed_str.astype(int)+1)] = np.array([doe_1a['TE_scaled'],
                                                                                              doe_2a['TE_scaled'],
                                                                                              doe_3a['TE_scaled']]).T
    for k in range(ubem.doe_archetypes):
        new_x[k, k, :] = ubem.doe_ref_buildings.values[:, k + 2]
        new_x[k, k, :] = sf * (new_x[k, k, :]) / np.mean(new_x[k, k, :8784]) + (1 - sf)

    AMX_chrystler = np.zeros([1, ubem.doe_archetypes, 8784])
    doe_datasets_needed = np.matmul(A_chrystler, ubem.m)
    for ind, t in enumerate(np.arange(8784)):
        AMX_chrystler[:, :, ind] = np.matmul(doe_datasets_needed, ubem.x[:, :, t])

    chrystler_prepared = create_prepared_building(ubem=ubem, amx=AMX_chrystler, a_rand=A_chrystler,
                                                  training_hours=8784, sim_ind=1)
    chrystler_timeseries = np.matmul(betas[0, :], chrystler_prepared) * chrystler_building_ll84['Energy[kBtu]'].values/8784.
    chrystler_building_hourly = pd.DataFrame({'Total Energy': chrystler_timeseries})
    beta_vec = betas[288,:]
    chrystler_building_hourly['Electricity'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='Electricity_scaled',
                                                                  energy=chrystler_building_hourly['Total Energy'])
    chrystler_building_hourly['Gas'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='Gas_scaled',
                                                                  energy=chrystler_building_hourly['Total Energy'])
    chrystler_building_hourly['Cooling'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='ECooling_scaled',
                                                                  energy=chrystler_building_hourly['Total Energy'])
    chrystler_building_hourly['Lights'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='ELights_scaled',
                                                                  energy=chrystler_building_hourly['Total Energy'])
    chrystler_building_hourly['Equipment'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='EEquipment_scaled',
                                                                  energy=chrystler_building_hourly['Total Energy'])
    chrystler_building_hourly['Gas_Heating'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='GHeating_scaled',
                                                                  energy=chrystler_building_hourly['Total Energy'])
    chrystler_building_hourly['Elec_Heating'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='EHeating_scaled',
                                                                  energy=chrystler_building_hourly['Total Energy'])
    chrystler_building_hourly['GWater_Heating'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='GWaterheat_scaled',
                                                                  energy=chrystler_building_hourly['Total Energy'])
    chrystler_building_hourly['Heating'] = chrystler_building_hourly['Gas_Heating'] + \
                                           chrystler_building_hourly['Elec_Heating'] + \
                                           chrystler_building_hourly['GWater_Heating']

    chrystler_building_hourly['Cooling'] = np.abs(chrystler_building_hourly['Cooling'])


    chrystler_building_hourly.to_csv(os.getcwd() + '/Data/Chrystler288.csv')

    def plot_2x2_hourly_load(ubem, building_hourly, start=0, end=168, duration=168):
        import matplotlib.pyplot as plt
        dates = ubem.doe_ref_buildings.iloc[0:duration, 1]

        plt.plot(np.arange(duration), building_hourly['Total Energy'].iloc[start:end], color='k')
        plt.plot(np.arange(duration), building_hourly['Electricity'].iloc[start:end], color='orange')
        plt.plot(np.arange(duration), building_hourly['Gas'].iloc[start:end], color='saddlebrown')
        plt.ylabel('Energy [kBtu]')
        plt.legend(('Total Energy', 'Electricity', 'Gas'), loc=2)
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        fig = plt.gcf()
        fig.set_size_inches(9, 4.5)
        plt.savefig(os.getcwd() + '/Figures/Chrystler_HourlyEnergy_' + str(start) + '_.pdf')
        plt.show()

        plt.plot(np.arange(duration), building_hourly['Cooling'].iloc[start:end], color='b')
        plt.plot(np.arange(duration), building_hourly['Heating'].iloc[start:end], color='r')
        plt.plot(np.arange(duration), building_hourly['GWater_Heating'].iloc[start:end], color='maroon')
        plt.legend(('Cooling', 'Heating', 'Water Heating'), loc=2)
        plt.ylabel('Energy [kBtu]')
        plt.xlabel('Hours')
        plt.xticks(np.arange(0, duration, step=24),
                   ('12am Sun', '12am Mon', '12am Tue', '12am Wed', '12am Thr', '12am Fri', '12am Sat'),
                   rotation=36)
        fig = plt.gcf()
        fig.set_size_inches(9, 5.5)
        plt.savefig(os.getcwd() + '/Figures/Chrystler_HourlyCoolHeat_' + str(start) + '_.pdf')
        plt.show()

    plot_2x2_hourly_load(ubem, chrystler_building_hourly, start=7*24, end=7*24+168, duration=168)
    plot_2x2_hourly_load(ubem, chrystler_building_hourly, start=152*24 + 3*24, end=152*24 + 3*24+168, duration=168)




