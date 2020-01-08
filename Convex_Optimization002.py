#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:25:27 2019

@author: jonathanroth
"""

import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Pool, Process
import timeit
import gc
import pickle
from datetime import date
import holidays
from bdateutil import isbday
import cvxpy as cvx
# from pyswarm import pso
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.local_best import LocalBestPSO
import pandas as pd
# np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='jmr385', api_key='eoGN5xwrqLAzwCZdAFih')
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
from sklearn.model_selection import train_test_split
import os
import random
from collections import Counter
os.environ["PATH"] += os.pathsep + '/usr/bin'

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
        self.m = pd.read_csv('/Users/jonathanroth/PycharmProjects/UBEM/M.csv').values[:, 1:].astype(float)  # [self.building_classes x K]
        self.a = pd.read_csv('/Users/jonathanroth/PycharmProjects/UBEM/A.csv').values[:, 2:].astype(float)  # [self.sample_buildings x self.building_classes]
        self.pluto_export = pd.read_csv('/Users/jonathanroth/PycharmProjects/UBEM/Pluto_export.csv')
        self.building_energy = self.pluto_export['Energy_kbtu'].values.astype(float)

        self.nyc_8784_electricity= pd.read_csv('/Users/jonathanroth/PycharmProjects/UBEM/NYC_8784_ElectricityUse.csv')
        self.city_electricity = self.nyc_8784_electricity['Load'].values.astype('float')
        self.city_electricity_scaled = self.city_electricity / np.mean(self.city_electricity[:self.total_hours])

        self.doe_ref_buildings = pd.read_csv('/Users/jonathanroth/PycharmProjects/UBEM/DOE_RefBuildings.csv')
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
        buildings = self.pluto_export['BldgClass2']
        buildings[666333] = 'V'  # Need at least two parcels with this building class type
        test_size = float(self.sample_buildings) / len(buildings)
        a_train, a_test, buildings_train, buildings_test = \
            train_test_split(self.a, buildings, test_size=test_size, random_state=sim_ind, stratify=buildings)  # TODO
        print('All Building Classes: ', self.pluto_export['BldgClass2'].value_counts())
        print('Types of each Building Class: ', buildings_test.value_counts())
        result = {'a_rand': a_test, 'buildings': buildings_test}
        return result

    def create_training_hours(self):
        rand_starting_hour = np.rint(np.random.uniform(0, self.total_hours))  # TODO
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

    def create_prepared_buildings(self, amx, a_rand, buildings, training_hours):
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

    def optimization_setup(self, prepared_buildings, training_hours):
        start  = timeit.default_timer()
        print('Setting up optimization parameters...')
        beta = cvx.Variable(6 * self.building_classes)  # <------------------------- 22
        Ec = np.reshape(self.city_electricity_scaled[training_hours], (self.modeling_hours,))
        Eaj = np.array([np.array([prepared_buildings[j, :, t] * beta
                                  for j in range(self.sample_buildings)]) for t in range(self.modeling_hours)])
        Ea = np.sum(Eaj, axis=1)

        print('Shape of Ec, Eaj, Ea', np.shape(Ec), np.shape(Eaj), np.shape(Ea))
        print('Starting Hour: ', training_hours[0])
        stop = timeit.default_timer()
        print('Finished! Total time to set up parameters: ', stop - start)  # ~100sec
        parameters = {'Ea': Ea, 'Ec': Ec, 'beta': beta}
        return parameters

    def optimization_solver(self, parameters, training_hours):
        # Optimization Solution
        print('Creating constraints and objective function...')
        f = parameters['Ea'] - parameters['Ec']
        obj = 0
        for t in range(self.modeling_hours):
            obj = obj + f[t] ** 2
        constraints = [1 >= parameters['beta'][:75], parameters['beta'][:75] >= 0]
        for i in range(self.building_classes):
            constraints += [parameters['beta'][i] + parameters['beta'][i + 25] + parameters['beta'][i + 50] == 1]
        problem = cvx.Problem(cvx.Minimize(obj), constraints)

        print('Solving Optimization Function...')
        start = timeit.default_timer()
        problem.solve()
        stop = timeit.default_timer()
        print('Finished! Total time to solve optimization: ', stop - start)  # ~250sec | 8sec vs.
        optimization_results = {'parameters': parameters, 'problem': problem}
        return optimization_results

    def plot_beta(self, optimization_results):

        from matplotlib import rcParams
        beta = optimization_results['parameters']['beta']
        problem = optimization_results['problem']
        plt.figure(figsize=(10, 5))
        rcParams.update({'font.size': 20})

        plt.hist(beta.value[:75])
        plt.xlabel(r'$\beta$')
        plt.ylabel('Occurences')
        plt.show()
        # plt.savefig('BetaDistribution.pdf')
        print("status:", problem.status)
        print("optimal value", problem.value)
        print('Largest beta', np.max(beta.value))

    def plot_optimization_results(self, optimization_results, prepared_buildings, training_hours):
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
        print('Relative error: %.2f %% ' % np.mean(error))
        return error

    def simulation(self, sim_ind):

        # Create Matrices
        start = timeit.default_timer()
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
        # self.plot_beta(optimization_results=optimization_results)
        error = self.plot_optimization_results(optimization_results=optimization_results,
                                               prepared_buildings=prepared_buildings,
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


def check_error(pos, one_mc_simulation=False, sim_num=0, sim_buildings=500):
    # CHECK IF ERROR ALIGNS
    num_buildings = all_buildings.shape[0]
    mc_simulations = len(betas)
    if not one_mc_simulation:
        building_beta_assignment = np.rint(pos).astype(int)
        Eaj_hat = np.array([all_buildings[j, :, :].T * betas[building_beta_assignment[j]] for j in range(num_buildings)])
        adj = len(betas)  # np.sum(Ea_hat) / np.sum(Ec) # TODO: This or 4?
    else:
        building_beta_assignment = np.repeat(sim_num, sim_buildings)
        Eaj_hat = np.array([all_buildings[j+(sim_num*sim_buildings), :, :].T * betas[sim_num] for j in range(sim_buildings)])
        adj = 1
    Ea_hat = np.sum(Eaj_hat, axis=2)
    Ea_hat = np.sum(Ea_hat, axis=0).reshape(ubem.modeling_hours, )
    error = np.mean(np.abs(Ea_hat - adj * Ec[:ubem.modeling_hours]) / (adj * Ec[:ubem.modeling_hours]) * 100)
    print('Checking Error: ', error)
    return error


def monte_carlo_simulator_parallel(num_simulations):
    start = timeit.default_timer()

    pool = mp.Pool(4)
    simulation_results = pool.map(ubem.simulation, [i for i in np.arange(5, 5 + num_simulations)])
    pool.close()
    pool.join()

    stop = timeit.default_timer()
    print('ALL SIMULATIONS TIME: ', stop - start)  # ~100sec

    return simulation_results


def create_all_buildings(ubem, total_simulations):
    all_buildings = []
    for sim_ind in np.arange(total_simulations):
        a_rand, buildings = ubem.create_a_rand(sim_ind=sim_ind + 15).values()  # TODO: Del +15???
        training_hours = np.arange(ubem.modeling_hours)
        amx = ubem.create_amx(a_rand=a_rand, training_hours=training_hours)
        prepared_buildings = ubem.create_prepared_buildings(amx=amx, a_rand=a_rand, buildings=buildings, training_hours=training_hours)
        all_buildings.append(prepared_buildings)
        print('Finished creating prepared_buildings...')

    all_buildings = np.concatenate(all_buildings, axis=0)
    return all_buildings

if __name__ == '__main__':

    ubem = UBEM_Simulator(sample_buildings=1000, modeling_hours=1000)  # 6148
    final_results = []
    for i in np.arange(10):
        res_long = ubem.monte_carlo_simulator_light(start_num=0, num_simulations=1)
        final_results.append(res_long)
    # res_parallel = monte_carlo_simulator_parallel(4)
    pickle.dump(final_results, open('/Users/jonathanroth/PycharmProjects/UBEM/res0_10_1000T_1000B_light.obj', 'wb'))

    # MERGE
    res0_3 = pickle.load(open('/Users/jonathanroth/PycharmProjects/UBEM/res0_3_1000T_500B_light.obj', 'rb'))
    res4_7 = pickle.load(open('/Users/jonathanroth/PycharmProjects/UBEM/res4_7_1000T_500B_light.obj', 'rb'))
    res8_11 = pickle.load(open('/Users/jonathanroth/PycharmProjects/UBEM/res8_11_1000T_500B_light.obj', 'rb'))
    ds = [res0_3, res4_7, res8_11]
    res = {}
    for k in res0_3.keys():
        res[k] = list(np.concatenate(np.array(list(d[k] for d in ds))))

    # PREPARE DATA
    # betas = res['betas_vec']
    # training_hours = np.arange(ubem.modeling_hours)
    # Ec = np.reshape(ubem.city_electricity_scaled[training_hours], (ubem.modeling_hours,))

    betas = [r['betas_vec'][0] for r in final_results]
    betas_rounded = np.around(betas, decimals=4)
    training_hours = np.arange(ubem.modeling_hours)
    Ec = np.reshape(ubem.city_electricity_scaled[training_hours], (ubem.modeling_hours,))

    # RUN PARTICLE SWARM OPTIMIZER
    pso_result = []
    pso_cost = []
    for i in np.arange(10000):
        all_buildings = create_all_buildings(ubem=ubem, total_simulations=2)
        cost, pos = pso(ubem=ubem, all_buildings=all_buildings, betas=betas, Ec=Ec, n_particles=30, iters=30, global_pso=False)
        pso_result.append(pos)
        pso_cost.append(cost)


    # random_results = random_betas_new(all_buildings=all_buildings, betas=betas, Ec=Ec, iters=10)
    # check_error(pos=pos, one_mc_simulation=False, sim_num=0, sim_buildings=all_buildings.shape[0])


    def random_betas_parallel2(all_buildings, betas, iters=10):
        print('--- Starting to make random betas ---')
        start = timeit.default_timer()
        num_buildings = all_buildings.shape[0]
        mc_simulations = len(betas)
        pool = mp.Pool(4)

        # TWO
        #%%time
        pool = mp.Pool(4)
        res_vec = []
        for i in iters:
            res = pool.map(random_simulation)


        # THREE
        #%%time
        pool = mp.Pool(4)
        results = pool.map(random_simulation, [i for i in range(iters)])
        pool.close()
        pool.join()
        stop = timeit.default_timer()
        print('Finished! Total time when Parallel: ', stop - start)  # ~100sec
        return res


    def random_simulation(i):
        mc_simulations, num_buildings = 5, 2500
        building_beta_assignment = np.random.randint(0, mc_simulations-1, size=num_buildings)
        beta_matrix = np.array(betas)[building_beta_assignment]
        Ea_hat = np.einsum('ijk,ij->k', all_buildings, beta_matrix)
        adj = len(betas)
        error = np.mean(np.abs(Ea_hat - adj * Ec[:ubem.modeling_hours]) / (adj * Ec[:ubem.modeling_hours]) * 100)
        results = {'Assignments': building_beta_assignment, 'Error': error}
        print(i, 'MAPE: ', results['Error'])
        return results


    def random_simulation2(i, mc_simulations, num_buildings, betas):
        building_beta_assignment = np.random.randint(0, mc_simulations-1, size=num_buildings)
        beta_matrix = np.array(betas)[building_beta_assignment]
        Ea_hat = np.einsum('ijk,ij->k', all_buildings, beta_matrix)
        adj = num_buildings/1000
        error = np.mean(np.abs(Ea_hat - adj * Ec[:ubem.modeling_hours]) / (adj * Ec[:ubem.modeling_hours]) * 100)
        results = {'Assignments': building_beta_assignment, 'Error': error}
        print(i, 'MAPE: ', results['Error'])
        return results


    def recreate_simulations(i, betas):
        building_beta_assignment = np.repeat(i, 1000)
        j_range = list(np.arange(1000)) if i == 0 else list(np.arange(1000*(i-1), 1000*i))
        beta_matrix = np.array(betas)[building_beta_assignment]
        Ea_hat = np.einsum('ijk,ij->k', all_buildings[j_range, :, :], beta_matrix)
        adj = 1
        error = np.mean(np.abs(Ea_hat - adj * Ec[:ubem.modeling_hours]) / (adj * Ec[:ubem.modeling_hours]) * 100)
        results = {'Assignments': building_beta_assignment, 'Error': error}
        print(i, 'MAPE: ', results['Error'])
        return results


    def plot_one_beta(betas, beta_num=0):
        from matplotlib import rcParams
        beta_dist1 = [b[beta_num] for b in betas]
        # beta_dist2 = [b[beta_num+25] for b in betas]
        # beta_dist3 = [b[beta_num+50] for b in betas]

        plt.figure(figsize=(10, 5))
        rcParams.update({'font.size': 20})

        plt.hist(beta_dist1)
        # plt.hist(beta_dist2)
        # plt.hist(beta_dist3)

        plt.xlabel(r'$\beta$')
        plt.ylabel('Occurences')
        plt.show()
        # plt.savefig('BetaDistribution.pdf')

    def recreate_simulations2(i, betas, all_buildings):
        building_beta_assignment = np.repeat(i, all_buildings.shape[0])
        j_range = list(np.arange(all_buildings.shape[0]))
        beta_matrix = np.array(betas)[building_beta_assignment]
        Ea_hat = np.einsum('ijk,ij->k', all_buildings[:, :, :], beta_matrix)
        adj = all_buildings.shape[0]/1000
        error = np.mean(np.abs(Ea_hat - adj * Ec[:ubem.modeling_hours]) / (adj * Ec[:ubem.modeling_hours]) * 100)
        results = {'Assignments': building_beta_assignment, 'Error': error}
        print(i, 'MAPE: ', results['Error'])
        return results
    # for i in range(25):
    #     plot_one_beta(betas, beta_num=i)

    # res_test = pickle.load(open('/Users/jonathanroth/PycharmProjects/UBEM/5sims_1000T_500B.obj', 'rb'))
    # [np.mean(er) for er in res_test['error_vec']]

    #%%time
    results2 = []
    for j in range(50):
        print('ITERATION: ', j)
        all_buildings = create_all_buildings(ubem=ubem, total_simulations=2)
        results2.append([recreate_simulations2(i, betas, all_buildings) for i in np.arange(10)])

        # random_simulation2(i, 8, 2000, betas)

    list_of_simulations = [pickle.load(open(os.getcwd() + '/test005_1000_1000_50_' + str(15 + num*50) + '.obj', 'rb'))
                           for num in np.arange(10)]

    # PSO
    betas = np.array([sim[1] for simulations in list_of_simulations for sim in simulations])
    indices = np.array([sim[1] for simulations in list_of_simulations for sim in simulations])
    training_hours = np.arange(1000)
    Ec = np.reshape(ubem.city_electricity_scaled[training_hours], (ubem.modeling_hours,))

    all_buildings = create_all_buildings(ubem=ubem, total_simulations=4)
    cost, pos = pso(ubem=ubem, all_buildings=all_buildings, betas=betas, Ec=Ec, n_particles=30, iters=30,
                    global_pso=False)

