"""
Created on Thu Jan 19 18:25:27 2019

@author: jonathanroth
"""

import numpy as np
import pickle
import pandas as pd
import os
import timeit
import re
import matplotlib.pyplot as plt
from ubem_simulation_pso import UBEM_Simulator, get_simulation_errors, scale_all_doe_datasets, plot_2x2_hourly_load, \
    create_hourly_load, create_one_building_timeseries, plot_all_hourly_loads


def create_all_buildings(ubem, training_hours, sim_num=65):
    a_rand, buildings = ubem.create_a_rand(sim_ind=sim_num).values()
    amx = ubem.create_amx(a_rand=a_rand, training_hours=training_hours)
    prepared_buildings = ubem.create_prepared_buildings(amx=amx,
                                                        a_rand=a_rand,
                                                        buildings=buildings,
                                                        training_hours=training_hours,
                                                        sim_ind=sim_num)
    print('Finished creating prepared_buildings...')
    return prepared_buildings


def calculate_error(betas, Ec, prepared_buildings, sim_num=65):
    beta = betas[sim_num-15, :]
    num_buildings = prepared_buildings.shape[0]
    modeling_hours = prepared_buildings.shape[2]

    # Optimization prediction
    Eaj_hat = np.array([prepared_buildings[j, :, :].T * beta for j in range(num_buildings)])
    Ea_hat = np.sum(Eaj_hat, axis=2)
    Ea_hat = np.sum(Ea_hat, axis=0).reshape(modeling_hours, )

    error = np.abs(Ea_hat - Ec[:modeling_hours]) / (Ec[:modeling_hours]) * 100
    return np.mean(error)


def out_of_sample_errors(ubem, betas, Ec, training_hours):
    error_vec = []
    start = timeit.default_timer()
    for sim_num in np.arange(15, 515):
        prepared_buildings = create_all_buildings(ubem, training_hours, sim_num=sim_num)
        error = calculate_error(betas, Ec, prepared_buildings, sim_num=sim_num)
        end = timeit.default_timer()

        print('sim_num: ', sim_num, ' | Error: ', error, ' | Time: ', start-end)
        error_vec.append(error)

    return error_vec


if __name__ == '__main__':
    starting_hour = 1000
    total_hours = 1000

    ubem = UBEM_Simulator(sample_buildings=1000, modeling_hours=total_hours)  # 6148
    # Extract simulations and data
    list_of_simulations = [pickle.load(open(os.getcwd() + '/Data/test005_1000_1000_50_' + str(15 + num*50) + '.obj', 'rb'))
                           for num in np.arange(10)]

    betas = np.array([sim[1] for simulations in list_of_simulations for sim in simulations])
    indices = np.array([sim[1] for simulations in list_of_simulations for sim in simulations])
    training_hours = np.arange(starting_hour, starting_hour+total_hours)
    Ec = np.reshape(ubem.city_electricity_scaled[training_hours], (ubem.modeling_hours,))
    all_errors = get_simulation_errors()  # UNORDERED!

    # CHECK ERRORS: OUT-OF-SAMPLE
    # all_errors = np.array(all_errors).flatten()
    # prepared_buildings = create_all_buildings(ubem, training_hours, sim_num=77)
    # error = calculate_error(betas, Ec, prepared_buildings, sim_num=77)
    # print(error)

    # RUN 500 TIMES: OUT-OF-SAMPLE ERROR FOR CONVX. SIMULATION
    # error_vec = out_of_sample_errors(ubem, betas, Ec, training_hours)
    # print(np.mean(error_vec))
    pickle.dump(error_vec, open(os.getcwd() + '/Data/of_of_sample_1000hrs.obj', 'wb'))


    # PLOT ONE BUILDING, ALL ENERGY CURVES
    ubem2 = UBEM_Simulator(sample_buildings=1000, modeling_hours=8784)  # 6148
    doe_list = np.array(scale_all_doe_datasets(calculate=False))
    all_buildings = [create_one_building_timeseries(ubem, betas, '1012970023', doe_list, beta_num=i, modeling_hours=8784, ll84=True) for i in np.arange(betas.shape[0])]
    plot_all_hourly_loads(all_buildings, start=3720, end=3720+168)


