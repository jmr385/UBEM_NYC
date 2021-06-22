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
    create_hourly_load, create_one_building_timeseries #, plot_all_hourly_loads


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


def plot_all_hourly_loads(all_buildings, betas, betas_to_plot=2, start=0, end=168, duration=168):

    select_betas = np.array([0, 30, 54, 72, 76, 86, 97, 104, 105, 108, 113])
    for i in np.arange(betas_to_plot):  #betas.shape[0]
        # shift_vec = int(np.rint(np.random.normal(0, 1.5)))
        shift_vec = 0
        plt.plot(np.arange(duration), all_buildings[i]['Total Energy'].iloc[start+shift_vec:end+shift_vec], color='k', alpha=0.1)
        plt.plot(np.arange(duration), all_buildings[i]['Electricity'].iloc[start+shift_vec:end+shift_vec], color='orange', alpha=0.1)
        plt.plot(np.arange(duration), all_buildings[i]['Gas'].iloc[start+shift_vec:end+shift_vec], color='saddlebrown', alpha=0.1)

    plt.ylabel('Energy [kBtu]')
    plt.legend(('Total Energy', 'Electricity', 'Gas'), loc=2, frameon=False)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches(9, 4.5)
    plt.savefig(os.getcwd() + '/Figures/Chrystler_HourlyEnergy_All_' + str(start) + '_.pdf')
    plt.show()

    for i in np.arange(betas_to_plot):  #betas.shape[0]
        # shift_vec = int(np.rint(np.random.normal(0, 1.5)))
        shift_vec = 0
        plt.plot(np.arange(duration), all_buildings[i]['Cooling'].iloc[start+shift_vec:end+shift_vec], color='b', alpha=0.1)
        plt.plot(np.arange(duration), all_buildings[i]['Heating'].iloc[start+shift_vec:end+shift_vec], color='r', alpha=0.1)
        plt.plot(np.arange(duration), all_buildings[i]['GWater_Heating'].iloc[start+shift_vec:end+shift_vec], color='maroon', alpha=0.1)

    plt.legend(('Cooling', 'Heating', 'Water Heating'), loc=2, frameon=False)
    plt.ylabel('Energy [kBtu]')
    plt.xlabel('Hours')
    plt.xticks(np.arange(0, duration, step=24),
               ('12am Sun', '12am Mon', '12am Tue', '12am Wed', '12am Thr', '12am Fri', '12am Sat'),
               rotation=36)
    fig = plt.gcf()
    fig.set_size_inches(9, 5.5)
    plt.savefig(os.getcwd() + '/Figures/Chrystler_HourlyCoolHeat_All_' + str(start) + '_.pdf')
    plt.show()

    return 0


if __name__ == '__main__':
    starting_hour = 1000
    total_hours = 1000

    ubem = UBEM_Simulator(sample_buildings=1000, modeling_hours=total_hours)  # 6148
    # Extract simulations and data
    list_of_simulations = [pickle.load(open(os.getcwd() + '/Data/test005_1000_1000_50_' + str(15 + num*50) + '.obj', 'rb'))
                           for num in np.arange(10)]

    # EXTRACT PARAMETERS AND IMPORTANT DATA
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
    # pickle.dump(error_vec, open(os.getcwd() + '/Data/of_of_sample_1000hrs.obj', 'wb'))

    # CREATE HOURLY DATA FOR ONE BUILDING
    one_building_name = 'CHANGE NAME HERE'
    one_building_bbl_code = '1012970023'  # TODO: change this to bbl code you wish to reconstruct
    doe_list = np.array(scale_all_doe_datasets(calculate=False))
    one_building_hourly = create_one_building_timeseries(ubem=ubem,
                                                         betas=betas,
                                                         bbl=one_building_bbl_code,
                                                         doe_list=doe_list,
                                                         beta_num=288,  # TODO: take one of the 500 betas to construct one profile
                                                         modeling_hours=8784,
                                                         ll84=True)

    # save to csv file
    one_building_hourly.to_csv(os.getcwd() + '/Data/' + one_building_name + '.csv')

    # PLOT ONE BUILDING, ALL ENERGY CURVES
    doe_list = np.array(scale_all_doe_datasets(calculate=False))
    betas_to_plot = 2
    # all_buildings = [create_one_building_timeseries(ubem, betas, '1012970023', doe_list, beta_num=i, modeling_hours=8784, ll84=True) for i in np.arange(betas.shape[0])]
    all_buildings = [create_one_building_timeseries(ubem, betas, '1012970023', doe_list, beta_num=i, modeling_hours=8784, ll84=True) for i in np.arange(betas_to_plot)]
    plot_all_hourly_loads(all_buildings=all_buildings, betas=betas, betas_to_plot=betas_to_plot, start=100, end=100+168, duration=168)


