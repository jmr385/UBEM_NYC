"""
Created on Thu Jan 19 18:25:27 2019

@author: jonathanroth
"""

import numpy as np
import pickle
import pandas as pd
import os
from ubem_simulation_pso import UBEM_Simulator


def create_hourly_load(ubem, beta_vec, A_building, doe_1a, doe_2a, doe_3a, column_name, energy):
    i_entry = np.nonzero(A_building[0, :])[0]  # Determine the PLUTO class for row j
    hours = doe_1a.shape[0]

    d1 = np.array(doe_1a[column_name] * beta_vec[i_entry]).reshape(hours, 1)
    d2 = np.array(doe_2a[column_name] * beta_vec[i_entry + 25]).reshape(hours, 1)
    d3 = np.array(doe_3a[column_name] * beta_vec[i_entry + 50]).reshape(hours, 1)

    t1 = ubem.temperature * beta_vec[i_entry + 75]
    c1 = ubem.cdd * beta_vec[i_entry + 100]
    b1 = ubem.bday * beta_vec[i_entry + 125]

    final_vec = d1 + d2 + d3 + t1.reshape(hours, 1) + c1.reshape(hours, 1) + b1.reshape(hours, 1)

    return final_vec * np.array(energy).reshape(hours,1)


def create_one_building_timeseries(ubem, betas, bbl, doe_list, beta_num=288, modeling_hours=8784, ll84=False):

    ubem.pluto_export['BBL'] = ubem.pluto_export['BBL'].astype(str)
    building_pluto = ubem.pluto_export.loc[ubem.pluto_export['BBL'] == bbl]
    A_building = ubem.a[building_pluto.index,]
    doe_datasets_needed = np.matmul(A_building, ubem.m)
    doe_datasets_needed_3 = list(np.nonzero(doe_datasets_needed)[1])
    print(doe_datasets_needed_3)
    three_doe_datasets = doe_list[doe_datasets_needed_3]
    energy = ubem.pluto_export.loc[ubem.pluto_export['BBL'] == bbl, 'Energy_kbtu'].values

    if ll84 == True:
        ll84 = pd.read_csv(os.getcwd() + '/Data/LL84.csv')
        ll84['Energy[kBtu]'] = ll84['Site EUI (kBtu/ft²)'] * ll84['DOF Gross Floor Area']
        ll84['BBL'] = ll84['BBL - 10 digits'].astype(str)
        bbl = bbl + '.0'
        energy = ll84.loc[ll84['BBL'] == bbl, 'Energy[kBtu]'].values

    doe_1a = three_doe_datasets[0]
    doe_2a = three_doe_datasets[1]
    doe_3a = three_doe_datasets[2]
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

    beta_vec = betas[beta_num, :]
    building_hourly['Electricity'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='Electricity_scaled',
                                                                  energy=building_hourly['Total Energy'])
    building_hourly['Gas'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='Gas_scaled',
                                                                  energy=building_hourly['Total Energy'])
    building_hourly['Cooling'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='ECooling_scaled',
                                                                  energy=building_hourly['Total Energy'])
    building_hourly['Lights'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='ELights_scaled',
                                                                  energy=building_hourly['Total Energy'])
    building_hourly['Equipment'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='EEquipment_scaled',
                                                                  energy=building_hourly['Total Energy'])
    building_hourly['Gas_Heating'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='GHeating_scaled',
                                                                  energy=building_hourly['Total Energy'])
    building_hourly['Elec_Heating'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='EHeating_scaled',
                                                                  energy=building_hourly['Total Energy'])
    building_hourly['GWater_Heating'] = create_hourly_load(ubem=ubem, beta_vec=beta_vec, A_building=A_chrystler,
                                                                  doe_1a=doe_1a, doe_2a=doe_2a, doe_3a=doe_3a,
                                                                  column_name='GWaterheat_scaled',
                                                                  energy=building_hourly['Total Energy'])
    building_hourly['Heating'] = building_hourly['Gas_Heating'] + \
                                 building_hourly['Elec_Heating'] + \
                                 building_hourly['GWater_Heating']

    building_hourly['Cooling'] = np.abs(building_hourly['Cooling'])

    return building_hourly


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


if __name__ == '__main__':
    total_hours = 1000
    ubem = UBEM_Simulator(sample_buildings=1000, modeling_hours=total_hours)  # 6148
    # Extract simulations
    list_of_simulations = [pickle.load(open(os.getcwd() + '/Data/test005_1000_1000_50_' + str(15 + num*50) + '.obj', 'rb'))
                           for num in np.arange(10)]

    # PSO
    betas = np.array([sim[1] for simulations in list_of_simulations for sim in simulations])
    indices = np.array([sim[1] for simulations in list_of_simulations for sim in simulations])
    training_hours = np.arange(total_hours)
    Ec = np.reshape(ubem.city_electricity_scaled[training_hours], (ubem.modeling_hours,))
