import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import os

os.chdir("../")
from scipy.cluster.vq import kmeans
import myconfig


def main():
    Npyr = 2200 # Всего пирамидных нейронов 159000, по 70 в каждом кластере
    Npyrdeep = Npyrsup = Npyr // 2



    # Hyperparameters
    OUTPLACE_FIRINGRATE_GEN = 'lognormal' # 'normal' or 'constant'
    INPLACE_FIRINGRATE_GEN = 'lognormal' # 'normal' or 'constant'
    PLACESIZE_GEN = 'lognormal' # 'normal' or 'constant'

    TRACK_LENGTH = 400 # cm
    PLACECELLSPROB = 0.5 # Вероятность пирамидного нейрона стать клеткой места в одном лабиринте
    PHASEPRECPROB = 0.5  # Вероятность обнаружить фазовую прецессию у клетки места
    PLACESIZE_MEAN = 20  # см, Средний размер поля места в дорсальном гиппокампе.
    PLACESIZE_STD = 5  # см, Стандартное отклонение размера поля места в дорсальном гиппокампе.

    PEAKFIRING = 8.0
    PEAKFIRING_STD = 0.4
    OUTPLACEFIRING = 0.5
    OUTPLACEFIRING_STD = 0.7


    PLACESIZE_SLOPE_DV = 1.8e-2 # cm (поля места) / mkm (по оси DV)
    THETA_SLOPE_DV = 1.3e-3 # rad / mkm (по оси DV)
    THETA_R_SLOPE_DV = -6.25e-05 # 1 / mkm (по оси DV)
    THETA_R_0 = 0.25

    PHASEPREC_SLOPE_DEEP_0 = 6.0
    PHASEPREC_SLOPE_SUP_0 = 4.0

    PHASEPREC_ONSET_DEEP_0 = 2.79
    PHASEPREC_ONSET_SUP_0 = 3.84

    PHASEPREC_SLOPE_DECREASE_DV = -1.8e-03  # rad / mkm (по оси DV)

    THETA_LOCALPHASE_DEEP = 0.0 # rad
    THETA_LOCALPHASE_SUP = np.pi # rad




    filepath = myconfig.SCRIPTS4PARAMSGENERATION + "CA1_anatomy.csv"

    CA1_flat = pd.read_csv(filepath, header=0)
    StepH = CA1_flat["H"][1] - CA1_flat["H"][0]
    #CA1_flat["H"] = CA1_flat["H"].max() - CA1_flat["H"]
    #print(CA1_flat["H"].size)

    Square_CA1 = StepH * CA1_flat["L"].sum()
    #print("Square of CA1 field =", Square_CA1, "mkm^2")


    StepProxDistDeep = 250 # Square_CA1 / StepH / Npyrdeep # mkm
    StepProxDistSup = 250 #Square_CA1 / StepH / Npyrsup # mkm

    right_bound = 0.5*CA1_flat["L"]
    left_bound = -0.5*CA1_flat["L"]

    pyramidal_cells = []

    for radial_axis in ["deep", "sup"]:

        if radial_axis == "deep":
            radial_axis_pos = 1
            StepProxDist = StepProxDistDeep
            ThetaPhase = THETA_LOCALPHASE_DEEP
            preces_slope0 = PHASEPREC_SLOPE_DEEP_0
            precess_onset0 = PHASEPREC_ONSET_DEEP_0
            Npyrpops = Npyrdeep

        elif radial_axis == "sup":
            radial_axis_pos = -1
            StepProxDist = StepProxDistSup
            ThetaPhase = THETA_LOCALPHASE_SUP
            preces_slope0 = PHASEPREC_SLOPE_SUP_0
            precess_onset0 = PHASEPREC_ONSET_SUP_0
            Npyrpops = Npyrsup


        pyr_coodinates_x = np.empty(shape=0, dtype=np.float64)
        pyr_coodinates_y = np.empty_like(pyr_coodinates_x)

        for slice_idx, l in enumerate(CA1_flat["L"]):
            lb = left_bound[slice_idx] + 0.5 * StepProxDist
            rb = right_bound[slice_idx] - 0.5 * StepProxDist

            tmp_x = np.arange(lb, rb, 0.01 * StepProxDist)
            pyr_coodinates_x = np.append(pyr_coodinates_x, tmp_x)
            pyr_coodinates_y = np.append(pyr_coodinates_y, np.zeros_like(tmp_x) + CA1_flat["H"][slice_idx])

        print("N points for clustering =", pyr_coodinates_x.size)
        points = np.stack([pyr_coodinates_x, pyr_coodinates_y]).transpose()
        selected, _ = kmeans(points, Npyrpops)

        pyr_coodinates_x = selected[:, 0]
        pyr_coodinates_y = selected[:, 1]
        pyr_coodinates_z = np.zeros_like(pyr_coodinates_x) + radial_axis_pos

        for pyrs_x, pyrs_y, pyrs_z in zip(pyr_coodinates_x, pyr_coodinates_y, pyr_coodinates_z):

            if pyrs_y < myconfig.DV_MIN or pyrs_y > myconfig.DV_MAX:
                continue

            if PLACECELLSPROB < np.random.rand():
                center_place_field = np.random.uniform(low=0.0, high=TRACK_LENGTH, size=1)[0]
            else:
                center_place_field = -1000000

            if PHASEPRECPROB < np.random.rand() and center_place_field > 0:
                phase_precession_slope = PHASEPREC_SLOPE_DECREASE_DV * pyrs_y + preces_slope0
            else:
                phase_precession_slope = 0.0

            place_size = (PLACESIZE_SLOPE_DV * pyrs_y + PLACESIZE_MEAN) / 6
            place_size_std = (PLACESIZE_SLOPE_DV * pyrs_y + PLACESIZE_STD) / 6


            if PLACESIZE_GEN == 'lognormal':
                place_size = np.random.lognormal(mean=np.log(place_size), sigma=0.05*place_size_std) # !!!!!!

            elif PLACESIZE_GEN == 'normal':
                place_size = np.random.normal(loc=place_size, scale=place_size_std)

            outplacefiringrate = OUTPLACEFIRING
            if OUTPLACE_FIRINGRATE_GEN == 'lognormal':
                outplacefiringrate =  np.random.lognormal(mean=np.log(outplacefiringrate), sigma=OUTPLACEFIRING_STD)
            elif OUTPLACE_FIRINGRATE_GEN == 'normal':
                outplacefiringrate = np.random.normal(loc=outplacefiringrate, scale=OUTPLACEFIRING_STD)


            inplacefiringrate = PEAKFIRING
            if INPLACE_FIRINGRATE_GEN == 'lognormal':
                inplacefiringrate =  np.random.lognormal(mean=np.log(inplacefiringrate), sigma=PEAKFIRING_STD)
            elif INPLACE_FIRINGRATE_GEN == 'normal':
                inplacefiringrate = np.random.normal(loc=inplacefiringrate, scale=PEAKFIRING_STD)




            pyr_cell = {
                "type" : "CA1 Pyramidal",

                "x_anat" : pyrs_x,
                "y_anat" : pyrs_y,
                "z_anat" : pyrs_z,

                "ThetaFreq" : myconfig.ThetaFreq,

                "OutPlaceFiringRate" : outplacefiringrate,
                "OutPlaceThetaPhase": THETA_SLOPE_DV * pyrs_y + ThetaPhase,  # DV
                "R": THETA_R_SLOPE_DV * pyrs_y + THETA_R_0,


                "InPlacePeakRate" : inplacefiringrate,
                "CenterPlaceField" : float(center_place_field),
                "SigmaPlaceField" : place_size,


                "SlopePhasePrecession" : phase_precession_slope, # DV
                "PrecessionOnset" : THETA_SLOPE_DV * pyrs_y + precess_onset0,

                "MinFiringRate": 0.1,
                "MaxFiringRate": 50.0,

            }

            pyramidal_cells.append(pyr_cell)


    with open(myconfig.STRUCTURESOFNET + "pyramidal_cells.pickle", mode="bw") as file:
        pickle.dump(pyramidal_cells, file)

    return pyr_coodinates_x, pyr_coodinates_y, pyr_coodinates_z, right_bound, left_bound, CA1_flat


if __name__ == "__main__":

    pyr_coodinates_x, pyr_coodinates_y, pyr_coodinates_z, right_bound, left_bound, CA1_flat = main()
    # print(pyr_coodinates_x.size)
    #
    # fig, axes = plt.subplots()
    # axes.plot( right_bound, CA1_flat["H"], color="blue")
    # axes.plot( left_bound, CA1_flat["H"], color="blue")
    #
    # axes.scatter(pyr_coodinates_x, pyr_coodinates_y)
    #
    # plt.show()