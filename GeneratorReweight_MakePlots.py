import h5py
import numpy as np
import pickle
import xgboost as xgb

import matplotlib.pyplot as plt

import os

manyBins = 100

vars_meta = [ ["isNu", 2, 0, 1],
              ["isNue", 2, 0, 1],
              ["isNumu", 2, 0, 1],
              ["isNutau", 2, 0, 1],
              ["cc", 2, 0, 1],
              ["Enu_true", manyBins, 0, 10],
              ["ELep", manyBins, 0, 5],
              ["CosLep", manyBins, -1, 1],
              ["Q2", manyBins, 0, 10],
              ["W", manyBins, 0, 5],
              ["x", manyBins, 0, 1],
              ["y", manyBins, 0, 1],
              ["nP", 15, 0, 15],
              ["nN", 15, 0, 15],
              ["nipip", 10, 0, 10],
              ["nipim", 10, 0, 10],
              ["nipi0", 10, 0, 10],
              ["niem", 10, 0, 10],
              ["eP", manyBins-1, 1/manyBins , 5],
              ["eN", manyBins-1, 1/manyBins, 5],
              ["ePip", manyBins-1, 1/manyBins, 5],
              ["ePim", manyBins-1, 1/manyBins, 5],
              ["ePi0", manyBins-1, 1/manyBins, 5],
              ["eOther", manyBins-1, 1/manyBins, 5] ]
              

Enu_slices = [[0., 1.],
              [1., 2.],
              [2., 3.],
              [3., 4.],
              [4., 5.],
              [5., 6.],
              [6., 7.],
              [7., 8.],
              [8., 9.],
              [9., 10.]]

nu_type = [12, -12, 14, -14]


def makePlots(originh5, originName, targeth5, targetName) :

    modelName = "bdtrw_"+originName+"_to_"+targetName

    os.makedirs("Plots/{0}".format(modelName))
    
    fOrigin = h5py.File(originh5, "r")
    fTarget = h5py.File(targeth5, "r")
    
    testOriginData = fOrigin["test_data"]
    testTargetData = fTarget["test_data"]

    xgbTestOriginData = xgb.DMatrix(testOriginData[()])
    xgbTestTargetData = xgb.DMatrix(testTargetData[()])

    model = xgb.Booster({'nthread' : 1})
    model.load_model(modelName+".xgb")

    margins = model.predict(xgbTestOriginData, output_margin = True)
    prob = 1/(1+np.exp(-1*margins))

    weights = np.exp(margins)

    marginsTarget = model.predict(xgbTestTargetData, output_margin = True)
    probTarget = 1/(1+np.exp(-1*marginsTarget))

    norm = float(len(testOriginData[:,0]))/len(testTargetData[:,0])
    print("RELATIVE NORMALIZATION {0}".format(norm))
    
    # Overall plots
    for i in range(len(testOriginData[0])) :
        plt.hist(testOriginData[:,i], histtype = 'step', label = 'origin', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
        plt.hist(testTargetData[:,i], weights = [norm]*len(testTargetData[:,0]), histtype = 'step', label = 'target', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
        plt.hist(testOriginData[:,i], weights = weights, histtype = 'step', label = 'reweighted', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
        plt.legend()
        plt.xlabel(vars_meta[i][0])
        plt.savefig("Plots/{0}/{0}_{1}.png".format(modelName, i))
        plt.clf()

    # Broken down plots (in energy bins and by neutrino type)
    for e_bin, e_range in enumerate(Enu_slices) :
        for nu in nu_type :
            print(nu, e_bin)
            thisDir = "Plots/{0}/Nu_type_{1}_Ebin_{2}/".format(modelName, nu, e_bin)
            os.makedirs(thisDir)
            for i in range(len(testOriginData[0])) :
                selOrigin = np.logical_and(testOriginData[:,5] >= e_range[0], testOriginData[:,5] < e_range[1])
                selTarget = np.logical_and(testTargetData[:,5] >= e_range[0], testTargetData[:,5] < e_range[1])
                if nu > 0 :
                    selOrigin = np.logical_and(selOrigin, testOriginData[:,0] > 0.5)
                    selTarget = np.logical_and(selTarget, testTargetData[:,0] > 0.5)
                else :
                    selOrigin = np.logical_and(selOrigin, testOriginData[:,0] < 0.5)
                    selTarget = np.logical_and(selTarget, testTargetData[:,0] < 0.5)

                if abs(nu) == 12 :
                    selOrigin = np.logical_and(selOrigin, testOriginData[:,1] > 0.5)
                    selTarget = np.logical_and(selTarget, testTargetData[:,1] > 0.5)
                elif abs(nu) == 14 :
                    selOrigin = np.logical_and(selOrigin, testOriginData[:,2] > 0.5)
                    selTarget = np.logical_and(selTarget, testTargetData[:,2] > 0.5)
                elif abs(nu) == 16 :
                    selOrigin = np.logical_and(selOrigin, testOriginData[:,3] > 0.5)
                    selTarget = np.logical_and(selTarget, testTargetData[:,3] > 0.5)
                    
                plt.hist(testOriginData[:,i][selOrigin], histtype = 'step', label = 'origin', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
                plt.hist(testTargetData[:,i][selTarget], weights = [norm]*len(testTargetData[:,0][selTarget]), histtype = 'step', label = 'target', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
                plt.hist(testOriginData[:,i][selOrigin], weights = weights[selOrigin], histtype = 'step', label = 'reweighted', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
                plt.legend()
                plt.xlabel(vars_meta[i][0])
                plt.savefig("{0}/{1}_{2}.png".format(thisDir, modelName, i))
                plt.clf()
            
            
    plt.figure()
    hOrigin, bins, patches = plt.hist(prob, bins = 600, range = (-0.1, 1.1), label = 'origin')
    hTarget, bins, patches = plt.hist(probTarget, bins = 600, range = (-0.1, 1.1), label = 'target')
    binCenters = np.add(bins[1:], bins[:-1])/2.
    plt.plot(binCenters, np.add(hOrigin, hTarget))
    plt.legend()
    plt.savefig("Plots/{0}/{0}_networkOut.png".format(modelName))
    plt.clf

    plt.figure()
    plt.plot([1,0], [0, 1])
    calib = np.divide(hOrigin, np.add(hOrigin, hTarget))
    plt.plot(binCenters, calib)
    plt.savefig("Plots/{0}/{0}_networkCalib.png".format(modelName))
    
    plt.figure()
    with open(modelName+".eval", "rb") as fEval :
        evalDict = pickle.load(fEval)

        nTrainEvals = len(evalDict["train"]["rmse"])
        nTestEvals = len(evalDict["test"]["rmse"])

        testInterval = int(nTrainEvals/nTestEvals)
        
        plt.plot(range(nTrainEvals), evalDict["train"]["rmse"], label = "Training set")
        plt.plot(range(0, nTrainEvals, testInterval), evalDict["test"]["rmse"], label = "Test set")
        plt.legend()
        plt.savefig("Plots/{0}/{0}_trainingCurve.png".format(modelName))

def main() :
    makePlots("argon_GENIEv2.h5", "GENIEv2", "argon_NUWRO.h5", "NUWRO")
    
    
if __name__ == "__main__" :
    main()
