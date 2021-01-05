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

    for i in range(len(testOriginData[0])) :
        plt.hist(testOriginData[:,i], histtype = 'step', label = 'origin', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
        plt.hist(testTargetData[:,i], histtype = 'step', label = 'target', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
        plt.hist(testOriginData[:,i], weights = weights, histtype = 'step', label = 'reweighted', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
        plt.legend()
        plt.xlabel(vars_meta[i][0])
        plt.savefig("Plots/{0}/{0}_{1}.png".format(modelName, i))
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
