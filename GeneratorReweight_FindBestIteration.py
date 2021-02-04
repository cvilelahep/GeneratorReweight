import os
import pickle
import h5py
import numpy as np
import xgboost as xgb
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg"

class dataSet :
    def __init__ (self, file_name, test, use_frac) :
        f = h5py.File(file_name, "r")

        ds = None
        if test :
            ds = f["test_data"]
        else :
            ds = f["train_data"]
        
        n = len(ds)
        
        self.nUse = int(use_frac*n)
        self.data = ds[:self.nUse]
        self.xgbData = xgb.DMatrix(self.data)


def getTestScaleFactors(originDataSet, targetDataSet) :
    
    testOriginRates = np.zeros((2,3))
    testTargetRates = np.zeros((2,3))
    
    for nubarness in [0, 1] :
        if nubarness == 0 :
            # Get nus
            barnessMaskTestOrigin = originDataSet.data[:,0] > 0.5 
            barnessMaskTestTarget = targetDataSet.data[:,0] > 0.5 
        else :
            # Get nubars
            barnessMaskTestOrigin = originDataSet.data[:, 0] < 0.5 
            barnessMaskTestTarget = targetDataSet.data[:, 0] < 0.5 
        for nugeneration in [0, 1, 2] :
            testOriginRates[nubarness][nugeneration] = np.count_nonzero(np.logical_and(barnessMaskTestOrigin, originDataSet.data[:, 1+nugeneration] > 0.5))
            testTargetRates[nubarness][nugeneration] = np.count_nonzero(np.logical_and(barnessMaskTestTarget, targetDataSet.data[:, 1+nugeneration] > 0.5))

    print("Nu type rates test origin:")
    print(testOriginRates)

    print("Nu type rates test target:")
    print(testTargetRates)
    
    testScaleFactors = np.divide(testTargetRates, testOriginRates)
    testScaleFactors[testOriginRates==0] = 1.

    return testScaleFactors

def findBestIteration(originh5, originName, targeth5, targetName, outDir, ntree_limit = 0, originDataset = None, targetDataset = None, testScaleFactors = None) :
    use_frac = 1.0

    if originDataset == None :
        originDataset = dataSet(originh5, True, 1.0)
        
    if targetDataset == None :
        targetDataset = dataSet(targeth5, True, 1.0)

    fOrigin = h5py.File(originh5, "r")
    fTarget = h5py.File(targeth5, "r")

    nTestOrigin = len(fOrigin["test_data"])
    nTestTarget = len(fTarget["test_data"])

    nUseTestOrigin = originDataset.nUse
    nUseTestTarget = targetDataset.nUse

    testOriginData = originDataset.data
    testTargetData = targetDataset.data

    xgbTestTargetData = targetDataset.xgbData
    xgbTestOriginData = originDataset.xgbData

    print("Got DMatrices")

    modelName = "bdtrw_"+originName+"_to_"+targetName

    model = xgb.Booster({'nthread' : 14})
    model.load_model(outDir+"/"+modelName+".xgb")

    print("Got the model")

    margins = model.predict(xgbTestOriginData, output_margin = True, ntree_limit = ntree_limit)
    prob = 1/(1+np.exp(-1*margins))
    weights = np.exp(margins)
    
    print("Got the origin weights")

    weight_cap = 1000

    probCapped = prob.copy()
    probCapped[weights > weight_cap] = 1/(1+np.exp(-1*weight_cap))

    marginsTarget = model.predict(xgbTestTargetData, output_margin = True, ntree_limit = ntree_limit)
    probTarget = 1/(1+np.exp(-1*marginsTarget))
    weightsTarget = np.exp(marginsTarget)

    print("Got the target weights")

    try :
        len(testScaleFactors)
    except TypeError :
        testScaleFactors = getTestScaleFactors(originDataset, targetDataset)
    
    print("Test scale factors")
    print(testScaleFactors)

    eval_set_weight = np.ones(nUseTestOrigin)
    eval_set_barness = testOriginData[:, 0] > 0.5
    eval_set_nueness = testOriginData[:, 1] > 0.5
    eval_set_numuness = testOriginData[:, 2] > 0.5
    eval_set_nutauness = testOriginData[:, 3] > 0.5

    eval_set_weight[np.logical_and(eval_set_barness, eval_set_nueness)] = testScaleFactors[0, 0]
    eval_set_weight[np.logical_and(eval_set_barness, eval_set_numuness)] = testScaleFactors[0, 1]
    eval_set_weight[np.logical_and(eval_set_barness, eval_set_nutauness)] = testScaleFactors[0, 2]

    eval_set_weight[np.logical_and(np.logical_not(eval_set_barness), eval_set_nueness)] = testScaleFactors[1, 0]
    eval_set_weight[np.logical_and(np.logical_not(eval_set_barness), eval_set_numuness)] = testScaleFactors[1, 1]
    eval_set_weight[np.logical_and(np.logical_not(eval_set_barness), eval_set_nutauness)] = testScaleFactors[1, 2]

    del eval_set_barness, eval_set_nueness, eval_set_numuness, eval_set_nutauness
    
    probTargetCapped = probTarget.copy()
    probTargetCapped[weightsTarget > weight_cap] = 1/(1+np.exp(-1*weight_cap))
    
    plt.figure()
    hOrigin, bins, patches = plt.hist(prob, bins = 600, range = (-0.1, 1.1), histtype = 'step', label = 'origin', weights = eval_set_weight)
    hOriginCapped, bins, patches = plt.hist(probCapped, bins = 600, range = (-0.1, 1.1), histtype = 'step', label = 'origin capped', weights = eval_set_weight)
    hTarget, bins, patches = plt.hist(probTarget, bins = 600, range = (-0.1, 1.1), histtype = 'step', label = 'target')
    hTargetCapped, bins, patches = plt.hist(probTargetCapped, bins = 600, range = (-0.1, 1.1), histtype = 'step', label = 'target capped')
    binCenters = np.add(bins[1:], bins[:-1])/2.
    plt.plot(binCenters, np.add(hOrigin, hTarget))
    plt.legend()
    plt.savefig(outDir+"/{0}_networkOut_{1}.png".format(modelName, ntree_limit))
    plt.clf


    plt.figure()
    plt.plot([1,0], [0, 1])
    calib = np.divide(hOrigin, np.add(hOrigin, hTarget))
    calibCapped = np.divide(hOriginCapped, np.add(hOriginCapped, hTargetCapped))
    
    area = 0.
    print("CALIB", calib)
    print("BIN CENTERS", binCenters)
    for i in range(len(calib)) :
        print("LOOPING", calib[i], binCenters[-i], calib[i]-binCenters[-i], abs(calib[i]-binCenters[-i]))
        if not np.isnan(calib[i]) :
            area += abs(calib[i]-binCenters[-i])
       
    plt.plot(binCenters, calib, label = "calib")
    plt.plot(binCenters, calibCapped, label = "calib capped")
    plt.legend()
    plt.xlabel("Target probability")
    plt.ylabel(r"$\frac{Origin}{Origin + Target}$")
    plt.savefig(outDir+"/{0}_networkCalib_{1}.png".format(modelName, ntree_limit))
    plt.clf()

    probBoundary = np.logical_and(binCenters > 0, binCenters < 1)

    expected_weights = binCenters[probBoundary]/(1.-binCenters[probBoundary])
    binned_weights = 1./calibCapped[probBoundary] - 1.

    plt.plot(binCenters[probBoundary], expected_weights, label = "Expected weights")
    plt.plot(binCenters[probBoundary], binned_weights, label = "Predicted weights")
    plt.legend()
    plt.xlabel("Target probability")
    plt.ylabel("Weight")
    plt.yscale('log')
    plt.savefig(outDir+"/{0}_networkCalib_weights_{1}.png".format(modelName, ntree_limit))
    plt.clf()

    areaWeights = 0.
    for i in range(len(binned_weights)) :
        areaWeights += abs( binned_weights[i] - expected_weights[i])

    print(ntree_limit, area, areaWeights)

    plt.hist(weights, histtype = 'step', label = "origin", bins = 200, range = (0, 1000))
    plt.hist(weightsTarget, histtype = 'step', label = "target", bins = 200, range = (0, 1000))
    plt.yscale("log")
    plt.legend()
    plt.savefig(outDir+"/{0}_weightdistros_{1}.png".format(modelName, ntree_limit))

    return area, areaWeights
    
def plotTrainCurve(originName, targetName, outDir) :
    modelName = "bdtrw_"+originName+"_to_"+targetName
    
    with open(outDir+"/"+modelName+".eval", "rb") as f :
        eval_result = pickle.load(f)

    fig = plt.figure()
    plt.plot(eval_result["test"]["logloss"])
    plt.plot(eval_result["train"]["logloss"])
    fig.savefig(outDir+"/trainCurve.png")
    fig.savefig(outDir+"/trainCurve.pdf")

def runEverything(dataOrigin, nameOrigin, dataTarget, nameTarget, outDir) :
    plotTrainCurve(nameOrigin, nameTarget, outDir)

    originDataset = dataSet(dataOrigin, True, 1.0)
    targetDataset = dataSet(dataTarget, True, 1.0)

    testScaleFactors = getTestScaleFactors(originDataset, targetDataset)

    findBestIteration(dataOrigin, nameOrigin, dataTarget, nameTarget, outDir, originDataset = originDataset, targetDataset = targetDataset, testScaleFactors = testScaleFactors)

#    ntrees = [10, 25, 50, 75, 100, 125, 150, 200, 250, 500, 750, 1000, 1500, 2000]
    ntrees = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 225, 250, 500]

    areas = []
    areasWeights = []
    for n in ntrees :
        area, areaWeights = findBestIteration(dataOrigin, nameOrigin, dataTarget, nameTarget, outDir, n, originDataset = originDataset, targetDataset = targetDataset, testScaleFactors = testScaleFactors)
        areas.append(area)
        areasWeights.append(areaWeights)

    fig = plt.figure()

    plt.plot(ntrees, areas)
    fig.savefig(outDir+"/areaVsTrees.png")
    plt.clf()
    plt.plot(ntrees, areasWeights)
    fig.savefig(outDir+"/areaWeightsVsTrees.png")

    with open(outDir+"area_vs_trees.p", "wb") as f :
        pickle.dump({'ntrees' : ntrees, 'areas' : areas, 'areasWeights' : areasWeights}, f)


    


def main() :

#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced_lambda10"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_2000trees_balanced"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced_lambda10_eta0.3_colsample0.7"

    originName = "GENIEv2"
#    targetName = "NUWRO"
#    targetName = "NEUT"
    targetName = "GENIEv3_G18_10a_02_11a"
#    targetName = "GENIEv3_G18_10b_00_000"
    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/TestProduction_2000/generator_reweight_"+originName+"_"+targetName+"/"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/TestProduction/generator_reweight_"+originName+"_"+targetName+"/"

    runEverything("/gpfs/home/crfernandesv/GeneratorReweight/argon_"+originName+".h5", originName, "/gpfs/home/crfernandesv/GeneratorReweight/argon_"+targetName+".h5", targetName, outDir)
    


if __name__ == "__main__" :
    main()
