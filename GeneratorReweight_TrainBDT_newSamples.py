import os
import sys
import pickle
import h5py
import numpy as np
import xgboost as xgb
import random


def randomSampleLarge(N, frac, chunk) :
    n = int(N/chunk)
    remainder = N%chunk

    print(n, remainder)
    
    ret = []
    
    for i_chunk in range(n) :
        ret += list(np.sort(random.sample(range(i_chunk*chunk, (i_chunk+1)*chunk), int(chunk*frac))))
    if remainder > 0 :
        ret += list(np.sort(random.sample(range(n*chunk, n*chunk+remainder), int(remainder*frac))))
    
    return ret
        

def trainXGB(originh5, originName, targeth5, targetName, outDir, xgbModel) :

    use_frac = 0.5

    fOrigin = h5py.File(originh5, "r")
    fTarget = h5py.File(targeth5, "r")

    nTrainOrigin = len(fOrigin["train_data"])
    nTrainTarget = len(fTarget["train_data"])

    nUseTrainOrigin = int(use_frac*nTrainOrigin)
    nUseTrainTarget = int(use_frac*nTrainTarget)

    print(nTrainOrigin, nTrainTarget)

    trainData = np.concatenate((fOrigin["train_data"][:nUseTrainOrigin], fTarget["train_data"][:nUseTrainTarget]))
    print("Concatenation DONE")
    trainLabels = [[0]]*nUseTrainOrigin+[[1]]*nUseTrainTarget
    print("trainLabels DONE")

    trainOriginRates = np.zeros((2,3))
    trainTargetRates = np.zeros((2,3))
    
    for nubarness in [0, 1] :
        if nubarness == 0 :
            # Get nus
            barnessMaskTrainOrigin = trainData[:nUseTrainOrigin, 0] > 0.5 
            barnessMaskTrainTarget = trainData[nUseTrainOrigin:, 0] > 0.5 
        else :
            # Get nubars
            barnessMaskTrainOrigin = trainData[:nUseTrainOrigin, 0] < 0.5 
            barnessMaskTrainTarget = trainData[nUseTrainOrigin:, 0] < 0.5 
        for nugeneration in [0, 1, 2] :
            trainOriginRates[nubarness][nugeneration] = np.count_nonzero(np.logical_and(barnessMaskTrainOrigin, trainData[:nUseTrainOrigin, 1+nugeneration] > 0.5))
            trainTargetRates[nubarness][nugeneration] = np.count_nonzero(np.logical_and(barnessMaskTrainTarget, trainData[nUseTrainOrigin:, 1+nugeneration] > 0.5))

    print("Nu type rates train origin:")
    print(trainOriginRates)

    print("Nu type rates train target:")
    print(trainTargetRates)
    
    trainScaleFactors = np.divide(trainTargetRates, trainOriginRates)
    trainScaleFactors[trainOriginRates==0] = 1.

    print("Train scale factors")
    print(trainScaleFactors)

    sample_weight = np.ones(nUseTrainOrigin+nUseTrainTarget)
    sample_barness = trainData[:nUseTrainOrigin, 0] > 0.5
    sample_nueness = trainData[:nUseTrainOrigin, 1] > 0.5
    sample_numuness = trainData[:nUseTrainOrigin, 2] > 0.5
    sample_nutauness = trainData[:nUseTrainOrigin, 3] > 0.5

    sample_weight[:nUseTrainOrigin][np.logical_and(sample_barness, sample_nueness)] = trainScaleFactors[0, 0]
    sample_weight[:nUseTrainOrigin][np.logical_and(sample_barness, sample_numuness)] = trainScaleFactors[0, 1]
    sample_weight[:nUseTrainOrigin][np.logical_and(sample_barness, sample_nutauness)] = trainScaleFactors[0, 2]

    sample_weight[:nUseTrainOrigin][np.logical_and(np.logical_not(sample_barness), sample_nueness)] = trainScaleFactors[1, 0]
    sample_weight[:nUseTrainOrigin][np.logical_and(np.logical_not(sample_barness), sample_numuness)] = trainScaleFactors[1, 1]
    sample_weight[:nUseTrainOrigin][np.logical_and(np.logical_not(sample_barness), sample_nutauness)] = trainScaleFactors[1, 2]
    
    del sample_barness, sample_nueness, sample_numuness, sample_nutauness

    xgb_train_data = xgb.DMatrix(trainData, label = trainLabels, weight = sample_weight)
    print("DMatrix DONE")
    del trainData, trainLabels, sample_weight
    print("Deleted numpy arrays")
    
    nTestOrigin = len(fOrigin["test_data"])
    nTestTarget = len(fTarget["test_data"])

    nUseTestOrigin = int(use_frac*nTestOrigin)
    nUseTestTarget = int(use_frac*nTestTarget)

    print(nTestOrigin, nTestTarget)

    testData = np.concatenate((fOrigin["test_data"][:nUseTestOrigin], fTarget["test_data"][:nUseTestTarget]))
    print("Concatenation DONE")
    
    testLabels = [[0]]*nUseTestOrigin+[[1]]*nUseTestTarget
    print("testLabels DONE")

    testOriginRates = np.zeros((2,3))
    testTargetRates = np.zeros((2,3))
    
    for nubarness in [0, 1] :
        if nubarness == 0 :
            # Get nus
            barnessMaskTestOrigin = testData[:nUseTestOrigin, 0] > 0.5 
            barnessMaskTestTarget = testData[nUseTestOrigin:, 0] > 0.5 
        else :
            # Get nubars
            barnessMaskTestOrigin = testData[:nUseTestOrigin, 0] < 0.5 
            barnessMaskTestTarget = testData[nUseTestOrigin:, 0] < 0.5 
        for nugeneration in [0, 1, 2] :
            testOriginRates[nubarness][nugeneration] = np.count_nonzero(np.logical_and(barnessMaskTestOrigin, testData[:nUseTestOrigin, 1+nugeneration] > 0.5))
            testTargetRates[nubarness][nugeneration] = np.count_nonzero(np.logical_and(barnessMaskTestTarget, testData[nUseTestOrigin:, 1+nugeneration] > 0.5))

    

    print("Nu type rates test origin:")
    print(testOriginRates)

    print("Nu type rates test target:")
    print(testTargetRates)
    
    testScaleFactors = np.divide(testTargetRates, testOriginRates)
    testScaleFactors[testOriginRates==0] = 1.

    print("Test scale factors")
    print(testScaleFactors)

    eval_set_weight = np.ones(nUseTestOrigin+nUseTestTarget)
    eval_set_barness = testData[:nUseTestOrigin, 0] > 0.5
    eval_set_nueness = testData[:nUseTestOrigin, 1] > 0.5
    eval_set_numuness = testData[:nUseTestOrigin, 2] > 0.5
    eval_set_nutauness = testData[:nUseTestOrigin, 3] > 0.5

    eval_set_weight[:nUseTestOrigin][np.logical_and(eval_set_barness, eval_set_nueness)] = testScaleFactors[0, 0]
    eval_set_weight[:nUseTestOrigin][np.logical_and(eval_set_barness, eval_set_numuness)] = testScaleFactors[0, 1]
    eval_set_weight[:nUseTestOrigin][np.logical_and(eval_set_barness, eval_set_nutauness)] = testScaleFactors[0, 2]

    eval_set_weight[:nUseTestOrigin][np.logical_and(np.logical_not(eval_set_barness), eval_set_nueness)] = testScaleFactors[1, 0]
    eval_set_weight[:nUseTestOrigin][np.logical_and(np.logical_not(eval_set_barness), eval_set_numuness)] = testScaleFactors[1, 1]
    eval_set_weight[:nUseTestOrigin][np.logical_and(np.logical_not(eval_set_barness), eval_set_nutauness)] = testScaleFactors[1, 2]

    del eval_set_barness, eval_set_nueness, eval_set_numuness, eval_set_nutauness

    xgb_test_data = xgb.DMatrix(testData, label = testLabels, weight = eval_set_weight)
    print("DMatrix DONE")
    del testData, testLabels, eval_set_weight
    print("Deleted numpy arrays")

    print("Number of events:\nTrain {0} + {1}\nTest {2} + {3}".format(nTrainOrigin, nTrainTarget, nTestOrigin, nTestTarget))
    print("Number of used events:\nTrain {0} + {1}\nTest {2} + {3}".format(nUseTrainOrigin, nUseTrainTarget, nUseTestOrigin, nUseTestTarget))

    params = {}
    params['nthread'] = 14
    params["tree_method"] = "auto"
#    params['gpu_id'] = 0
#    params["tree_method"] = "gpu_hist"
    params["objective"] = 'reg:logistic'
#    params["objective"] = 'binary:logitraw'
    params["eta"] = 0.5
    params["max_depth"] = 6
    params["min_child_weight"] = 100
    params["subsample"] = 0.5
    params["lambda"] = 1
    params["colsample_bytree"] = 0.5
    params["colsample_bylevel"] = 0.5
#    params["alpha"] = 5
#    params["lambda"] = 1
#    params["alpha"] = 1
    params["feature_selector"] = "greedy"
    params["update"] = "grow_colmaker,prune,refresh"
    params["refresh_leaf"] = True
    params["process_type"] = "default"
    params["eval_metric"] = "logloss"
    evals = [(xgb_train_data, "train"), (xgb_test_data, "test")]
    eval_result = {}
    model = None

    model = xgb.train(params = params, dtrain = xgb_train_data, num_boost_round = 1000, evals = evals, evals_result = eval_result , verbose_eval = 5, early_stopping_rounds=10, xgb_model = xgbModel)

    modelName = "bdtrw_"+originName+"_to_"+targetName
    
    try :
        os.remove(outDir+"/"+modelName+".xgb")
    except OSError :
        pass

    model.save_model(outDir+"/"+modelName+".xgb")

    with open(outDir+"/"+modelName+".eval", "wb") as f :
        pickle.dump(eval_result, f)

    with open(outDir+"/"+modelName+".params", "wb") as f :
        pickle.dump(params, f)

    
def main() :

    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced_lambda200_eta0.3_colsample0.7"
    originDataset = "/gpfs/scratch/crfernandesv/GeneratorReweight/argon_GENIEv2.h5"
    originName = "GENIEv2"
    targetDataset = "/gpfs/scratch/crfernandesv/GeneratorReweight/argon_NUWRO.h5"
    targetName = "NUWRO"
    xgbModel = None

    if len(sys.argv) >= 6 :
        outDir = sys.argv[1]
        originDataset = sys.argv[2]
        originName = sys.argv[3]
        targetDataset = sys.argv[4]
        targetName = sys.argv[5]

    if len(sys.argv) == 7 :
        xgbModel = sys.argv[6]

    print("Training with following arguments:")
    print("Out dir {0}".format(outDir))
    print("Origin dataset {0}".format(originDataset))
    print("Origin dataset name {0}".format(originName))
    print("Target dataset {0}".format(targetDataset))
    print("Target dataset name {0}".format(targetName))
    print("XGB model {0}".format(xgbModel))
    
    try :
        os.makedirs(outDir)
    except :
        pass
    trainXGB(originDataset, originName, targetDataset, targetName, outDir, xgbModel)

if __name__ == "__main__" :
    main()
