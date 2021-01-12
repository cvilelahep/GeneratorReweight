import os
import pickle
import h5py
import numpy as np
import xgboost as xgb

import shutil

sampledir = "/disk/cvilela/GeneratorReweight/LargeSamples/"
N_MERGE_CHUNKS = 10 # Split in N chunks when merging origin and target data to fit in memory

def trainXGB(originh5, originName, targeth5, targetName) :

    modelName = "bdtrw_"+originName+"_to_"+targetName
    
    # First, create temporary file which will hold train and target data
    print("Copying origin")
    shutil.copy(originh5, sampledir+"/"+modelName+".h5")
    print("DONE")

    print("Merging origin and target train datasets")
    fMerged = h5py.File(sampledir+"/"+modelName+".h5", "r+")
    trainData = fMerged["train_data"]
    
    trainDataTarget = h5py.File(targeth5, "r")["train_data"]
    print("got target train data")
    
    nTrainOrigin = len(trainData)      
    nTrainTarget = len(trainDataTarget)

    chunk_size = int(nTrainTarget/N_MERGE_CHUNKS)
    chunk_remainder = nTrainTarget%N_MERGE_CHUNKS
    
    print("resizing...")
    trainData.resize(nTrainOrigin+nTrainTarget, axis = 0)
    print("resized")
    print("copying in {0} chunks of {1} events + {2}".format(N_MERGE_CHUNKS,chunk_size, chunk_remainder))
    for i_chunk in range(N_MERGE_CHUNKS) :
        print("chunk number {0}...".format(i_chunk))
        trainData[nTrainOrigin+i_chunk*chunk_size:nTrainOrigin+(i_chunk+1)*chunk_size] = trainDataTarget[i_chunk*chunk_size:(i_chunk+1)*chunk_size]
        print("chunk number {0} done".format(i_chunk))
    if chunk_remainder != 0 :
        trainData[-chunk_remainder:] = trainDataTarget[-chunk_remainder:]
    print("copied")
    trainLabels = [[0]]*nTrainOrigin+[[1]]*nTrainTarget
    print("DONE")

    print("Merging origin and target test datasets")
    testData = fMerged["test_data"]
    testDataTarget = h5py.File(targeth5, "r")["test_data"]
    
    nTestOrigin = len(testData) 
    nTestTarget = len(testDataTarget)

    testData.resize(nTestOrigin+nTestTarget, axis = 0)
    testData[-nTestTarget:] = testDataTarget
    testLabels = [[0]]*nTestOrigin+[[1]]*nTestTarget
    print("DONE")

#    xgb_train_data = xgb.DMatrix(trainData, label = trainLabels)
#    xgb_test_data = xgb.DMatrix(testData, label = testLabels)
    
    params = {}
    params['nthread'] = 16
    params["tree_method"] = "auto"
#    params['gpu_id'] = 0
#    params["tree_method"] = "gpu_hist"
    params["objective"] = 'reg:logistic'
    params["eta"] = 0.3
    params["max_depth"] = 6
    params["min_child_weight"] = 100
#    params["subsample"] = 0.5
    params["subsample"] = 0.05
    params["lambda"] = 5
    params["alpha"] = 5
#    params["lambda"] = 1
#    params["alpha"] = 1
    params["feature_selector"] = "greedy"
    params["update"] = "grow_colmaker,prune,refresh"
    params["refresh_leaf"] = True
    params["process_type"] = "default"

#    evals = [(xgb_train_data, "train"), (xgb_test_data, "test")]
#    eval_result = {}
#    model = None

#    model = xgb.train(params = params, dtrain = xgb_train_data, num_boost_round = 1000, evals = evals, evals_result = eval_result , verbose_eval = 5)

    model = xgb.XGBRegressor(n_estimators = 1000,
                            max_depth = params["max_depth"],
                            learning_rate = params["eta"],
                            verbosity = 1,
                            tree_method = params["tree_method"],
                            n_jobs = params['nthread'],
                            min_child_weight = params["min_child_weight"],
                            subsample = params["subsample"])
    model.fit(trainData, trainLabels, eval_set = [(trainData, trainLabels), (testData, testLabels)], eval_metric = 'logloss', verbose = True, early_stopping_rounds = 5 )

    eval_result = model.evals_result()
    
    modelName = "bdtrw_"+originName+"_to_"+targetName
    
    try :
        os.remove(modelName+".xgb")
    except OSError :
        pass

    model.save_model(modelName+".xgb")

    with open(modelName+".eval", "wb") as f :
        pickle.dump(eval_result, f)
    # Clean up
    os.remove(sampledir+"/"+modelName+".h5")
    
def main() :
    trainXGB(sampledir+"/argon_GENIEv2.h5", "GENIEv2", sampledir+"/argon_NUWRO.h5", "NUWRO")

if __name__ == "__main__" :
    main()
