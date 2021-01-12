import os
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
        

def trainXGB(originh5, originName, targeth5, targetName) :

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
    #trainLabels = [[0]]*nTrainOrigin+[[1]]*nTrainTarget
    trainLabels = [[0]]*nUseTrainOrigin+[[1]]*nUseTrainTarget
    print("trainLabels DONE")
    xgb_train_data = xgb.DMatrix(trainData, label = trainLabels)
    print("DMatrix DONE")
    del trainData, trainLabels
    print("Deleted numpy arrays")

    print("Got labels")
    
    xgb_train_data = xgb.DMatrix(trainData, label = trainLabels)

    print("Got train data")
    
    nTestOrigin_raw = len(fOrigin["test_data"])
    nTestTarget_raw = len(fTarget["test_data"])

    nTestOrigin = int(use_fraction*nTestOrigin_raw)
    nTestTarget = int(use_fraction*nTestTarget_raw)

    indices_test_origin = np.random.randint(low = 0, high = nTestOrigin_raw, size = nTestOrigin)
    indices_test_origin.sort()
    indices_test_target = np.random.randint(low = 0, high = nTestTarget_raw, size = nTestOrigin)
    indices_test_target.sort()
    
    print("Number of events:\nTrain {0} + {1}\nTest {2} + {3}".format(nTrainOrigin, nTrainTarget, nTestOrigin, nTestTarget))
    
    testData = np.concatenate((fOrigin["test_data"][indices_test_origin], fTarget["test_data"][indices_test_target]))
    testLabels = [[0]]*nTestOrigin+[[1]]*nTestTarget

    xgb_test_data = xgb.DMatrix(testData, label = testLabels)
    del testData, testLabels
    
    params = {}
    params['nthread'] = 12
    params["tree_method"] = "auto"
#    params['gpu_id'] = 0
#    params["tree_method"] = "gpu_hist"
    params["objective"] = 'reg:logistic'
#    params["objective"] = 'binary:logitraw'
    params["eta"] = 0.3
    params["max_depth"] = 6
    params["min_child_weight"] = 100
    params["subsample"] = 0.5
    params["lambda"] = 5
    params["alpha"] = 5
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

    model = xgb.train(params = params, dtrain = xgb_train_data, num_boost_round = 1000, evals = evals, evals_result = eval_result , verbose_eval = 5, early_stopping_rounds=10)

    modelName = "bdtrw_"+originName+"_to_"+targetName
    
    try :
        os.remove(modelName+".xgb")
    except OSError :
        pass

    model.save_model(modelName+".xgb")

    with open(modelName+".eval", "wb") as f :
        pickle.dump(eval_result, f)

    
def main() :
    trainXGB("/gpfs/scratch/crfernandesv/GeneratorReweight/argon_GENIEv2.h5", "GENIEv2", "/gpfs/scratch/crfernandesv/GeneratorReweight/argon_NUWRO.h5", "NUWRO")

if __name__ == "__main__" :
    main()
