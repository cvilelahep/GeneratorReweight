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

    use_fraction = 0.005
    
    fOrigin = h5py.File(originh5, "r")
    fTarget = h5py.File(targeth5, "r")

    nTrainOrigin_raw = len(fOrigin["train_data"])
    nTrainTarget_raw = len(fTarget["train_data"])

    indices_train_origin = randomSampleLarge(nTrainOrigin_raw, use_fraction, 10000)
    indices_train_target = randomSampleLarge(nTrainTarget_raw, use_fraction, 10000)

    nTrainOrigin = len(indices_train_origin)
    nTrainTarget = len(indices_train_target)
    
    print(nTrainOrigin_raw, nTrainTarget_raw, nTrainOrigin, nTrainTarget)
        
    trainData = np.concatenate((fOrigin["train_data"][indices_train_origin], fTarget["train_data"][indices_train_target]))

    print("Done concatenating")

    
    del indices_train_origin, indices_train_origin

    trainLabels = [[0]]*nTrainOrigin+[[1]]*nTrainTarget

    print("Got labels")
    
    xgb_train_data = xgb.DMatrix(trainData, label = trainLabels)

    print("Got train data")
    
    exit()
    
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
    
    params = {}
    params['nthread'] = 16
    params["tree_method"] = "auto"
#    params['gpu_id'] = 0
#    params["tree_method"] = "gpu_hist"
    params["objective"] = 'reg:logistic'
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

    evals = [(xgb_train_data, "train"), (xgb_test_data, "test")]
    eval_result = {}
    model = None

    model = xgb.train(params = params, dtrain = xgb_train_data, num_boost_round = 5000, evals = evals, evals_result = eval_result , verbose_eval = 5, early_stopping_rounds=10)

    modelName = "bdtrw_"+originName+"_to_"+targetName
    
    try :
        os.remove(modelName+".xgb")
    except OSError :
        pass

    model.save_model(modelName+".xgb")

    with open(modelName+".eval", "wb") as f :
        pickle.dump(eval_result, f)

    
def main() :
    trainXGB("/disk/cvilela/GeneratorReweight/LargeSamples/argon_GENIEv2.h5", "GENIEv2", "/disk/cvilela/GeneratorReweight/LargeSamples/argon_NUWRO.h5", "NUWRO")

if __name__ == "__main__" :
    main()
