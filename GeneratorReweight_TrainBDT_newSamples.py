import os
import pickle
import h5py
import numpy as np
import xgboost as xgb

def trainXGB(originh5, originName, targeth5, targetName) :

    fOrigin = h5py.File(originh5, "r")
    fTarget = h5py.File(targeth5, "r")

    nTrainOrigin = len(fOrigin["train_data"])
    nTrainTarget = len(fTarget["train_data"])
    
    trainData = np.concatenate((fOrigin["train_data"], fTarget["train_data"]))
    trainLabels = [[0]]*nTrainOrigin+[[1]]*nTrainTarget

    nTestOrigin = len(fOrigin["test_data"])
    nTestTarget = len(fTarget["test_data"])
    
    testData = np.concatenate((fOrigin["test_data"], fTarget["test_data"]))
    testLabels = [[0]]*nTestOrigin+[[1]]*nTestTarget

    xgb_train_data = xgb.DMatrix(trainData, label = trainLabels)
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

    model = xgb.train(params = params, dtrain = xgb_train_data, num_boost_round = 1000, evals = evals, evals_result = eval_result , verbose_eval = 5)

    modelName = "bdtrw_"+originName+"_to_"+targetName
    
    try :
        os.remove(modelName+".xgb")
    except OSError :
        pass

    model.save_model(modelName+".xgb")

    with open(modelName+".eval", "wb") as f :
        pickle.dump(eval_result, f)

    
def main() :
    trainXGB("argon_GENIEv2.h5", "GENIEv2", "argon_NUWRO.h5", "NUWRO")

if __name__ == "__main__" :
    main()
