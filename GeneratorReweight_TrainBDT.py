import sys
import os
import argparse
import math

import pickle
import numpy as np
import xgboost as xgb

def main() :
    maxRows = 50000000
#    maxRows = 50000
    trainFraction = 0.8

    storeName = '/storage/shared/cvilela/GeneratorReweight/GeneratorRW.h5'

    store = pd.HDFStore(storeName, 'r')

    # Populate dict of existing samples:
    samples = {}
    samples["CAF_GENIE"] = {}
    samples["NuWro"] = {}
    for generator, sample in samples.iteritems() :
        sample["FHC"] = {"ND" : [], "FD" : [] }
        sample["RHC"] = {"ND" : [], "FD" : [] }
        sample["FHC"]["ND"] = ["numu"]
        sample["FHC"]["FD"] = ["numu", "nue"]
        sample["RHC"]["ND"] = ["numu", "numubar"]
        sample["RHC"]["FD"] = ["numu", "numubar", "nue", "nuebar"]

    samples_for_df = { "generator" : [],
                       "runmode"   : [],
                       "detector"  : [],
                       "species"   : [],
                       "nrows"     : [] }
    
    for generator, sample in samples.iteritems() :
        for runmode, detectors in sample.iteritems() :
            for detector, speciesList in detectors.iteritems() :
                for species in speciesList :
                    samples_for_df["generator"].append(generator)
                    samples_for_df["runmode"].append(runmode)
                    samples_for_df["detector"].append(detector)
                    samples_for_df["species"].append(species)
                    samples_for_df["nrows"].append(store.get_storer(generator+'_'+species+'_'+detector+'_'+runmode+'_train').nrows)


    chunkSizes = pd.DataFrame.from_dict(samples_for_df)

    totalRows = chunkSizes["nrows"].sum()
    
    N_chunks = int(math.ceil(float(totalRows)/maxRows))
    
    chunkSizes["chunksize"] = (chunkSizes["nrows"]/N_chunks).apply(np.floor)
                    
    
    print "RUNNING", N_chunks, "ITERATIONS"

    params = {}
    params['nthread'] = 1
    params["tree_method"] = "auto"
    params["objective"] = 'reg:logistic'
    params["eta"] = 0.3
    params["max_depth"] = 6
    params["min_child_weight"] = 100
    params["subsample"] = 0.5
    params["lambda"] = 5
    params["alpha"] = 5
    params["feature_selector"] = "greedy"
    params["update"] = "grow_colmaker,prune,refresh"
    params["refresh_leaf"] = True
    params["process_type"] = "default"

    eval_result = []
    model = None
        
#    for n_iteration in range(N_chunks) :
    for n_iteration in [0] :
        print "START ITERATION", n_iteration
        eval_result.append({})

        dflistTest = []
        dflistTrain = []

        for generator, sample in samples.iteritems() :
            for runmode, detectors in sample.iteritems() :
                for detector, speciesList in detectors.iteritems() :
                    for species in speciesList :
                        print generator, runmode, detector, species
                        chunksize = chunkSizes.loc[(chunkSizes["generator"] == generator) & (chunkSizes["runmode"] == runmode) & (chunkSizes["detector"] == detector) & (chunkSizes["species"] == species)]['chunksize'].iloc[0]

                        chunksizeTrain = int(trainFraction*chunksize)
                        chunksizeTest = int((1-trainFraction)*chunksize)

                        tempdfTrain = pd.read_hdf(store, generator+'_'+species+'_'+detector+'_'+runmode+'_train', start = int(n_iteration*chunksizeTrain), stop = int((n_iteration+1)*chunksizeTrain))
                        tempdfTest = pd.read_hdf(store, generator+'_'+species+'_'+detector+'_'+runmode+'_test', start = int(n_iteration*chunksizeTest), stop = int((n_iteration+1)*chunksizeTest))

                        dflistTrain.append(tempdfTrain)
                        dflistTest.append(tempdfTest)
                    
        dfTrain = pd.concat(dflistTrain, sort = False)
        del dflistTrain[:]
        dfTest = pd.concat(dflistTest, sort = False)
        del dflistTest[:]

        # Calculate weights to normalize the NuWro inputs to the CAFAna inputs in terms of flux, detector and run-mode. Don't want BDT to learn to tell between the models from these parameters!!!
        
        # Use the on-hot encoded values for this.
        print "isND", "isFHC", "isMu", "isNu"
        for det in [0,1] :
            for runmode in [0,1] :
                for flavour in [0,1] :
                    for barness in [0,1] :
                        print det, runmode, flavour, barness
                        nNuWro = dfTrain.loc[(dfTrain['label'] == 1) & (dfTrain['isND'] == det) & (dfTrain['isFHC'] == runmode) & (dfTrain['isMu'] == flavour) & (dfTrain['isNu'] == barness)]['weight'].sum()
                        nGENIE =  dfTrain.loc[(dfTrain['label'] == 0) & (dfTrain['isND'] == det) & (dfTrain['isFHC'] == runmode) & (dfTrain['isMu'] == flavour) & (dfTrain['isNu'] == barness)]['weight'].sum()

                        if nNuWro :
                            dfTrain.loc[(dfTrain['label'] == 1) & (dfTrain['isND'] == det) & (dfTrain['isFHC'] == runmode) & (dfTrain['isMu'] == flavour) & (dfTrain['isNu'] == barness), 'weight'] *= nGENIE/nNuWro
                            dfTest.loc[(dfTest['label'] == 1) & (dfTest['isND'] == det) & (dfTest['isFHC'] == runmode) & (dfTest['isMu'] == flavour) & (dfTest['isNu'] == barness), 'weight'] *= nGENIE/nNuWro
                        
        labelsTrain = dfTrain["label"]
        dfTrain.drop(columns = ["label"], inplace = True)
        weightsTrain = dfTrain["weight"]
        dfTrain.drop(columns = ["weight"], inplace = True)

        dataTrain = xgb.DMatrix(dfTrain, label=labelsTrain, weight = weightsTrain)

        del dfTrain, labelsTrain, weightsTrain

        labelsTest = dfTest["label"]
        dfTest.drop(columns = ["label"], inplace = True)
        weightsTest = dfTest["weight"]
        dfTest.drop(columns = ["weight"], inplace = True)

        dataTest = xgb.DMatrix(dfTest, label=labelsTest, weight = weightsTest)

        del dfTest, labelsTest, weightsTest

        evals = [(dataTrain, "train"), (dataTest, "test")]
        testEvalsResult = {}

        if not model :
            # BDT hyperparameters
            model = xgb.train(params = params, dtrain = dataTrain, num_boost_round = 500, evals = evals, evals_result = eval_result[-1] , verbose_eval = 5)
        else :
            # BDT hyperparameters
            model = xgb.train(params = params, dtrain = dataTrain, num_boost_round = 500, evals = evals, evals_result = eval_result[-1] , verbose_eval = 5, xgb_model = model)
    
    try :
        os.remove("/storage/shared/cvilela/GeneratorReweight/GeneratorRW.xgb")
    except OSError :
        pass

    if model :
        model.save_model("/storage/shared/cvilela/GeneratorReweight/GeneratorRW.xgb")
    pickle.dump(eval_result, open("/storage/shared/cvilela/GeneratorReweight/GeneratorRW.eval.p", "w"))

if __name__ == "__main__" :
    main()
