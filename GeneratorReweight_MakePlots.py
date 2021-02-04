import h5py
import numpy as np
import pickle
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import os

manyBins = 100

vars_meta = [ ["isNu", 2, 0, 1, r"$\nu / \bar{\nu}$ flag"],
              ["isNue", 2, 0, 1, r"$\nu_{e}$ flag"],
              ["isNumu", 2, 0, 1, r"$\nu_{\mu}$ flag"],
              ["isNutau", 2, 0, 1, r"$\nu_{\tau}$ flag"],
              ["cc", 2, 0, 1, "CC flag"],
              ["Enu_true", manyBins, 0, 10, "Neutrino energy [GeV]"],
              ["ELep", manyBins, 0, 5, "Lepton energy [GeV]"],
              ["CosLep", manyBins, -1, 1, r"cos$\theta_{\ell}$"],
              ["Q2", manyBins, 0, 10, r"Q^2"],
              ["W", manyBins, 0, 5, r"W [GeV/$c^{2}$]"],
              ["x", manyBins, 0, 1, "x"],
              ["y", manyBins, 0, 1, "y"],
              ["nP", 15, 0, 15, "Number of protons"],
              ["nN", 15, 0, 15, "Number of neutrons"],
              ["nipip", 10, 0, 10, r"Number of $\pi^{+}$"],
              ["nipim", 10, 0, 10, r"Number of $\pi^{-}$"],
              ["nipi0", 10, 0, 10, r"Number of $\pi^{0}$"],
              ["niem", 10, 0, 10, r"Number of EM objects"],
              ["eP", manyBins-1, 1./manyBins , 5, "Total proton kinetic energy"],
              ["eN", manyBins-1, 1./manyBins, 5, "Total neutron kinetic energy"],
              ["ePip", manyBins-1, 1./manyBins, 5, r"Total $\pi^{+}$ kinetic energy"],
              ["ePim", manyBins-1, 1./manyBins, 5, r"Total $\pi^{-}$ kinetic energy"],
              ["ePi0", manyBins-1, 1./manyBins, 5, r"Total $\pi^{0}$ kinetic energy"],
              ["eOther", manyBins-1, 1./manyBins, 5, "Total \"other\" kinetic energy" ] ]
              

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


def platt(x, originMargins, targetMargins, originPreWeights, targetPreWeights) :

    A = x[0]
    B = x[1]

    pOrigin = 1/(1 + np.exp( A * originMargins + B) )
    pTarget = 1/(1 + np.exp( A * targetMargins + B) )

    summand = np.sum(np.multiply(targetPreWeights, np.log(pTarget)))
    summand += np.sum(np.multiply(originPreWeights, np.log(1-pOrigin)))

    ret = -1*summand
    return ret

def makePlots(originh5, originName, targeth5, targetName, outDir, breakdown = True, ntrees = 0, doPlattScaling = False) :

    use_frac = 1.0

    weight_cap = 1000

    modelName = "bdtrw_"+originName+"_to_"+targetName
    try :
        os.makedirs(outDir+"/Plots/{0}".format(modelName))
    except :
        pass
    
    fOrigin = h5py.File(originh5, "r")
    fTarget = h5py.File(targeth5, "r")
    
    nTestOrigin = len(fOrigin["test_data"])
    nTestTarget = len(fTarget["test_data"])

    nUseTestOrigin = int(use_frac*nTestOrigin)
    nUseTestTarget = int(use_frac*nTestTarget)

    testOriginData = fOrigin["test_data"][:nUseTestOrigin]
    testTargetData = fTarget["test_data"][:nUseTestTarget]

    testOriginRates = np.zeros((2,3))
    testTargetRates = np.zeros((2,3))
    
    for nubarness in [0, 1] :
        if nubarness == 0 :
            # Get nus
            barnessMaskTestOrigin = testOriginData[:,0] > 0.5 
            barnessMaskTestTarget = testTargetData[:,0] > 0.5 
        else :
            # Get nubars
            barnessMaskTestOrigin = testOriginData[:, 0] < 0.5 
            barnessMaskTestTarget = testTargetData[:, 0] < 0.5 
        for nugeneration in [0, 1, 2] :
            testOriginRates[nubarness][nugeneration] = np.count_nonzero(np.logical_and(barnessMaskTestOrigin, testOriginData[:, 1+nugeneration] > 0.5))
            testTargetRates[nubarness][nugeneration] = np.count_nonzero(np.logical_and(barnessMaskTestTarget, testTargetData[:, 1+nugeneration] > 0.5))

    print("Nu type rates test origin:")
    print(testOriginRates)

    print("Nu type rates test target:")
    print(testTargetRates)
    
    testScaleFactors = np.divide(testTargetRates, testOriginRates)
    testScaleFactors[testOriginRates==0] = 1.

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

    xgbTestTargetData = xgb.DMatrix(testTargetData)
    xgbTestOriginData = xgb.DMatrix(testOriginData)

    model = xgb.Booster({'nthread' : 24})
    model.load_model(outDir+"/"+modelName+".xgb")

    margins = np.array(model.predict(xgbTestOriginData, output_margin = True, ntree_limit = ntrees))
    marginsTarget = model.predict(xgbTestTargetData, output_margin = True, ntree_limit = ntrees)

    if doPlattScaling :
        print("Doing platt scaling")
        x0 = np.array([-1., 0.])
#        res = minimize(platt, args = (margins, marginsTarget, eval_set_weight, [1.]*len(marginsTarget)), x0 = x0)
        res = minimize(platt, args = (margins, marginsTarget, eval_set_weight, [1.]*len(marginsTarget)), x0 = x0, method = "Nelder-Mead")
        print("RESULTS: A = {0} B = {1}".format(res.x[0], res.x[1]))
        print(res)
        weights = np.exp(-1*(res.x[0]*margins + res.x[1]))
        prob = 1/(1+np.exp(res.x[0]*margins + res.x[1]))
        weightsTarget = np.exp(-1*(res.x[0]*marginsTarget + res.x[1]))
        probTarget = 1/(1+np.exp(res.x[0]*marginsTarget + res.x[1]))
    else :
        weights = np.exp(margins)
        weightsTarget = np.exp(marginsTarget)
        prob = 1/(1+np.exp(-1*margins))
        probTarget = 1/(1+np.exp(-1*marginsTarget))

    weight_norm_before_cap = np.sum(weights)

    weights[weights > weight_cap] = weight_cap

    weight_norm_after_cap = np.sum(weights)
    
    probCapped = prob.copy()
    probCapped[weights > weight_cap] = 1/(1+np.exp(-1*weight_cap))
    
    probTargetCapped = probTarget.copy()
    probTargetCapped[weightsTarget > weight_cap] = 1/(1+np.exp(-1*weight_cap))

    plt.figure()

    plt.hist(weights, bins = 100, range = (0, 100))
    plt.xlabel("Event weight")
    plt.yscale("log")
    plt.savefig(outDir+"/Plots/{0}_eventWeights.png".format(modelName))

    plt.clf()

    
    hOrigin, bins, patches = plt.hist(prob, bins = 600, range = (-0.1, 1.1), histtype = 'step', label = 'Origin', weights = eval_set_weight)
    hOriginCapped, bins, patches = plt.hist(probCapped, bins = 600, range = (-0.1, 1.1), histtype = 'step', label = 'Origin capped', weights = eval_set_weight)
    hTarget, bins, patches = plt.hist(probTarget, bins = 600, range = (-0.1, 1.1), histtype = 'step', label = 'target')
    hTargetCapped, bins, patches = plt.hist(probTargetCapped, bins = 600, range = (-0.1, 1.1), histtype = 'step', label = 'Target capped')
    binCenters = np.add(bins[1:], bins[:-1])/2.
    plt.plot(binCenters, np.add(hOrigin, hTarget))
    plt.legend()
    plt.savefig(outDir+"/Plots/{0}_networkOut.png".format(modelName))
    plt.clf

    plt.figure()
    plt.plot([1,0], [0, 1])
    calib = np.divide(hOrigin, np.add(hOrigin, hTarget))
    calibCapped = np.divide(hOriginCapped, np.add(hOriginCapped, hTargetCapped))
           
    plt.plot(binCenters, calib, label = "No cap on weights")
    plt.plot(binCenters, calibCapped, label = "Capped weights")
    plt.legend()
    plt.xlabel("Target probability")
    plt.ylabel(r"$\frac{Origin}{Origin + Target}$")
    plt.savefig(outDir+"/Plots/{0}_networkCalib.png".format(modelName))
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
    plt.savefig(outDir+"/Plots/{0}_networkCalib_weights.png".format(modelName))
    plt.clf()

    with open(outDir+"/{0}_calibration_data.p".format(modelName), "wb") as fOutCalib :
        pickle.dump({'bins': binCenters, 'calib' : calib, 'calib_capped' : calibCapped, 'binned_weights' : binned_weights, 'expected_weights' : expected_weights}, fOutCalib)

#    weight_cap_scale_factor = weight_norm_before_cap/weight_norm_after_cap
    weight_cap_scale_factor = len(weights)/weight_norm_after_cap

    print("SUMMED WEIGHTS: before cap {0}; after cap {1}; ratio {2}".format(weight_norm_before_cap, weight_norm_after_cap, weight_cap_scale_factor))
    print("Number of weighted events: {0}".format(len(weights)))
    
    weights *= weight_cap_scale_factor

    norm = float(nUseTestOrigin/nUseTestTarget)
    print("RELATIVE NORMALIZATION {0}".format(norm))
    
    fig, axs = plt.subplots(2, sharex = True, gridspec_kw={'hspace' : 0} )

    # Overall plots
    for i in range(len(fTarget["test_data"][0])) :
        valOrigin, bins, _ = axs[0].hist(testOriginData[:,i], histtype = 'step', label = 'origin', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]), weights = eval_set_weight)
        valTarget, _, _ = axs[0].hist(testTargetData[:,i], histtype = 'step', label = 'target', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
        valReweight, _, _ = axs[0].hist(testOriginData[:,i], weights = np.multiply(weights, eval_set_weight), histtype = 'step', label = 'reweighted', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
        axs[0].legend()
        axs[1].step(bins[:-1], np.divide(valOrigin, valTarget), where = 'post')
        axs[1].step(bins[:-1], np.divide(valTarget, valTarget), where = 'post')
        axs[1].step(bins[:-1], np.divide(valReweight, valTarget), where = 'post')
        plt.xlabel(vars_meta[i][4])
        axs[1].set_ylabel("Ratio to target")
        axs[1].set_ylim(0.81, 1.19)
        fig.savefig(outDir+"/Plots/{0}/{0}_{1}.png".format(modelName, i))
        axs[0].clear()
        axs[1].clear()

    if not breakdown :
        return

    # Broken down plots (in energy bins and by neutrino type)
    for e_bin, e_range in enumerate(Enu_slices) :
        for nu in nu_type :
            print(nu, e_bin)
            thisDir = outDir+"/Plots/{0}/Nu_type_{1}_Ebin_{2}/".format(modelName, nu, e_bin)
            try :
                os.makedirs(thisDir)
            except :
                pass
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
                    
                plt.hist(testOriginData[:,i][selOrigin], histtype = 'step', label = 'origin', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]), weights = eval_set_weight[selOrigin])
                plt.hist(testTargetData[:,i][selTarget], histtype = 'step', label = 'target', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
                plt.hist(testOriginData[:,i][selOrigin], weights = np.multiply(weights[selOrigin], eval_set_weight[selOrigin]), histtype = 'step', label = 'reweighted', bins = vars_meta[i][1], range = (vars_meta[i][2], vars_meta[i][3]))
                plt.legend()
                plt.xlabel(vars_meta[i][0])
                plt.savefig("{0}/{1}_{2}.png".format(thisDir, modelName, i))
                plt.clf()

def main() :
    
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_2000trees_balanced"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_2000trees_balanced_500trees"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced_500trees"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced_lambda10"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced_lambda10_eta0.3"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced_lambda10_eta0.3_colsample0.7"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced_lambda200_eta0.3_colsample0.7"
#    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/bdtrw_numuOnly_lessReg_1000trees_balanced_lambda5_eta0.3_colsample0.7"

    platt = False

    originModel = "GENIEv2"
#    targetModel = "GENIEv3_G18_10a_02_11a"
#    ntrees = 150#    ntrees = 125
#    targetModel = "GENIEv3_G18_10b_00_000"
#    ntrees = 90 #ntrees = 250 #ntrees = 100
#    targetModel = "NEUT"
#    ntrees = 90 #ntrees = 100
    targetModel = "NUWRO"
    ntrees = 200
    outDir = "/gpfs/home/crfernandesv/GeneratorReweight/TestProduction_CalibratedNtrees/generator_reweight_"+originModel+"_"+targetModel

    #ntrees = 0
    #ntrees = 100
    #ntrees = 500
    #ntrees = 500
    #ntrees = 250

    makePlots("argon_"+originModel+".h5", originModel, "argon_"+targetModel+".h5", targetModel, outDir, ntrees = ntrees, breakdown = False, doPlattScaling = platt)
    try :
        os.makedirs(outDir+"_validate_dune_FDFHC_14")
    except :
        pass
    for model_file in [".eval", ".xgb", ".params"] :
        try :
            os.symlink(outDir+"/bdtrw_"+originModel+"_to_"+targetModel+model_file, outDir+"_validate_dune_FDFHC_14/"+"/bdtrw_"+originModel+"_to_"+targetModel+model_file)
        except :
            pass
    makePlots("dune_argon_FDFHC_"+originModel+"_14_FDFHC.h5", originModel, "dune_argon_FDFHC_"+targetModel+"_14_FDFHC.h5", targetModel, outDir+"_validate_dune_FDFHC_14", ntrees = ntrees, breakdown = False, doPlattScaling = platt)
    
    try :
        os.makedirs(outDir+"_validate_dune_NDFHC_14")
    except :
        pass
    for model_file in [".eval", ".xgb", ".params"] :
        try :
            os.symlink(outDir+"/bdtrw_"+originModel+"_to_"+targetModel+model_file, outDir+"_validate_dune_NDFHC_14/"+"/bdtrw_"+originModel+"_to_"+targetModel+model_file)
        except :
            pass
    makePlots("dune_argon_NDFHC_"+originModel+"_14_NDFHC.h5", originModel, "dune_argon_NDFHC_"+targetModel+"_14_NDFHC.h5", targetModel, outDir+"_validate_dune_NDFHC_14", ntrees = ntrees, breakdown = False, doPlattScaling = platt)
    
if __name__ == "__main__" :
    main()
