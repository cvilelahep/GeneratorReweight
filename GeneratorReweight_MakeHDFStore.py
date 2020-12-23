import os
import glob

#import pandas as pd
import numpy as np
import uproot4
import h5py
import awkward1 as ak

m = {}
m["P"] = 0.93827
m["N"] = 0.93957
m["piC"] = 0.13957
m["pi0"] = 0.13498
m["kC"] = 0.49368
m["k0"] = 0.49764

samples = {}

sampledir = "/disk/cvilela/GeneratorReweight/Samples/"

samples["argon_GENIEv2"] = sampledir+"nuisflat_argon_GENIEv2.root"
samples["argon_GENIEv3_G18_10a_02_11a"] = sampledir+"argon_GENIEv3_G18_10a_02_11a"
samples["argon_GENIEv3_G18_10b_00_000"] = sampledir+"nuisflat_argon_GENIEv3_G18_10b_00_000.root"
samples["argon_NEUT"] = sampledir+"nuisflat_argon_NEUT.root"
samples["argon_NUWRO"] = sampledir+"nuisflat_argon_NUWRO.root"

CAF_fNames = {}
CAF_fNames["ND"] = {"FHC" : "/storage/shared/cvilela/GeneratorReweight/ND_FHC_FV_[0-3][0-9].root",
#                    "RHC" : "/storage/shared/cvilela/GeneratorReweight/ND_RHC_FV_[0-3][0-9].root"}
                    "RHC" : "/storage/shared/cvilela/GeneratorReweight/ND_RHC_FV_0[0-9].root"} # For now use 1/4 of RHC stats

CAF_fNames["FD_nonswap"] = {"FHC" : "/storage/shared/cvilela/GeneratorReweight/FD_FHC_nonswap.root",
                            "RHC" : "/storage/shared/cvilela/GeneratorReweight/FD_RHC_nonswap.root"}

CAF_fNames["FD_nueswap"] = {"FHC" : "/storage/shared/cvilela/GeneratorReweight/FD_FHC_nueswap.root",
                            "RHC" : "/storage/shared/cvilela/GeneratorReweight/FD_RHC_nueswap.root"}

NuWro_fName = "/storage/shared/cvilela/GeneratorReweight/NuWroCAFTruth.root"

NuWro_files = glob.glob(NuWro_fName)

def nuisflatToH5(fNameNuis, fNameh5, trainFraction) :

    if os.path.exists(fNameh5):
        os.remove(fNameh5)

    with h5py.File(fNameh5, 'w') as hf:
        for i, fName in enumerate(glob.glob(fNameNuis)) :
            with uproot4.open(fNameNuis+":GenericVectors_flat_VARS") as tree :
                treeArr = tree.arrays(nuisReadVars)

                Lepmask = (treeArr["pdg"] == 11) + (treeArr["pdg"] == -11) + (treeArr["pdg"] == 13) + (treeArr["pdg"] == -13) + (treeArr["pdg"] == 15) + (treeArr["pdg"] == -15)
                Numask = (treeArr["pdg"] == 12) + (treeArr["pdg"] == -12) + (treeArr["pdg"] == 14) + (treeArr["pdg"] == -14) + (treeArr["pdg"] == 16) + (treeArr["pdg"] == -16) 
                Pmask = treeArr["pdg"] == 2212
                Nmask = treeArr["pdg"] == 2112
                Pipmask = treeArr["pdg"] == 211
                Pimmask = treeArr["pdg"] == -211
                Pi0mask = treeArr["pdg"] == 111
                Kpmask = treeArr["pdg"] == 321
                Kmmask = treeArr["pdg"] == -321
                K0mask = (treeArr["pdg"] == 311) + (treeArr["pdg"] == -311) + (treeArr["pdg"] == 130) + (treeArr["pdg"] == 310)
                EMmask = treeArr["pdg"] == 22

                othermask = (Numask + Lepmask + Pmask + Nmask + Pipmask + Pimmask + Pi0mask + Kpmask + Kmmask + K0mask + EMmask) == False
                
                treeArr["nP"] = ak.count_nonzero(Pmask, axis = 1)
                treeArr["nN"] = ak.count_nonzero(Nmask, axis = 1)
                treeArr["nipip"] = ak.count_nonzero(Pipmask, axis = 1)
                treeArr["nipim"] = ak.count_nonzero(Pimmask == -211, axis = 1)
                treeArr["nipi0"] = ak.count_nonzero(Pi0mask == 111, axis = 1)
                treeArr["nikp"] = ak.count_nonzero(Kpmask == 321, axis = 1)
                treeArr["nikm"] = ak.count_nonzero(Kmmask == -321, axis = 1)
                treeArr["nik0"] = ak.count_nonzero(K0mask, axis = 1)
                treeArr["niem"] = ak.count_nonzero(EMmask, axis = 1)

                treeArr["eP"] = ak.sum(treeArr["E"][Pmask], axis = 1) - treeArr["nP"]*m["P"]
                treeArr["eN"] = ak.sum(treeArr["E"][Nmask], axis = 1) - treeArr["nN"]*m["N"]
                treeArr["ePip"] = ak.sum(treeArr["E"][Pipmask], axis = 1) - treeArr["nipip"]*m["piC"]
                treeArr["ePim"] = ak.sum(treeArr["E"][Pimmask], axis = 1) - treeArr["nipim"]*m["piC"]
                treeArr["ePi0"] = ak.sum(treeArr["E"][Pi0mask], axis = 1) - treeArr["nipi0"]*m["pi0"]

                treeArr["eOther"] = ak.sum(treeArr["E"][othermask] - (treeArr["E"][othermask]**2-treeArr["px"][othermask]**2-treeArr["py"][othermask]**2-treeArr["pz"][othermask]**2)**0.5, axis = 1)

                print(treeArr[["eP", "eN", "nN"]])
                print(ak.to_numpy(treeArr[["cc", "Enu_true", "ELep", "CosLep", "Q2", "W", "x", "y", "nP", "nN", "nipip", "nipim", "nipi0", "niem", "eP", "eN", "ePip", "ePim", "ePi0", "eOther"]]))
                
                exit()
                
#                if i == 0 :

def main() :

    
    nuisflatToH5(samples["argon_GENIEv2"], "test.h5", 0.8)
    exit()
    """
    storeName = '/storage/shared/cvilela/GeneratorReweight/GeneratorRW.h5'
    trainFraction = 0.8

    try :
        os.remove(storeName) 
    except OSError :
        pass

    store = pd.HDFStore(storeName)

    varListNuWro = []
    varListCAF = []
    
    for var in rwVars :
        varListCAF.append(var)
        varListNuWro.append(var)
        
    for cvwgtVar in cvWeightList :
            varListCAF.append(cvwgtVar)

    # Get NuWro samples
    for f in NuWro_files : 
        fUproot = uproot.open(f)
        for det in ['ND', 'FD'] :
            for mode in [ ['numode', 'FHC'], ['nubarmode', 'RHC' ] ] :
                for species in ['numu', 'nue', 'numubar', 'nuebar'] :
                    print "Processing", f, det, mode[0], species

                    try :
                        t = fUproot[species+'_'+det+'_'+mode[0]+'/cafTruthTree']
                    except KeyError :
                        print species+'_'+det+'_'+mode[0]+'/cafTruthTree', "not found in", f, "Continuing..."
                        continue
                    
                    d = t.arrays(varListNuWro, outputtype=pd.DataFrame)
        
                    scaleVars(d, ["Ev", "LepE", "W", "eP", "eN", "ePip", "ePim", "ePi0", "eOther"], 1./1e3)
                    scaleVars(d, ["Q2"], 1./1e6)

                    # Training for different detector / horn current mode / nu species separately, so don't need nuPDG as a feature
                    # All events seem to be CC, so remove that feature, too.
                    d.drop(columns=["nuPDG", "isCC"], inplace = True) 
        
                    # Add column with weight. Will be dummy for NuWro file, at least for now, but this way can treat both GENIE and NuWro the same
                    d["weight"] = 1.

                    d["label"] = 1

                    # Do one-hot encoding:
                    oneHotEncoding(df = d, detector = det, runmode = mode[1], species = species) 

                    # Split into training and testing and store
                    mask = np.random.rand(len(d)) < trainFraction

                    store.append('NuWro_'+species+'_'+det+'_'+mode[1]+'_train', d[mask])
                    store.append('NuWro_'+species+'_'+det+'_'+mode[1]+'_test', d[~mask])
                    del d
                    del t
        del fUproot

    # Get CAF samples
    for det in ['ND', 'FD_nonswap', 'FD_nueswap'] :
        for mode in [ ['numode', 'FHC'], ['nubarmode', 'RHC' ] ] :
            for f in glob.glob(CAF_fNames[det][mode[1]]) :

                # Open file, get dataframe
                fUproot = uproot.open(f)
                try :
                    t = fUproot['cafTree']
                except KeyError :
                    print "caf not found in", f, "Continuing..."
                    continue

                d = t.arrays(varListCAF, outputtype=pd.DataFrame)

                # Only dealing with CC events so get rid of everything else
                d.drop(d[d.isCC != 1].index, inplace=True)
                # Don't need isCC column anymore
                d.drop(columns=["isCC"], inplace=True)

                # First calculate total cvweight and drop all cvweight columns
                d['weight'] = 1.
                for cvWeight in cvWeightList :
                    d['weight'] *= d[cvWeight]
                d.drop(columns=cvWeightList, inplace = True)

                # Add label
                d["label"] = 0
                
                detToWrite = 'FD'

                if det == 'FD_nonswap' and mode[0] == 'numode' :
                    speciesToWrite = [ ['numu', 14] ]
                elif det == 'FD_nonswap' and mode[0] == 'nubarmode' :
                    speciesToWrite = [ ['numu', 14], ['numubar', -14]] 
                elif det == 'FD_nueswap' and mode[0] == 'numode' :
                    speciesToWrite = [ ['nue', 12] ] 
                elif det == 'FD_nueswap' and mode[0] == 'nubarmode' :
                    speciesToWrite = [ ['nue', 12], ['nuebar', -12]] 
                elif det == 'ND' :
                    speciesToWrite = [ ['nue', 12], ['nuebar', -12], ['numu', 14], ['numubar', -14]] 
                    detToWrite = 'ND'
                else :
                    print "Don't know what to do with this detector, runmode combo!! quitting.", det, mode[0]
                    exit(-1)
                    


                for species in speciesToWrite:
                    print "Processing", f, det, mode[0], species[0]
                    dftemp = d.loc[d['nuPDG'] == species[1], d.columns != 'nuPDG']
                    
                    # Do one-hot encoding:
                    oneHotEncoding(df = dftemp, detector = det, runmode = mode[1], species = species[0]) 
                    
                    # Split into training and testing and store
                    mask = np.random.rand(len(dftemp)) < trainFraction

                    store.append('CAF_GENIE_'+species[0]+'_'+detToWrite+'_'+mode[1]+'_train', dftemp[mask])
                    store.append('CAF_GENIE_'+species[0]+'_'+detToWrite+'_'+mode[1]+'_test', dftemp[~mask])
                    
                    del dftemp

                del d
                del t
                del fUproot


def scaleVars(df, varList, scaleFactor) :
    for var in varList :
        df[var] = df[var]*scaleFactor

def oneHotEncoding(df, detector, runmode, species) :

    if detector in ["ND"] :
        df["isND"] = 1
    elif detector in ["FD_nueswap", "FD_nonswap", "FD"] :
        df["isND"] = 0
    else :
        print "I don't know this detector:", detector, "Quitting..."
        exit(-1)
        
    if runmode in ["FHC"] :
        df["isFHC"] = 1
    elif runmode in ["RHC"] :
        df["isFHC"] = 0
    else :
        print "I don't know this run mode:", runmode, "Quitting..."
        exit(-1)
        
    if species in ["nue", "numu" ] :
        df["isNu"] = 1
    elif species in ["nuebar", "numubar"] :
        df["isNu"] = 0
    else :
        print "I don't know this species:", species, "Quitting..."
        exit(-1)
        
    if species in ["numu", "numubar" ] :
        df["isMu"] = 1
    elif species in ["nue", "nuebar"] :
        df["isMu"] = 0
    else :
        print "I don't know this species:", species, "Quitting..."
        exit(-1)
        
"""

nuisReadVars = ["cc",
                "PDGnu",
                "Enu_true",
                "ELep",
                "CosLep",
                "Q2",
                "W",
                "x",
                "y",
                "nfsp",
                "pdg",
                "E",
                "px",
                "py",
                "pz"]
        
rwVars = ["isCC",
          "nuPDG",
          "Ev",
          "LepE",
          "LepNuAngle",
          "Q2",
          "W",
          "X",
          "Y",
          "nP",
          "nN",
          "nipip",
          "nipim",
          "nipi0",
          "niem",
          "niother",
          "nNucleus",
          "nUNKNOWN",
          "eP",
          "eN",
          "ePip",
          "ePim",
          "ePi0",
          "eOther"]

cvWeightList = ["MaCCQE_cvwgt",
                "VecFFCCQEshape_cvwgt",
                "MaNCEL_cvwgt",
                "EtaNCEL_cvwgt",
                "MaCCRES_cvwgt",
                "MvCCRES_cvwgt",
                "MaNCRES_cvwgt",
                "MvNCRES_cvwgt",
                "RDecBR1gamma_cvwgt",
                "RDecBR1eta_cvwgt",
                "Theta_Delta2Npi_cvwgt",
                "AhtBY_cvwgt",
                "BhtBY_cvwgt",
                "CV1uBY_cvwgt",
                "CV2uBY_cvwgt",
                "FormZone_cvwgt",
                "MFP_pi_cvwgt",
                "FrCEx_pi_cvwgt",
                "FrElas_pi_cvwgt",
                "FrInel_pi_cvwgt",
                "FrAbs_pi_cvwgt",
                "FrPiProd_pi_cvwgt",
                "MFP_N_cvwgt",
                "FrCEx_N_cvwgt",
                "FrElas_N_cvwgt",
                "FrInel_N_cvwgt",
                "FrAbs_N_cvwgt",
                "FrPiProd_N_cvwgt",
                "CCQEPauliSupViaKF_cvwgt",
                "Mnv2p2hGaussEnhancement_cvwgt",
                "MKSPP_ReWeight_cvwgt",
                "E2p2h_A_nu_cvwgt",
                "E2p2h_B_nu_cvwgt",
                "E2p2h_A_nubar_cvwgt",
                "E2p2h_B_nubar_cvwgt",
                "NR_nu_n_CC_2Pi_cvwgt",
                "NR_nu_n_CC_3Pi_cvwgt",
                "NR_nu_p_CC_2Pi_cvwgt",
                "NR_nu_p_CC_3Pi_cvwgt",
                "NR_nu_np_CC_1Pi_cvwgt",
                "NR_nu_n_NC_1Pi_cvwgt",
                "NR_nu_n_NC_2Pi_cvwgt",
                "NR_nu_n_NC_3Pi_cvwgt",
                "NR_nu_p_NC_1Pi_cvwgt",
                "NR_nu_p_NC_2Pi_cvwgt",
                "NR_nu_p_NC_3Pi_cvwgt",
                "NR_nubar_n_CC_1Pi_cvwgt",
                "NR_nubar_n_CC_2Pi_cvwgt",
                "NR_nubar_n_CC_3Pi_cvwgt",
                "NR_nubar_p_CC_1Pi_cvwgt",
                "NR_nubar_p_CC_2Pi_cvwgt",
                "NR_nubar_p_CC_3Pi_cvwgt",
                "NR_nubar_n_NC_1Pi_cvwgt",
                "NR_nubar_n_NC_2Pi_cvwgt",
                "NR_nubar_n_NC_3Pi_cvwgt",
                "NR_nubar_p_NC_1Pi_cvwgt",
                "NR_nubar_p_NC_2Pi_cvwgt",
                "NR_nubar_p_NC_3Pi_cvwgt",
                "BeRPA_A_cvwgt",
                "BeRPA_B_cvwgt",
                "BeRPA_D_cvwgt",
                "BeRPA_E_cvwgt",
                "C12ToAr40_2p2hScaling_nu_cvwgt",
                "C12ToAr40_2p2hScaling_nubar_cvwgt",
                "nuenuebar_xsec_ratio_cvwgt",
                "nuenumu_xsec_ratio_cvwgt",
                "SPPLowQ2Suppression_cvwgt",
                "FSILikeEAvailSmearing_cvwgt"]




if __name__ == "__main__" :
    main()
