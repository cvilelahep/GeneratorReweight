import os
import glob

#import pandas as pd
import numpy as np
import uproot4
import h5py
import awkward1 as ak
import random

m = {}
m["P"] = 0.93827
m["N"] = 0.93957
m["piC"] = 0.13957
m["pi0"] = 0.13498
m["kC"] = 0.49368
m["k0"] = 0.49764

samples = {}

sampledir = "/disk/cvilela/GeneratorReweight/LargeSamples/"
samples["argon_GENIEv2"] = sampledir+"flat_argon_*_GENIEv2*.root"
#samples["argon_GENIEv3_G18_10a_02_11a"] = sampledir+"flat_argon_*_GENIEv3_G18_10a_02_11a*.root"
#samples["argon_GENIEv3_G18_10b_00_000"] = sampledir+"flat_argon_*_GENIEv3_G18_10b_00_000*.root"
samples["argon_NEUT"] = sampledir+"flat_argon_*_NEUT*.root"
#samples["argon_NUWRO"] = sampledir+"flat_argon_*_NUWRO.root"


def nuisflatToH5(fNameNuis, fNameh5, trainFraction) :

    if os.path.exists(fNameh5):
        os.remove(fNameh5)

    with h5py.File(fNameh5, 'w') as hf:
        fileList = glob.glob(fNameNuis)
        random.shuffle(fileList)
        
        for i, fName in enumerate(fileList) :
            with uproot4.open(fName+":FlatTree_VARS") as tree :
                print("Reading {0}".format(fName))
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
                treeArr["nipim"] = ak.count_nonzero(Pimmask, axis = 1)
                treeArr["nipi0"] = ak.count_nonzero(Pi0mask, axis = 1)
                treeArr["nikp"] = ak.count_nonzero(Kpmask, axis = 1)
                treeArr["nikm"] = ak.count_nonzero(Kmmask, axis = 1)
                treeArr["nik0"] = ak.count_nonzero(K0mask, axis = 1)
                treeArr["niem"] = ak.count_nonzero(EMmask, axis = 1)

                treeArr["eP"] = ak.sum(treeArr["E"][Pmask], axis = 1) - treeArr["nP"]*m["P"]
                treeArr["eN"] = ak.sum(treeArr["E"][Nmask], axis = 1) - treeArr["nN"]*m["N"]
                treeArr["ePip"] = ak.sum(treeArr["E"][Pipmask], axis = 1) - treeArr["nipip"]*m["piC"]
                treeArr["ePim"] = ak.sum(treeArr["E"][Pimmask], axis = 1) - treeArr["nipim"]*m["piC"]
                treeArr["ePi0"] = ak.sum(treeArr["E"][Pi0mask], axis = 1) - treeArr["nipi0"]*m["pi0"]

                treeArr["eOther"] = ak.sum(treeArr["E"][othermask] - (treeArr["E"][othermask]**2-treeArr["px"][othermask]**2-treeArr["py"][othermask]**2-treeArr["pz"][othermask]**2)**0.5, axis = 1)

                treeArr["isNu"] = treeArr["PDGnu"] > 0
                treeArr["isNue"] = abs(treeArr["PDGnu"]) == 12
                treeArr["isNumu"] = abs(treeArr["PDGnu"]) == 14
                treeArr["isNutau"] = abs(treeArr["PDGnu"]) == 16

                varOut = ["isNu", "isNue", "isNumu", "isNutau", "cc", "Enu_true", "ELep", "CosLep", "Q2", "W", "x", "y", "nP", "nN", "nipip", "nipim", "nipi0", "niem", "eP", "eN", "ePip", "ePim", "ePi0"] #, "eOther"]
                
                treeArr = ak.values_astype(treeArr[varOut], np.float32)
                
                data = ak.to_numpy(treeArr)
                data = data.view(np.float32).reshape((len(data), len(varOut)))
                
                print(data.shape)
                
                split = int(trainFraction*len(data))
                
                if i == 0 :
                    # Create hdf5 dataset
                    hf.create_dataset("train_data", data = data[:split],maxshape = (None, len(varOut)), chunks = True)
                    hf.create_dataset("test_data", data = data[split:], maxshape = (None, len(varOut)), chunks = True)
                else :
                    # Extend existing dataset
                    hf["train_data"].resize((hf["train_data"].shape[0] + data[:split].shape[0]), axis = 0)
                    hf["train_data"][-data[:split].shape[0]:] = data[:split]

                    hf["test_data"].resize((hf["test_data"].shape[0] + data[split:].shape[0]), axis = 0)
                    hf["test_data"][-data[split:].shape[0]:] = data[split:]
                    

def main() :

    for sample, fName in samples.items() :
        nuisflatToH5(fName, sampledir+"/"+sample+".h5", 0.9)


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
        

if __name__ == "__main__" :
    main()
