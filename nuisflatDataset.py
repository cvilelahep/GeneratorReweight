import torch
from torch.utils.data import Dataset, Sampler

import h5py

import numpy as np

class nuisflatDataset(Dataset) :
    def __init__(self, file_names, test = False) :
        print("INITIALISING DATASET")
        self._lengths = []
        self._overallLength = 0

        if test :
            self._key = "test_data"
        else :
            self._key = "train_data"

        self._file_names = file_names

        for i, filename in enumerate(file_names) :
            print("APPENDING FILE {0}".format(i))
            with h5py.File(filename, 'r') as f :
                self._lengths.append(len(f[self._key]))
                self._overallLength += self._lengths[-1]

        self._fileEnds = np.cumsum(self._lengths)
        print("ENDS INITIALISATION")
        
    def __len__(self) :
#        print("LEN CALLED {0}".format(self._overallLength))
        return self._overallLength

    def __getitem__(self, i) :
#        print("GETTING ITEM {0}".format(i))
        
        fileNumber = np.digitize(i, self._fileEnds)
        this_i = i if fileNumber == 0 else i - self._fileEnds[fileNumber-1] 

        with h5py.File(self._file_names[fileNumber], 'r') as f :
            features = f[self._key][this_i]

        # W is NaN in 0.01% of the NUWRO file. For now replace with uniformly distributed number in the [0, 10] GeV/c^2 range
        if np.isnan(features[9]) :
            features[9] = np.random.random()*10
            
        if sum(np.isnan(features)) > 0 :
            print("DATASET FOUND NAN")
            print(features)
            print("FILE {0} i {1}".format(fileNumber, this_i))
            exit(-1)
            
        
        return { "features" : features,
                 "label" : fileNumber}


class largedataSampler(Sampler) :
    def __init__ (self, data_source, sequence_length = 20) :
        self.data_source = data_source

        self.sequence_length = sequence_length

        self.i_sequence = self.sequence_length

        self.seq_Start = 0
        
    def __len__(self) :
        return len(self.data_source)
    def __iter__(self) :
        return self
    def __next__(self) :
        if self.i_sequence >= self.sequence_length :
            self.i_sequence = 0
            self.seq_start = np.random.randint(low = 0, high = len(self.data_source) - self.sequence_length)
        ret = self.seq_start+self.i_sequence
        self.i_sequence += 1
        return ret
#        return np.random.randint(low = 0, high = len(self.data_source))
