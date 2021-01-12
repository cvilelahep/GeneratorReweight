import nuisflatDataset
import torch

#@profile
def runTest(n) :

    dataFiles = ["/disk/cvilela/GeneratorReweight/LargeSamples/argon_GENIEv2.h5",
                 "/disk/cvilela/GeneratorReweight/LargeSamples/argon_NUWRO.h5"]

    num_workers = 4

    train_dataset = nuisflatDataset.nuisflatDataset(dataFiles, test = False)
    test_dataset = nuisflatDataset.nuisflatDataset(dataFiles, test = True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler = torch.utils.data.BatchSampler(nuisflatDataset.largedataSampler(data_source = train_dataset, sequence_length = 100), batch_size = 2000, drop_last = False), num_workers = num_workers)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler = torch.utils.data.BatchSampler(nuisflatDataset.largedataSampler(data_source = test_dataset, sequence_length = 100), batch_size = 2000, drop_last = False), num_workers = num_workers)
    print("INITIALISED DATALOADER")
    
    it = iter(train_loader)
    print("GOT ITERATOR")

    for i, data in enumerate(train_loader) :
        print(data)
        if i >= n :
            break

if __name__ == "__main__" :
    runTest(100)
