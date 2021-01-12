import nuisflatDataset
import generatorReweightPytorchModel
import torch
import time
import numpy as np
import os

class BLOB :
    pass

def main() :

#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    outDir = "/disk/cvilela/GeneratorReweight/NN/"

    try :
        os.makedirs(outDir)
    except (FileExistsError) :
        pass
    
    blob = BLOB()
    blob.net = generatorReweightPytorchModel.generatorReweightPytorchModel(device).to(device)
    blob.criterion = torch.nn.BCEWithLogitsLoss(reduction = 'none')
    blob.optimizer = torch.optim.Adam(blob.net.parameters(), lr=1e-4, eps = 1e-8)
    # DEFAULTS: lr = 0.001, eps = 1e-8
    blob.data = None
    blob.label = None

    dataFiles = ["/disk/cvilela/GeneratorReweight/LargeSamples/argon_GENIEv2.h5", "/disk/cvilela/GeneratorReweight/LargeSamples/argon_NUWRO.h5"]

    num_workers = 6
    batch_size = 5000
    sequence_length = 250

    
    train_dataset = nuisflatDataset.nuisflatDataset(dataFiles, test = False)
    test_dataset = nuisflatDataset.nuisflatDataset(dataFiles, test = True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler = torch.utils.data.BatchSampler(nuisflatDataset.largedataSampler(data_source = train_dataset, sequence_length = sequence_length), batch_size = batch_size, drop_last = False), num_workers = num_workers, pin_memory = False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler = torch.utils.data.BatchSampler(nuisflatDataset.largedataSampler(data_source = test_dataset, sequence_length = sequence_length), batch_size = batch_size, drop_last = False), num_workers = num_workers, pin_memory = False)

    # Training loop
    TRAIN_EPOCH = 1.0 
    blob.net.train()
    epoch = 0.
    iteration = 0.

    fTrainLossTracker = open(outDir+"/"+'trainLoss.log', 'w', buffering = 1)
    fValidationLossTracker = open(outDir+"/"+'validationLoss.log', 'w', buffering = 1)

    while epoch < TRAIN_EPOCH :
        print('Epoch', epoch, int(epoch+0.5), 'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for i, data in enumerate(train_loader) :
            generatorReweightPytorchModel.FillLabel(blob,data)
            generatorReweightPytorchModel.FillData(blob,data)
            
            res = generatorReweightPytorchModel.forward(blob, True)
            
            generatorReweightPytorchModel.backward(blob)

            epoch += 1./len(train_loader)
            iteration += 1
            
            fTrainLossTracker.write(str(epoch)+","+str(iteration)+","+str(res['loss'])+'\n')

            if i == 0 or (i+1)%10 == 0 :
                print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])
                
            
            if (i+1)%100 == 0 :
                with torch.no_grad() :
                    blob.net.eval()
                    test_data = next(iter(test_loader))
                    generatorReweightPytorchModel.FillLabel(blob,test_data)
                    generatorReweightPytorchModel.FillData(blob,test_data)
                    res = generatorReweightPytorchModel.forward(blob, False)
                    blob.net.train()
                    fValidationLossTracker.write(str(epoch)+","+str(iteration)+","+str(res['loss'])+'\n')
                    blob.net.train()
                    print('VALIDATION', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])

            if (i%10000 == 0) and (i > 0) :
                torch.save(blob.net.state_dict(), outDir+"/"+"generatorReweightPytorchModel_{0}.nn".format(i))
                    
            if epoch >= TRAIN_EPOCH :
                break
    
    fTrainLossTracker.close()
    fValidationLossTracker.close()
    torch.save(blob.net.state_dict(), outDir+"/"+"generatorReweightPytorchModel.nn")

if __name__ == '__main__' :
    main()
