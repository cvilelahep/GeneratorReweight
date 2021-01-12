import matplotlib.pyplot as plt
import numpy as np

fTrainLog = "/disk/cvilela/GeneratorReweight/NN/trainLoss.log"
fTestLog = "/disk/cvilela/GeneratorReweight/NN/validationLoss.log"

test_epoch = []
test_loss = []

train_epoch = []
train_loss = []

with open(fTestLog, "r") as f :
    for line in f.readlines() :
        splitline = line.split(',')
        test_epoch.append(float(splitline[0]))
        test_loss.append(float(splitline[2]))

with open(fTrainLog, "r") as f :
    for line in f.readlines() :
        splitline = line.split(',')
        train_epoch.append(float(splitline[0]))
        train_loss.append(float(splitline[2]))


plt.plot(train_epoch, train_loss, label = 'Train')
plt.plot(test_epoch, test_loss, label = 'Test')
plt.legend()
plt.show()

