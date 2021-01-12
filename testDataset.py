import nuisflatDataset
import numpy

def runTest(n, randomize = True) :

    dataFiles = ["/disk/cvilela/GeneratorReweight/LargeSamples/argon_GENIEv2.h5",
                 "/disk/cvilela/GeneratorReweight/LargeSamples/argon_NUWRO.h5"]

    ds = nuisflatDataset.nuisflatDataset(dataFiles, test = False)

    ret = []
    
    if not randomize :
        start = numpy.random.randint(0, len(ds)-n)
        for i in range(n) :
            ret.append(ds[start+i])
    else :
        for i in numpy.random.randint(0, len(ds), size = n) :
            ret.append(ds[i])
    ret = numpy.hstack(ret)
    return ret

if __name__ == "__main__" :
    for ibatch in range(10) :
        print(runTest(200, True))
