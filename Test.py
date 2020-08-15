import matplotlib.pylab as plt
import numpy
from Auxiliary.Loader import Loader_IEMOCAP

if __name__ == '__main__':
    trainDataset, testDataset = Loader_IEMOCAP(batchSize=1, shuffleFlag=False)
    for batchIndex, [batchData, batchLabel, batchSeq] in enumerate(trainDataset):
        print(batchIndex, numpy.shape(batchData), numpy.shape(batchLabel), numpy.shape(batchSeq))
        batchData = batchData.squeeze()
        batchData = batchData.numpy().transpose(1, 0)[:, :200]
        print(numpy.shape(batchData))
        plt.imshow(batchData)
        plt.axis('off')
        plt.savefig('Result.png')
        plt.show()
        exit()
