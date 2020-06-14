import os
import torch
import numpy
from Auxiliary.Tools import LoadNetwork
from Auxiliary.Loader import Loader_IEMOCAP
from Model.EncoderDecoder import Encoder_CNN

if __name__ == '__main__':
    multiFlag, metaFlag = True, True
    cudaFlag = True

    trainDataset, testDataset = Loader_IEMOCAP(appointGender=None, appointSession=None, batchSize=32,
                                               multiFlag=multiFlag, shuffleFlag=False)
    encoder = Encoder_CNN(inChannel=3 if multiFlag else 1)
    encoderOptimizer = torch.optim.Adam(encoder.parameters(), lr=1E-3)
    if cudaFlag: encoder.cuda()

    loadPath = 'D:/PythonProjects_Data/IEMOCAP_Result/CNN_Reconstruction%s%s/' % (
        '_MultiFlag' if multiFlag else '', '_Meta' if metaFlag else '')
    encoder, encoderOptimizer = LoadNetwork(model=encoder, optimizer=encoderOptimizer,
                                            loadPath=os.path.join(loadPath, 'Encoder-9999-Parameter.pkl'))
    totalPredict, totalLabel = [], []
    for batchIndex, [batchData, batchSeq, batchLabel] in enumerate(trainDataset):
        totalLabel.extend(batchLabel.numpy())
        if cudaFlag: batchData = batchData.cuda()
        result = encoder(batchData)
        result = result.detach().cpu().numpy()
        result = numpy.transpose(result, [0, 2, 1, 3])
        result = result.reshape(
            [numpy.shape(result)[0], numpy.shape(result)[1], numpy.shape(result)[2] * numpy.shape(result)[3]])
        for index in range(len(batchSeq)):
            totalPredict.append(result[index, :int(batchSeq.numpy()[index] / 8), :])

    print(numpy.shape(totalPredict), numpy.shape(totalLabel))
    numpy.save(file='Reconstruction%s%s.npy' % ('_MultiFlag' if multiFlag else '', '_Meta' if metaFlag else ''),
               arr=totalPredict)
    numpy.save(file='ReconstructionLabel3.npy', arr=totalLabel)
