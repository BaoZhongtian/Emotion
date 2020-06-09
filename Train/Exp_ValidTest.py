import os
import torch
from Auxiliary.Tools import LoadNetwork
from Auxiliary.Loader import Loader_IEMOCAP
from Model.EncoderDecoder import Encoder_CNN, Decoder_CNN
from Auxiliary.TrainTemplate import TrainTemplate_CNN_EncoderDecoder, TrainTemplate_CNN_Meta

if __name__ == '__main__':
    multiFlag = False
    trainDataset, testDataset = Loader_IEMOCAP(appointGender=None, appointSession=None, batchSize=32,
                                               multiFlag=multiFlag)
    encoder = Encoder_CNN(inChannel=3 if multiFlag else 1)
    decoder = Decoder_CNN(outChannel=3 if multiFlag else 1)
    encoderOptimizer = torch.optim.Adam(encoder.parameters(), lr=1E-3)
    decoderOptimizer = torch.optim.Adam(decoder.parameters(), lr=1E-3)

    loadPath = 'D:/PythonProjects_New/Emotion/'
    encoder, encoderOptimizer = LoadNetwork(model=encoder, optimizer=encoderOptimizer,
                                            loadPath=os.path.join(loadPath, 'Encoder-9689-Parameter.pkl'))
    decoder, decoderOptimizer = LoadNetwork(model=decoder, optimizer=decoderOptimizer,
                                            loadPath=os.path.join(loadPath, 'Decoder-9689-Parameter.pkl'))
    TrainTemplate_CNN_EncoderDecoder(encoder=encoder, decoder=decoder, trainDataset=trainDataset, cudaFlag=True,
                                     saveFlag=False)
