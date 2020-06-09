from Auxiliary.Loader import Loader_IEMOCAP
from Model.EncoderDecoder import Encoder_CNN, Decoder_CNN
from Auxiliary.TrainTemplate import TrainTemplate_CNN_EncoderDecoder, TrainTemplate_CNN_Meta

if __name__ == '__main__':
    multiFlag = False
    trainDataset, testDataset = Loader_IEMOCAP(appointGender=None, appointSession=None, batchSize=32,
                                               multiFlag=multiFlag)
    encoder = Encoder_CNN(inChannel=3 if multiFlag else 1)
    decoder = Decoder_CNN(outChannel=3 if multiFlag else 1)

    ###########################################################

    savePath = 'D:/PythonProjects_Data/CNN_Reconstruction%s_Meta/' % ('_MultiFlag' if multiFlag else '')
    TrainTemplate_CNN_Meta(encoder=encoder, decoder=decoder, trainDataset=trainDataset, cudaFlag=True,
                           savePath=savePath, saveFlag=True)
