import torch
import numpy
from Model.AttentionBase import AttentionBase
from Auxiliary.Loader import Loader_IEMOCAP_OnlyText


class TCN_Text(AttentionBase):
    def __init__(self, attentionName, attentionScope, cudaFlag):
        super(TCN_Text, self).__init__(attentionName=attentionName, attentionScope=attentionScope,
                                       featuresNumber=256, cudaFlag=cudaFlag)
        self.embeddingLayer = torch.nn.Embedding(num_embeddings=50, embedding_dim=64)
        self.characterConv1D = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        self.tcnLayer1st = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1,
                                           padding=1)

    def forward(self, inputData, inputSeqLen):
        embeddingResult = self.embeddingLayer(
            inputData.view(inputData.size()[0] * inputData.size()[1], inputData.size()[2]))
        characterLavelResult = self.characterConv1D(embeddingResult.permute(0, 2, 1))
        maxPoolingResult, maxPoolingPosition = characterLavelResult.max(dim=2)
        characterFinal = maxPoolingResult.view(inputData.size()[0], inputData.size()[1], 128).permute(0, 2, 1)

        tcn1stResult = self.tcnLayer1st(characterFinal).relu()
        return tcn1stResult


if __name__ == '__main__':
    model = TCN_Text(attentionName='StandardAttention', attentionScope=0, cudaFlag=False)
    for appointGender in ['Female', 'Male']:
        for appointSession in range(1, 6):
            trainDataset, testDataset = Loader_IEMOCAP_OnlyText(appointGender=appointGender,
                                                                appointSession=appointSession)
            for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
                print(numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
                result = model(batchData, batchSeq)
                print(numpy.shape(result))
                exit()
