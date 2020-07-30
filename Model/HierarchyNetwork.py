import torch
import numpy
from Model.AttentionBase import AttentionBase
from Auxiliary.Loader import Loader_IEMOCAP_OnlyText


class HierarchyNetwork(AttentionBase):
    def __init__(self, attentionName, attentionScope, cudaFlag):
        super(HierarchyNetwork, self).__init__(attentionName=attentionName, attentionScope=attentionScope,
                                               featuresNumber=256, cudaFlag=cudaFlag)
        self.embeddingLayer = torch.nn.Embedding(num_embeddings=50, embedding_dim=64)
        self.characterConv1D = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.lstmLayer = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bias=True, batch_first=True,
                                       bidirectional=True)
        self.predictLayer = torch.nn.Linear(in_features=256, out_features=4, bias=True)

    def forward(self, inputData, inputSeqLen):
        embeddingResult = self.embeddingLayer(
            inputData.view(inputData.size()[0] * inputData.size()[1], inputData.size()[2]))
        characterLavelResult = self.characterConv1D(embeddingResult.permute(0, 2, 1))
        maxPoolingResult, maxPoolingPosition = characterLavelResult.max(dim=2)
        characterFinal = maxPoolingResult.view(inputData.size()[0], inputData.size()[1], 128)
        lstmResult, lstmState = self.lstmLayer(characterFinal)

        attentionResult, attentionHotMap = self.ApplyAttention(dataInput=lstmResult, attentionName=self.attentionName,
                                                               inputSeqLen=inputSeqLen, hiddenNoduleNumbers=256)
        predict = self.predictLayer(input=attentionResult)
        return predict, attentionHotMap


if __name__ == '__main__':
    trainDataset, testDataset = Loader_IEMOCAP_OnlyText(appointGender='Female', appointSession=1)
    model = HierarchyNetwork(attentionName='StandardAttention', attentionScope=10, cudaFlag=False)
    for batchIndex, [batchData, batchLabel, batchSeq] in enumerate(trainDataset):
        print(numpy.shape(batchData))
        result = model(batchData, batchSeq)
        print(numpy.shape(result))
        # print(result[0])
        exit()
