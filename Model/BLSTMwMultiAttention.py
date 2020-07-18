import torch
import numpy
from Auxiliary.Loader import Loader_IEMOCAP_Both
from Model.AttentionBase import AttentionBase_Multi


class BLSTMwMultiAttention(AttentionBase_Multi):
    def __init__(self, attentionName, attentionScope, attentionParameter, featuresNumber, cudaFlag):
        super(BLSTMwMultiAttention, self).__init__(
            attentionName=attentionName, attentionScope=attentionScope, attentionParameter=attentionParameter,
            featuresNumber=256, cudaFlag=cudaFlag)
        self.moduleName = 'BLSTM-Multi-W-%s' % attentionName[0]
        self.rnnLeftLayer = torch.nn.LSTM(
            input_size=featuresNumber, hidden_size=128, num_layers=2, bidirectional=True, bias=True, batch_first=True)
        self.rnnRightLayer = torch.nn.LSTM(
            input_size=320, hidden_size=128, num_layers=2, bidirectional=True, bias=True, batch_first=True)
        self.predict = torch.nn.Linear(in_features=512, out_features=4)

    def forward(self, batchData, batchSeq, batchRepre, batchRepreSeq):
        batchData = batchData.permute([0, 2, 1, 3])
        batchData = batchData.reshape(
            [batchData.size()[0], batchData.size()[1], batchData.size()[2] * batchData.size()[3]])
        leftBLSTMResult, _ = self.rnnLeftLayer(batchData)
        leftResult, _ = self.ApplyAttention(
            dataInput=leftBLSTMResult, attentionName=self.attentionName[0],
            attentionParameter=self.attentionParameter[0], inputSeqLen=batchSeq, hiddenNoduleNumbers=256)
        rightBLSTMResult, _ = self.rnnRightLayer(batchRepre)
        rightResult, _ = self.ApplyAttention(
            dataInput=rightBLSTMResult, attentionName=self.attentionName[1],
            attentionParameter=self.attentionParameter[1], inputSeqLen=batchRepreSeq, hiddenNoduleNumbers=256)

        concatResult = torch.cat([leftResult, rightResult], dim=1)
        predict = self.predict(concatResult)
        return predict


if __name__ == '__main__':
    trainDataset, testDataset = Loader_IEMOCAP_Both()
    model = BLSTMwMultiAttention(
        attentionName=['StandardAttention', 'StandardAttention'], attentionScope=[10, 10],
        attentionParameter=['LeftAttention', 'RightAttention'], featuresNumber=40, cudaFlag=False)

    for batchIndex, [batchData, batchSeq, batchRepre, batchRepreSeq, batchLabel] in enumerate(trainDataset):
        print(numpy.shape(batchData), numpy.shape(batchRepre))
        result = model(batchData, batchSeq, batchRepre, batchRepreSeq)
        print(numpy.shape(result))
        exit()
