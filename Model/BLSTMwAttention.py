import torch
from Model.AttentionBase import AttentionBase


class BLSTMwAttention(AttentionBase):
    def __init__(self, attentionName, attentionScope, featuresNumber, classNumber, cudaFlag):
        super(BLSTMwAttention, self).__init__(
            attentionName=attentionName, attentionScope=attentionScope, featuresNumber=128 * 2, cudaFlag=cudaFlag)
        self.moduleName = 'BLSTM-W-%s-%d' % (attentionName, attentionScope)
        self.rnnLayer = torch.nn.LSTM(input_size=featuresNumber, hidden_size=128, num_layers=2, bidirectional=True)
        self.predict = torch.nn.Linear(in_features=256, out_features=classNumber)

    def forward(self, inputData, inputSeqLen):
        inputData = inputData.float().squeeze()
        inputData = inputData.permute([0, 2, 1, 3])
        inputData = inputData.reshape(
            [inputData.size()[0], inputData.size()[1], inputData.size()[2] * inputData.size()[3]])
        rnnOutput, _ = self.rnnLayer(input=inputData, hx=None)
        attentionResult, attentionHotMap = self.ApplyAttention(
            dataInput=rnnOutput, attentionName=self.attentionName, inputSeqLen=inputSeqLen, hiddenNoduleNumbers=256)
        predict = self.predict(input=attentionResult)
        return predict, attentionHotMap

    def cudaTreatment(self):
        pass
