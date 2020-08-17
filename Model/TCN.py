import torch
import numpy
from Auxiliary.Loader import Loader_IEMOCAP
from Model.AttentionBase import AttentionBase, AttentionBase_Multi


class TCN(AttentionBase):
    def __init__(self, attentionName, attentionScope, cudaFlag, labelNumber=4):
        super(TCN, self).__init__(attentionName=attentionName, attentionScope=attentionScope,
                                  featuresNumber=128, cudaFlag=cudaFlag)
        self.tcnLayer1st = torch.nn.Conv1d(
            in_channels=120, out_channels=128, kernel_size=3, stride=1, dilation=1, bias=True)
        self.tcnLayer2nd = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=2, bias=True)
        self.tcnLayer3rd = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=4, bias=True)
        self.tcnLayer4th = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=8, bias=True)
        self.tcnLayer5th = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=16, bias=True)
        self.predictLayer = torch.nn.Linear(in_features=128, out_features=labelNumber, bias=True)

    def forward(self, inputData, inputSeqLen):
        inputData = inputData.permute(0, 1, 3, 2)
        inputData = inputData.reshape([inputData.size()[0], 120, inputData.size()[3]])
        padZeros = torch.zeros([inputData.size()[0], inputData.size()[1], 62])
        if self.cudaFlag: padZeros = padZeros.cuda()
        inputData = torch.cat([padZeros, inputData], dim=2)

        tcn1st = self.tcnLayer1st(inputData).relu()
        tcn2nd = self.tcnLayer2nd(tcn1st).relu()
        tcn3rd = self.tcnLayer3rd(tcn2nd).relu()
        tcn4th = self.tcnLayer4th(tcn3rd).relu()
        tcn5th = self.tcnLayer5th(tcn4th).relu().permute(0, 2, 1)
        attentionResult, attentionMap = self.ApplyAttention(dataInput=tcn5th, attentionName=self.attentionName,
                                                            inputSeqLen=inputSeqLen, hiddenNoduleNumbers=128)
        predict = self.predictLayer(attentionResult)
        return predict, attentionMap

    def cudaTreatment(self):
        pass


class TCN_Multi(AttentionBase_Multi):
    def __init__(self, attentionName, attentionScope, attentionParameter, cudaFlag, labelNumber=4):
        super(TCN_Multi, self).__init__(attentionName=attentionName, attentionScope=attentionScope,
                                        attentionParameter=attentionParameter, featuresNumber=128, cudaFlag=cudaFlag)
        self.tcnLayer1st = torch.nn.Conv1d(
            in_channels=120, out_channels=128, kernel_size=3, stride=1, dilation=1, bias=True)
        self.tcnLayer2nd = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=2, bias=True)
        self.tcnLayer3rd = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=4, bias=True)
        self.tcnLayer4th = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=8, bias=True)
        self.tcnLayer5th = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=16, bias=True)
        self.predictLayer = torch.nn.Linear(in_features=128 * len(self.attentionName), out_features=labelNumber,
                                            bias=True)

    def forward(self, inputData, inputSeqLen):
        inputData = inputData.permute(0, 1, 3, 2)
        inputData = inputData.reshape([inputData.size()[0], 120, inputData.size()[3]])
        padZeros = torch.zeros([inputData.size()[0], inputData.size()[1], 62])
        if self.cudaFlag: padZeros = padZeros.cuda()
        inputData = torch.cat([padZeros, inputData], dim=2)

        tcn1st = self.tcnLayer1st(inputData).relu()
        tcn2nd = self.tcnLayer2nd(tcn1st).relu()
        tcn3rd = self.tcnLayer3rd(tcn2nd).relu()
        tcn4th = self.tcnLayer4th(tcn3rd).relu()
        tcn5th = self.tcnLayer5th(tcn4th).relu().permute(0, 2, 1)
        totalAttentionResult, totalAttentionMap = [], []
        for index in range(len(self.attentionName)):
            attentionResult, attentionMap = self.ApplyAttention(
                dataInput=tcn5th, attentionName=self.attentionName[index],
                attentionParameter=self.attentionParameter[index], inputSeqLen=inputSeqLen, hiddenNoduleNumbers=128)
            totalAttentionResult.append(attentionResult)
            totalAttentionMap.append(attentionMap)
        totalAttentionResult = torch.cat(totalAttentionResult, dim=1)
        predict = self.predictLayer(totalAttentionResult)
        return predict, totalAttentionMap


if __name__ == '__main__':
    # model = TCN(attentionName='SelfAttention', attentionScope=0, cudaFlag=False)
    model = TCN_Multi(attentionName=['StandardAttention' for _ in range(2)], attentionScope=[0 for _ in range(2)],
                      attentionParameter=['Head%02d' % index for index in range(2)], cudaFlag=False)

    for appointGender in ['Female', 'Male']:
        for appointSession in range(1, 6):
            trainDataset, testDataset = Loader_IEMOCAP(appointGender=appointGender, appointSession=appointSession,
                                                       multiFlag=True)
            for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
                print(numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
                result, _ = model(batchData, batchSeq)
                print(numpy.shape(result))
                exit()
