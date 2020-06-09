import torch
import numpy
from Model.AttentionBase import AttentionBase


class Encoder(AttentionBase):
    def __init__(self, attentionName, attentionScope, featuresNumber, cudaFlag):
        super(Encoder, self).__init__(attentionName=attentionName, attentionScope=attentionScope, featuresNumber=256,
                                      cudaFlag=cudaFlag)
        self.firstLayer = torch.nn.Linear(in_features=featuresNumber, out_features=128, bias=True)
        self.blstmLayer = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bias=True, batch_first=True,
                                        bidirectional=True)

    def forward(self, batchData, batchSeq):
        firstOutput = self.firstLayer(input=batchData.view([-1, 40])).relu()
        blstmOutput, blstmState = self.blstmLayer(
            input=firstOutput.view([batchData.size()[0], batchData.size()[1], 128]))
        attentionResult, attentionMap = self.ApplyAttention(dataInput=blstmOutput, attentionName=self.attentionName,
                                                            inputSeqLen=batchSeq, hiddenNoduleNumbers=256)
        return attentionResult


class Encoder_CNN(torch.nn.Module):
    def __init__(self, inChannel=1):
        super(Encoder_CNN, self).__init__()
        self.conv1st = torch.nn.Conv2d(in_channels=inChannel, out_channels=8, kernel_size=8, stride=2, padding=3)
        self.conv2nd = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=8, stride=2, padding=3)
        self.conv3rd = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=8, stride=2, padding=3)

    def forward(self, inputData):
        conv1stResult = self.conv1st(inputData).relu()
        conv2ndResult = self.conv2nd(conv1stResult).relu()
        conv3rdResult = self.conv3rd(conv2ndResult).relu()
        return conv3rdResult


class Decoder(torch.nn.Module):
    def __init__(self, hiddenNoduleNumbers, featuresNumber, cudaFlag):
        self.hiddenNoduleNumbers = hiddenNoduleNumbers
        self.featuresNumber = featuresNumber
        self.cudaFlag = cudaFlag
        super(Decoder, self).__init__()
        self.firstLayer = torch.nn.Linear(in_features=40, out_features=256, bias=True)
        self.lstmLayer = torch.nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bias=True, batch_first=True,
                                       bidirectional=False)
        self.finalLayer = torch.nn.Linear(in_features=256, out_features=40, bias=True)

    def forward(self, inputData, currentState):
        firstOutput = self.firstLayer(input=inputData).relu()
        lstmOutput, lstmState = self.lstmLayer(input=firstOutput.unsqueeze(1), hx=currentState)
        finalOutput = self.finalLayer(input=lstmOutput.squeeze())
        return finalOutput, lstmState


class Decoder_CNN(torch.nn.Module):
    def __init__(self, outChannel=1):
        super(Decoder_CNN, self).__init__()
        self.trans1st = torch.nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=8, stride=2, padding=3)
        self.trans2nd = torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=8, stride=2, padding=3)
        self.trans3rd = torch.nn.ConvTranspose2d(in_channels=8, out_channels=outChannel, kernel_size=8, stride=2,
                                                 padding=3)

    def forward(self, dataInput):
        trans1stResult = self.trans1st(dataInput).relu()
        trans2ndResult = self.trans2nd(trans1stResult).relu()
        trans3rdResult = self.trans3rd(trans2ndResult)
        return trans3rdResult
