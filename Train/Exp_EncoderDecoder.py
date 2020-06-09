import os
import torch
import numpy
from Auxiliary.Loader import Loader_IEMOCAP
from Model.EncoderDecoder import Encoder, Decoder

cudaFlag = True


def MaskMechanism(batchPredict, batchSeq):
    mask = []
    for sample in batchSeq:
        mask.append(torch.cat([torch.ones(sample), torch.zeros(torch.max(batchSeq) - sample)]).unsqueeze(0))
    mask = torch.cat(mask, dim=0).unsqueeze(-1).repeat([1, 1, 40])
    if cudaFlag: mask = mask.cuda()
    return batchPredict.mul(mask)


if __name__ == '__main__':
    trainDataset, testDataset = Loader_IEMOCAP(appointGender='Female', appointSession=1, batchSize=32)
    encoder = Encoder(attentionName='StandardAttention', attentionScope=10, featuresNumber=40, cudaFlag=cudaFlag)
    decoder = Decoder(hiddenNoduleNumbers=256, featuresNumber=40, cudaFlag=cudaFlag)

    if cudaFlag:
        encoder.cuda()
        decoder.cuda()

    criterion = torch.nn.L1Loss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1E-3)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1E-3)

    for trainEpisode in range(100):
        episodeLoss = 0.0
        for batchIndex, [batchData, batchSeq, batchLabel] in enumerate(testDataset):
            if cudaFlag:
                batchData = batchData.cuda()
                batchLabel = batchLabel.cuda()

            middleVector = encoder(batchData=batchData, batchSeq=batchSeq)
            currentState = [torch.cat([middleVector.unsqueeze(0), middleVector.unsqueeze(0)]) for _ in range(2)]

            inputData = batchData[:, 0, :]
            batchPredict = [inputData.unsqueeze(1)]

            if numpy.random.rand() > 0.5:
                for step in range(1, torch.max(batchSeq).numpy()):
                    inputData, currentState = decoder(inputData=inputData, currentState=currentState)
                    batchPredict.append(inputData.unsqueeze(1))
            else:
                for step in range(1, torch.max(batchSeq).numpy()):
                    inputData, currentState = decoder(inputData=batchData[:, step, :], currentState=currentState)
                    batchPredict.append(inputData.unsqueeze(1))

            batchPredict = torch.cat(batchPredict, dim=1)
            maskedPredict = MaskMechanism(batchPredict=batchPredict, batchSeq=batchSeq)
            loss = 100 * criterion(input=maskedPredict, target=batchData)
            episodeLoss += loss
            print('\rBatch %d Loss = %f' % (batchIndex, loss), end='')

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        print('\n\t\t\tEpisode %d Total Loss = %f' % (trainEpisode, episodeLoss))
