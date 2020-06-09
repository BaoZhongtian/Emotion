import os
import torch
import numpy
from Auxiliary.Tools import SaveNetwork


def MaskMechanism(batchPredict, batchSeq, cudaFlag):
    mask = []
    for sample in batchSeq:
        mask.append(torch.cat([torch.ones(sample), torch.zeros(torch.max(batchSeq) - sample)]).unsqueeze(0))
    mask = torch.cat(mask, dim=0).unsqueeze(-1).unsqueeze(1).repeat([1, batchPredict.size()[1], 1, 40])
    if mask.size()[2] > batchPredict.size()[2]:
        mask = mask[:, :, :batchPredict.size()[2], :]
    if cudaFlag: mask = mask.cuda()
    return batchPredict.mul(mask)


def TrainTemplate_CNN_EncoderDecoder(encoder, decoder, trainDataset, cudaFlag, learningRate=1E-3,
                                     trainEpisode=10000, weight=100, encoderOptimizer=None, decoderOptimizer=None,
                                     saveFlag=False, savePath=None):
    if saveFlag and not os.path.exists(savePath): os.makedirs(savePath)
    if cudaFlag:
        encoder.cuda()
        decoder.cuda()

    criterion = torch.nn.L1Loss()
    if encoderOptimizer is None: encoderOptimizer = torch.optim.Adam(encoder.parameters(), lr=learningRate)
    if decoderOptimizer is None: decoderOptimizer = torch.optim.Adam(decoder.parameters(), lr=learningRate)

    for episode in range(trainEpisode):
        episodeLoss = 0.0
        if saveFlag: file = open(os.path.join(savePath, 'Loss-%04d.csv' % episode), 'w')
        for batchIndex, [batchData, batchSeq, _] in enumerate(trainDataset):
            if cudaFlag: batchData = batchData.cuda()
            result = encoder(batchData)
            predict = decoder(result)

            comparedData = batchData[:, :, :predict.size()[2], :]
            maskedPredict = MaskMechanism(batchPredict=predict, batchSeq=batchSeq, cudaFlag=cudaFlag)
            # print(numpy.shape(maskedPredict), numpy.shape(batchData))
            loss = weight * criterion(input=maskedPredict, target=comparedData)

            episodeLoss += loss
            print('\rBatch %d Loss = %f' % (batchIndex, loss), end='')

            encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()
            loss.backward()
            encoderOptimizer.step()
            decoderOptimizer.step()
            if saveFlag: file.write(str(loss.detach().cpu().numpy()) + '\n')
        print('\n\t\t\tEpisode %d Total Loss = %f' % (episode, episodeLoss))

    if saveFlag and episode % 10 == 9:
        SaveNetwork(model=encoder, optimizer=encoderOptimizer,
                    savePath=os.path.join(savePath, 'Encoder-%04d' % episode))
        SaveNetwork(model=decoder, optimizer=decoderOptimizer,
                    savePath=os.path.join(savePath, 'Decoder-%04d' % episode))


def TrainTemplate_CNN_Meta(encoder, decoder, trainDataset, cudaFlag, learningRate=1E-3,
                           trainEpisode=10000, weight=100, encoderOptimizer=None, decoderOptimizer=None,
                           saveFlag=False, savePath=None):
    if saveFlag and not os.path.exists(savePath): os.makedirs(savePath)
    if cudaFlag:
        encoder.cuda()
        decoder.cuda()

    criterion = torch.nn.L1Loss()
    if encoderOptimizer is None: encoderOptimizer = torch.optim.Adam(encoder.parameters(), lr=learningRate)
    if decoderOptimizer is None: decoderOptimizer = torch.optim.Adam(decoder.parameters(), lr=learningRate)

    for episode in range(trainEpisode):
        episodeLoss = 0.0
        encoderOptimizer.zero_grad()

        if saveFlag: file = open(os.path.join(savePath, 'Loss-%04d.csv' % episode), 'w')
        for batchIndex, [batchData, batchSeq, _] in enumerate(trainDataset):
            if cudaFlag: batchData = batchData.cuda()
            result = encoder(batchData)
            predict = decoder(result)

            comparedData = batchData[:, :, :predict.size()[2], :]
            maskedPredict = MaskMechanism(batchPredict=predict, batchSeq=batchSeq, cudaFlag=cudaFlag)
            # print(numpy.shape(maskedPredict), numpy.shape(batchData))
            loss = weight * criterion(input=maskedPredict, target=comparedData)

            episodeLoss += loss
            print('\rBatch %d Loss = %f' % (batchIndex, loss), end='')

            decoderOptimizer.zero_grad()
            loss.backward()
            decoderOptimizer.step()
            if saveFlag: file.write(str(loss.detach().cpu().numpy()) + '\n')
        print('\n\t\t\tEpisode %d Total Loss = %f' % (episode, episodeLoss))

    encoderOptimizer.step()
    if saveFlag and episode % 10 == 9:
        SaveNetwork(model=encoder, optimizer=encoderOptimizer,
                    savePath=os.path.join(savePath, 'Encoder-%04d' % episode))
        SaveNetwork(model=decoder, optimizer=decoderOptimizer,
                    savePath=os.path.join(savePath, 'Decoder-%04d' % episode))
