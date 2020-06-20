import os
import torch
import numpy
from Auxiliary.Tools import SaveNetwork, LoadNetwork


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


def Template_FluctuateSize(Model, trainDataset, testDataset, cudaFlag=True, saveFlag=True, savePath=None,
                           learningRate=1E-3, episodeNumber=100):
    # Check the Path
    if os.path.exists(savePath): return
    os.makedirs(savePath)
    os.makedirs(savePath + '-TestResult')

    print(Model)

    # In general, using the Adam and Cross Entropy Loss
    if cudaFlag: Model.cuda()
    optimizer = torch.optim.Adam(params=Model.parameters(), lr=learningRate)

    lossFunction = torch.nn.CrossEntropyLoss()

    for episode in range(episodeNumber):
        episodeLoss = 0.0
        with open(os.path.join(savePath, 'Loss-%04d.csv' % episode), 'w') as file:
            for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
                if cudaFlag:
                    batchData = batchData.cuda()
                    batchSeq = batchSeq.cuda()
                    batchLabel = batchLabel.cuda()
                # print(episode, numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
                result, _ = Model(inputData=batchData, inputSeqLen=batchSeq)
                loss = lossFunction(input=result, target=batchLabel)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                file.write(str(loss.detach().cpu().numpy()) + '\n')
                episodeLoss += loss.detach().cpu().numpy()
                print('\rTraining %d Loss = %f' % (batchNumber, loss.detach().cpu().numpy()), end='')
        print('\nEpisode %d Total Loss = %f' % (episode, episodeLoss))

        if saveFlag: torch.save(obj=Model, f=os.path.join(savePath, 'Network-%04d.pkl' % episode))

        testProbability, testPartPredict, testPartLabel = [], [], []
        for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(testDataset):
            if cudaFlag:
                batchData = batchData.cuda()
                batchSeq = batchSeq.cuda()
            testPartLabel.extend(batchLabel.numpy())

            result, _ = Model(inputData=batchData, inputSeqLen=batchSeq)
            result = result.detach().cpu().numpy()
            testProbability.extend(result)
            testPartPredict.extend(numpy.argmax(result, axis=1))

        precisionRate = float(numpy.sum([testPartPredict[v] == testPartLabel[v]
                                         for v in range(len(testPartPredict))])) / len(testPartPredict) * 100
        print('Episode Test Precision %f%%' % precisionRate)

        with open(os.path.join(savePath + '-TestResult', 'Result-%04d.csv' % episode), 'w') as file:
            for indexX in range(len(testProbability)):
                for indexY in range(len(testProbability[indexX])):
                    file.write(str(testProbability[indexX][indexY]) + ',')
                file.write(str(testPartLabel[indexX]) + '\n')


def Template_FluctuateSize_BothFeatures(Model, trainDataset, testDataset, cudaFlag=True, saveFlag=True, savePath=None,
                                        learningRate=1E-3, episodeNumber=100):
    # Check the Path
    if os.path.exists(savePath): return
    os.makedirs(savePath)
    os.makedirs(savePath + '-TestResult')

    print(Model)

    # In general, using the Adam and Cross Entropy Loss
    if cudaFlag:
        Model.cuda()
        Model.cudaTreatment()
    optimizer = torch.optim.Adam(params=Model.parameters(), lr=learningRate)

    lossFunction = torch.nn.CrossEntropyLoss()

    for episode in range(episodeNumber):
        episodeLoss = 0.0
        with open(os.path.join(savePath, 'Loss-%04d.csv' % episode), 'w') as file:
            for batchNumber, (batchData, batchSeq, batchRepre, batchRepreSeq, batchLabel) in enumerate(trainDataset):
                if cudaFlag:
                    batchData = batchData.cuda()
                    batchSeq = batchSeq.cuda()
                    batchRepre = batchRepre.cuda()
                    batchRepreSeq = batchRepreSeq.cuda()
                    batchLabel = batchLabel.cuda()
                # print(episode, numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
                result = Model(batchData, batchSeq, batchRepre, batchRepreSeq)
                loss = lossFunction(input=result, target=batchLabel)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                file.write(str(loss.detach().cpu().numpy()) + '\n')
                episodeLoss += loss.detach().cpu().numpy()
                print('\rTraining %d Loss = %f' % (batchNumber, loss.detach().cpu().numpy()), end='')
        print('\nEpisode %d Total Loss = %f' % (episode, episodeLoss))

        if saveFlag: torch.save(obj=Model, f=os.path.join(savePath, 'Network-%04d.pkl' % episode))

        testProbability, testPartPredict, testPartLabel = [], [], []
        for batchNumber, (batchData, batchSeq, batchRepre, batchRepreSeq, batchLabel) in enumerate(testDataset):
            if cudaFlag:
                batchData = batchData.cuda()
                batchSeq = batchSeq.cuda()
                batchRepre = batchRepre.cuda()
                batchRepreSeq = batchRepreSeq.cuda()
            testPartLabel.extend(batchLabel.numpy())

            result = Model(batchData, batchSeq, batchRepre, batchRepreSeq)
            result = result.detach().cpu().numpy()
            testProbability.extend(result)
            testPartPredict.extend(numpy.argmax(result, axis=1))

        precisionRate = float(numpy.sum([testPartPredict[v] == testPartLabel[v]
                                         for v in range(len(testPartPredict))])) / len(testPartPredict) * 100
        print('Episode Test Precision %f%%' % precisionRate)

        with open(os.path.join(savePath + '-TestResult', 'Result-%04d.csv' % episode), 'w') as file:
            for indexX in range(len(testProbability)):
                for indexY in range(len(testProbability[indexX])):
                    file.write(str(testProbability[indexX][indexY]) + ',')
                file.write(str(testPartLabel[indexX]) + '\n')


def Template_Fluctuate_BestChoose(Model, trainDataset, testDataset, savePath, cudaFlag=True, learningRate=1E-3,
                                  episodeNumber=100):
    # Check the Path
    if os.path.exists(savePath): return
    os.makedirs(savePath)
    os.makedirs(savePath + '-TestResult')

    print(Model)

    # In general, using the Adam and Cross Entropy Loss
    if cudaFlag: Model.cuda()
    optimizer = torch.optim.Adam(params=Model.parameters(), lr=learningRate)

    lossFunction = torch.nn.CrossEntropyLoss()

    frontTestLoss = 999999.9
    for episode in range(episodeNumber):
        episodeLoss = 0.0
        with open(os.path.join(savePath, 'Loss-%04d.csv' % episode), 'w') as file:
            for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(trainDataset):
                SaveNetwork(model=Model, optimizer=optimizer, savePath=os.path.join(savePath, 'Current'))
                if cudaFlag:
                    batchData = batchData.cuda()
                    batchSeq = batchSeq.cuda()
                    batchLabel = batchLabel.cuda()
                # print(episode, numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
                result, _ = Model(inputData=batchData, inputSeqLen=batchSeq)
                loss = lossFunction(input=result, target=batchLabel)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                file.write(str(loss.detach().cpu().numpy()) + '\n')
                episodeLoss += loss.detach().cpu().numpy()
                print('\rTraining %d Loss = %f' % (batchNumber, loss.detach().cpu().numpy()), end='')

                testTotalLoss = 0.0
                for testBatchNumber, (testBatchData, testBatchSeq, testBatchLabel) in enumerate(trainDataset):
                    if cudaFlag:
                        testBatchData = testBatchData.cuda()
                        testBatchSeq = testBatchSeq.cuda()
                        testBatchLabel = testBatchLabel.cuda()
                    result, _ = Model(inputData=testBatchData, inputSeqLen=testBatchSeq)
                    testLoss = lossFunction(input=result, target=testBatchLabel)
                    testTotalLoss += testLoss.detach().cpu().numpy()
                # print('\tTotal Test Loss = %f' % testTotalLoss)
                if testTotalLoss > frontTestLoss:
                    Model, optimizer = LoadNetwork(model=Model, optimizer=optimizer,
                                                   loadPath=os.path.join(savePath, 'Current-Parameter.pkl'))
                frontTestLoss = testTotalLoss
        print('\nEpisode %d Total Loss = %f' % (episode, episodeLoss))

        testProbability, testPartPredict, testPartLabel = [], [], []
        for batchNumber, (batchData, batchSeq, batchLabel) in enumerate(testDataset):
            if cudaFlag:
                batchData = batchData.cuda()
                batchSeq = batchSeq.cuda()
            testPartLabel.extend(batchLabel.numpy())

            result, _ = Model(inputData=batchData, inputSeqLen=batchSeq)
            result = result.detach().cpu().numpy()
            testProbability.extend(result)
            testPartPredict.extend(numpy.argmax(result, axis=1))

        precisionRate = float(numpy.sum([testPartPredict[v] == testPartLabel[v]
                                         for v in range(len(testPartPredict))])) / len(testPartPredict) * 100
        print('Episode Test Precision %f%%' % precisionRate)

        with open(os.path.join(savePath + '-TestResult', 'Result-%04d.csv' % episode), 'w') as file:
            for indexX in range(len(testProbability)):
                for indexY in range(len(testProbability[indexX])):
                    file.write(str(testProbability[indexX][indexY]) + ',')
                file.write(str(testPartLabel[indexX]) + '\n')
