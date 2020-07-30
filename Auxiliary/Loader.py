import os
import torch
import numpy
import torch.utils.data as torch_utils_data


class Collate_IEMOCAP:
    def __init__(self):
        pass

    def __DataTensor2D(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - len(dataInput), numpy.shape(dataInput)[1]],
                                                         dtype=torch.float)], axis=0)

    def __DataTensor3D(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros(
            [numpy.shape(dataInput)[0], maxLen - numpy.shape(dataInput)[1], numpy.shape(dataInput)[2]],
            dtype=torch.float)], axis=1)

    def __call__(self, batch):
        xs = [v[0] for v in batch]
        ys = torch.LongTensor([v[1] for v in batch])

        if len(numpy.shape(xs[0])) == 2:
            seq_lengths = torch.LongTensor([v for v in map(len, xs)])
            max_len = max([len(v) for v in xs])

            xs = numpy.array([self.__DataTensor2D(dataInput=v, maxLen=max_len) for v in xs], dtype=float)
            xs = torch.FloatTensor(xs).unsqueeze(1)
            return xs, seq_lengths, ys

        if len(numpy.shape(xs[0])) == 3:
            seq_lengths = torch.LongTensor([numpy.shape(v)[1] for v in xs])
            max_len = max([numpy.shape(v)[1] for v in xs])
            xs = numpy.array([self.__DataTensor3D(dataInput=v, maxLen=max_len) for v in xs], dtype=float)
            xs = torch.FloatTensor(xs)
            return xs, seq_lengths, ys
        raise RuntimeError("Could Not Allocate Batch")


class Collate_OnlyText:
    def __init__(self):
        pass

    def __call__(self, batch):
        xsCurrent = [v[0] for v in batch]
        xs = []
        for sample in xsCurrent:
            if len(numpy.shape(sample)) < 2:
                xs.append(numpy.reshape(sample, [1, -1]))
            else:
                xs.append(sample)
        ys = torch.LongTensor([v[1] for v in batch])
        seqLen = torch.LongTensor([v for v in map(len, xs)])
        maxSeqLen = max([len(v) for v in xs])
        maxCharLen = max([numpy.shape(v)[1] for v in xs])

        #####################################

        padingXs = []
        for index in range(len(xs)):
            padding = numpy.zeros([numpy.shape(xs[index])[0], maxCharLen - numpy.shape(xs[index])[1]])
            firstPadResult = numpy.concatenate([xs[index], padding], axis=1)
            padding = numpy.zeros([maxSeqLen - numpy.shape(xs[index])[0], maxCharLen])
            padingXs.append(numpy.concatenate([firstPadResult, padding], axis=0))
        padingXs = torch.LongTensor(padingXs)
        return padingXs, seqLen, ys


class Collate_BothRepresentation:
    def __init__(self):
        pass

    def __DataTensor2D(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - len(dataInput), numpy.shape(dataInput)[1]],
                                                         dtype=torch.float)], axis=0)

    def __DataTensor3D(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros(
            [numpy.shape(dataInput)[0], maxLen - numpy.shape(dataInput)[1], numpy.shape(dataInput)[2]],
            dtype=torch.float)], axis=1)

    def __call__(self, batch):
        xs = [v[0] for v in batch]
        ys = [v[1] for v in batch]
        zs = torch.LongTensor([v[2] for v in batch])

        if len(numpy.shape(xs[0])) == 2:
            xsLengths = torch.LongTensor([v for v in map(len, xs)])
            max_len = max([len(v) for v in xs])
            xs = numpy.array([self.__DataTensor2D(dataInput=v, maxLen=max_len) for v in xs], dtype=float)
            newXs = torch.FloatTensor(xs).unsqueeze(1)
        else:
            if len(numpy.shape(xs[0])) == 3:
                xsLengths = torch.LongTensor([numpy.shape(v)[1] for v in xs])
                max_len = max([numpy.shape(v)[1] for v in xs])
                xs = numpy.array([self.__DataTensor3D(dataInput=v, maxLen=max_len) for v in xs], dtype=float)
                newXs = torch.FloatTensor(xs)

        ysLengths = torch.LongTensor([v for v in map(len, ys)])
        max_len = max([len(v) for v in ys])
        ys = numpy.array([self.__DataTensor2D(dataInput=v, maxLen=max_len) for v in ys], dtype=float)
        newYs = torch.FloatTensor(ys)
        return newXs, xsLengths, newYs, ysLengths, zs


class Dataset_IEMOCAP(torch_utils_data.Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class Dataset_BothRepresentation(torch_utils_data.Dataset):
    def __init__(self, data, representation, label):
        self.data, self.representation, self.label = data, representation, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.representation[index], self.label[index]


def Loader_IEMOCAP(includePart=['improve', 'script'], appointGender=None, appointSession=None, batchSize=64,
                   multiFlag=False, shuffleFlag=True):
    def ConcatData(inputData):
        totalConcatData = []
        for sample in inputData:
            delta, deltaDelta = [], []
            for index in range(1, len(sample)):
                delta.append(sample[index] - sample[index - 1])
            delta = numpy.array(delta)
            for index in range(1, len(delta)):
                deltaDelta.append(delta[index] - delta[index - 1])
            deltaDelta = numpy.array(deltaDelta)

            concatData = numpy.concatenate(
                [sample[numpy.newaxis, :-2, :], delta[numpy.newaxis, :-1, :], deltaDelta[numpy.newaxis, :, :]], axis=0)
            totalConcatData.append(concatData)
        return totalConcatData

    loadPath = 'D:/PythonProjects_Data/IEMOCAP/DataSource_Audio/'
    trainData, trainLabel, testData, testLabel = [], [], [], []

    for part in includePart:
        for gender in ['Female', 'Male']:
            for session in range(1, 6):
                currentData = numpy.load(
                    file=os.path.join(loadPath, '%s-%s-Session%d-Data.npy' % (part, gender, session)),
                    allow_pickle=True)
                currentLabel = numpy.load(
                    file=os.path.join(loadPath, '%s-%s-Session%d-Label.npy' % (part, gender, session)),
                    allow_pickle=True)
                currentLabel = numpy.argmax(currentLabel, axis=1)

                if appointGender is not None and gender == appointGender and session == appointSession:
                    testData.extend(currentData)
                    testLabel.extend(currentLabel)
                else:
                    trainData.extend(currentData)
                    trainLabel.extend(currentLabel)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))

    if multiFlag:
        trainDataset = Dataset_IEMOCAP(data=ConcatData(trainData), label=trainLabel)
        if len(testData) != 0: testDataset = Dataset_IEMOCAP(data=ConcatData(testData), label=testLabel)
    else:
        trainDataset = Dataset_IEMOCAP(data=trainData, label=trainLabel)
        if len(testData) != 0: testDataset = Dataset_IEMOCAP(data=testData, label=testLabel)

    ##########################################################

    if len(testData) != 0:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_IEMOCAP()), \
               torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                           collate_fn=Collate_IEMOCAP())
    else:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_IEMOCAP()), None


def Loader_IEMOCAP_UnsupervisedFeatures(appointGender=None, appointSession=None, batchSize=64, metaFlag=False,
                                        multiFlag=False, shuffleFlag=True):
    loadPath = 'D:/PythonProjects_Data/IEMOCAP/UnsupervisedFeatures/'
    data = numpy.load(file=os.path.join(loadPath, 'Reconstruction%s%s.npy' % (
        '_MultiFlag' if multiFlag else '', '_Meta' if metaFlag else '')), allow_pickle=True)
    label = numpy.load(file=os.path.join(loadPath, 'ReconstructionLabel.npy'), allow_pickle=True)

    if appointGender is None and appointSession is None:
        trainDataset = Dataset_IEMOCAP(data=data, label=label)
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_IEMOCAP()), None

    loadPath = 'D:/PythonProjects_Data/IEMOCAP/DataSource_Audio/'
    startPosition = 0

    trainData, trainLabel, testData, testLabel = [], [], [], []
    for part in ['improve', 'script']:
        for gender in ['Female', 'Male']:
            for session in range(1, 6):
                currentLabel = numpy.load(
                    file=os.path.join(loadPath, '%s-%s-Session%d-Label.npy' % (part, gender, session)),
                    allow_pickle=True)
                if gender == appointGender and session == appointSession:
                    testData.extend(data[startPosition:startPosition + len(currentLabel)])
                    testLabel.extend(label[startPosition:startPosition + len(currentLabel)])
                else:
                    trainData.extend(data[startPosition:startPosition + len(currentLabel)])
                    trainLabel.extend(label[startPosition:startPosition + len(currentLabel)])
                startPosition += len(currentLabel)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
    trainDataset = Dataset_IEMOCAP(data=trainData, label=trainLabel)
    testDataset = Dataset_IEMOCAP(data=testData, label=testLabel)
    return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                       collate_fn=Collate_IEMOCAP()), \
           torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                       collate_fn=Collate_IEMOCAP())


def Loader_IEMOCAP_Both(appointGender=None, appointSession=None, batchSize=64, metaFlag=False,
                        multiFlag=False, shuffleFlag=True):
    def ConcatData(inputData):
        totalConcatData = []
        for sample in inputData:
            delta, deltaDelta = [], []
            for index in range(1, len(sample)):
                delta.append(sample[index] - sample[index - 1])
            delta = numpy.array(delta)
            for index in range(1, len(delta)):
                deltaDelta.append(delta[index] - delta[index - 1])
            deltaDelta = numpy.array(deltaDelta)

            concatData = numpy.concatenate(
                [sample[numpy.newaxis, :-2, :], delta[numpy.newaxis, :-1, :], deltaDelta[numpy.newaxis, :, :]], axis=0)
            totalConcatData.append(concatData)
        return totalConcatData

    def OriginDataLoadPart():
        loadPath = 'D:/PythonProjects_Data/IEMOCAP/DataSource_Audio/'
        trainData, trainLabel, testData, testLabel = [], [], [], []

        for part in ['improve', 'script']:
            for gender in ['Female', 'Male']:
                for session in range(1, 6):
                    currentData = numpy.load(
                        file=os.path.join(loadPath, '%s-%s-Session%d-Data.npy' % (part, gender, session)),
                        allow_pickle=True)
                    currentLabel = numpy.load(
                        file=os.path.join(loadPath, '%s-%s-Session%d-Label.npy' % (part, gender, session)),
                        allow_pickle=True)
                    currentLabel = numpy.argmax(currentLabel, axis=1)

                    if appointGender is not None and gender == appointGender and session == appointSession:
                        testData.extend(currentData)
                        testLabel.extend(currentLabel)
                    else:
                        trainData.extend(currentData)
                        trainLabel.extend(currentLabel)
        print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
        return trainData, trainLabel, testData, testLabel

    def UnsupervisedDataLoadPart():
        loadPath = 'D:/PythonProjects_Data/IEMOCAP/UnsupervisedFeatures/'
        data = numpy.load(file=os.path.join(loadPath, 'Reconstruction%s%s.npy' % (
            '_MultiFlag' if multiFlag else '', '_Meta' if metaFlag else '')), allow_pickle=True)
        label = numpy.load(file=os.path.join(loadPath, 'ReconstructionLabel.npy'), allow_pickle=True)

        loadPath = 'D:/PythonProjects_Data/IEMOCAP/DataSource_Audio/'
        startPosition = 0

        trainData, trainLabel, testData, testLabel = [], [], [], []
        for part in ['improve', 'script']:
            for gender in ['Female', 'Male']:
                for session in range(1, 6):
                    currentLabel = numpy.load(
                        file=os.path.join(loadPath, '%s-%s-Session%d-Label.npy' % (part, gender, session)),
                        allow_pickle=True)
                    if gender == appointGender and session == appointSession:
                        testData.extend(data[startPosition:startPosition + len(currentLabel)])
                        testLabel.extend(label[startPosition:startPosition + len(currentLabel)])
                    else:
                        trainData.extend(data[startPosition:startPosition + len(currentLabel)])
                        trainLabel.extend(label[startPosition:startPosition + len(currentLabel)])
                    startPosition += len(currentLabel)
        print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
        return trainData, trainLabel, testData, testLabel

    originTrainData, originTrainLabel, originTestData, originTestLabel = OriginDataLoadPart()
    unsuperTrainData, unsuperTrainLabel, unsuperTestData, unsuperTestLabel = UnsupervisedDataLoadPart()

    if multiFlag:
        trainDataset = Dataset_BothRepresentation(
            data=ConcatData(originTrainData), representation=unsuperTrainData, label=originTrainLabel)
        if len(originTestData) != 0: testDataset = Dataset_BothRepresentation(
            data=ConcatData(originTestData), representation=unsuperTestData, label=originTestLabel)
    else:
        trainDataset = Dataset_BothRepresentation(
            data=originTrainData, representation=unsuperTrainData, label=originTrainLabel)
        if len(originTestData) != 0: testDataset = Dataset_BothRepresentation(
            data=originTestData, representation=unsuperTestData, label=originTestLabel)

    ##########################################################
    if len(originTestData) != 0:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_BothRepresentation()), \
               torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                           collate_fn=Collate_BothRepresentation())
    else:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_BothRepresentation()), None


def Loader_IEMOCAP_OnlyText(appointGender=None, appointSession=None, batchSize=64, shuffleFlag=True):
    loadPath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/DataSource/OnlyText/'
    trainData, trainLabel, testData, testLabel = [], [], [], []
    for part in ['improve', 'script']:
        for gender in ['Female', 'Male']:
            for session in range(1, 6):
                currentLabel = numpy.load(
                    file=os.path.join(loadPath, '%s_Session%d_%s_Label.npy' % (part, session, gender)),
                    allow_pickle=True)
                currentData = numpy.load(
                    file=os.path.join(loadPath, '%s_Session%d_%s_Data.npy' % (part, session, gender)),
                    allow_pickle=True)
                if gender == appointGender and session == appointSession:
                    testData.extend(currentData)
                    testLabel.extend(currentLabel)
                else:
                    trainData.extend(currentData)
                    trainLabel.extend(currentLabel)

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel),
          numpy.sum(trainLabel, axis=0), numpy.sum(testLabel, axis=0))

    trainLabel = numpy.argmax(trainLabel, axis=1)
    trainDataset = Dataset_IEMOCAP(data=trainData, label=trainLabel)
    if len(testData) != 0:
        testLabel = numpy.argmax(testLabel, axis=1)
        testDataset = Dataset_IEMOCAP(data=testData, label=testLabel)
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_OnlyText()), \
               torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                           collate_fn=Collate_OnlyText())
    else:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_OnlyText()), None


if __name__ == '__main__':
    # Loader_IEMOCAP_OnlyText(appointGender='Female', appointSession=1)
    trainDataset, testDataset = Loader_IEMOCAP_OnlyText(appointGender='Female', appointSession=1)
    for batchIndex, [batchData, batchLabel, batchSeq] in enumerate(trainDataset):
        print(batchIndex, numpy.shape(batchData), numpy.shape(batchSeq), numpy.shape(batchLabel))
        # exit()
