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


class Dataset_IEMOCAP(torch_utils_data.Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def Loader_IEMOCAP(includePart=['improve', 'script'], appointGender=None, appointSession=None, batchSize=64,
                   multiFlag=False):
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
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True,
                                           collate_fn=Collate_IEMOCAP()), \
               torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                           collate_fn=Collate_IEMOCAP())
    else:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True,
                                           collate_fn=Collate_IEMOCAP()), None


if __name__ == '__main__':
    trainDataset, testDataset = Loader_IEMOCAP(appointGender='Female', appointSession=1, multiFlag=True)
    for batchIndex, [batchData, batchSeq, _] in enumerate(trainDataset):
        print(numpy.shape(batchData), batchSeq)
        exit()
