import numpy
import os
import matplotlib.pylab as plt

if __name__ == '__main__':
    scope = [611, 1508, 5377, 215, 546]
    for index in range(1, len(scope)):
        scope[index] += scope[index - 1]
    print(scope)
    for attentionName in ['StandardAttention']:
        for head in [2, 4, 8, 16]:
            loadpath = 'D:/PythonProjects_Data/FAU-AEC/TCN-Head%d/%s-TestResult/' % (head, attentionName)
            UAList, WAList = [], []
            for filename in os.listdir(loadpath)[99:]:
                data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',')
                confusionMatrix = numpy.zeros([5, 5])
                predictProbability = data[:, 0:5]
                predict = numpy.argmax(predictProbability, axis=1)

                counter = 0
                for index in range(len(predict)):
                    if index < scope[counter]:
                        label = counter
                    else:
                        counter += 1
                        label = counter
                    confusionMatrix[label][predict[index]] += 1

                # print(confusionMatrix)
                ua = numpy.average([item[idx] / numpy.sum(item) for idx, item in enumerate(confusionMatrix)])
                wa = numpy.sum(predict == data[:, 5]) / len(predict)
                UAList.append(ua)
                WAList.append(wa)
            print('%.2f%%\t%.2f%%' % (max(UAList) * 100, max(WAList) * 100))
