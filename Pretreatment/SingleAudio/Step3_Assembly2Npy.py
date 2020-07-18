import os
import numpy

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/IEMOCAP/Step2_Normalization/'
    savePath = 'D:/PythonProjects_Data/IEMOCAP/DataSource_Audio/'
    # os.makedirs(savePath)

    emotionDictionary = {'ang': 0, 'hap': 1, 'exc': 1, 'neu': 2, 'sad': 3}

    for partName in os.listdir(loadPath):
        for gender in os.listdir(os.path.join(loadPath, partName)):
            for sessionName in os.listdir(os.path.join(loadPath, partName, gender)):
                totalData, totalLabel = [], []
                for emotionName in os.listdir(os.path.join(loadPath, partName, gender, sessionName)):
                    label = [0, 0, 0, 0]
                    label[emotionDictionary[emotionName]] = 1
                    for fileName in os.listdir(os.path.join(loadPath, partName, gender, sessionName, emotionName)):
                        data = numpy.genfromtxt(
                            fname=os.path.join(loadPath, partName, gender, sessionName, emotionName, fileName),
                            dtype=float, delimiter=',')
                        totalData.append(data)
                        totalLabel.append(label)

                print(numpy.shape(totalData), numpy.shape(totalLabel), numpy.sum(totalLabel, axis=0))
                numpy.save(file=os.path.join(savePath, '%s-%s-%s-Data.npy' % (partName, gender, sessionName)),
                           arr=totalData)
                numpy.save(file=os.path.join(savePath, '%s-%s-%s-Label.npy' % (partName, gender, sessionName)),
                           arr=totalLabel)
