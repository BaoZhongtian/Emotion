import os
import numpy

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step5_TextDigitalize/'
    savePath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/DataSource/OnlyText/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    emotionDictionary = {'ang': [1, 0, 0, 0], 'exc': [0, 1, 0, 0], 'hap': [0, 1, 0, 0], 'neu': [0, 0, 1, 0],
                         'sad': [0, 0, 0, 1]}
    for partName in os.listdir(loadPath):
        for genderName in os.listdir(os.path.join(loadPath, partName)):
            for sessionName in os.listdir(os.path.join(loadPath, partName, genderName)):
                partData, partLabel = [], []
                for emotionName in os.listdir(os.path.join(loadPath, partName, genderName, sessionName)):
                    for fileName in os.listdir(os.path.join(loadPath, partName, genderName, sessionName, emotionName)):
                        data = numpy.genfromtxt(
                            fname=os.path.join(loadPath, partName, genderName, sessionName, emotionName, fileName),
                            dtype=int, delimiter=',')[0:-1]
                        partData.append(data)
                        partLabel.append(emotionDictionary[emotionName])
                        # print(data)
                        # exit()
                print(numpy.shape(partData), numpy.shape(partLabel), numpy.sum(partLabel, axis=0))
                numpy.save(file=os.path.join(savePath, partName + '_' + sessionName + '_' + genderName + '_Data.npy'),
                           arr=partData)
                numpy.save(file=os.path.join(savePath, partName + '_' + sessionName + '_' + genderName + '_Label.npy'),
                           arr=partLabel)
