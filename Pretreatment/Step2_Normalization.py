import os
import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    totalData = []

    for part in ['improve', 'script']:
        for gender in ['Female', 'Male']:
            loadpath = 'D:/PythonProjects_Data/IEMOCAP/Step1_SpectrumGeneration/%s/%s/' % (part, gender)
            savepath = 'D:/PythonProjects_Data/IEMOCAP/Step2_Normalization/%s/%s/' % (part, gender)
            for sessionName in os.listdir(loadpath):
                for emotionName in os.listdir(os.path.join(loadpath, sessionName)):
                    for filename in os.listdir(os.path.join(loadpath, sessionName, emotionName)):
                        data = numpy.genfromtxt(fname=os.path.join(loadpath, sessionName, emotionName, filename),
                                                dtype=float, delimiter=',')
                        totalData.extend(data)
                    print('Loading', sessionName, emotionName, numpy.shape(totalData))

    print(numpy.shape(totalData))
    totalData = scale(totalData)
    startPosition = 0

    for part in ['improve', 'script']:
        for gender in ['Female', 'Male']:
            loadpath = 'D:/PythonProjects_Data/IEMOCAP/Step1_SpectrumGeneration/%s/%s/' % (part, gender)
            savepath = 'D:/PythonProjects_Data/IEMOCAP/Step2_Normalization/%s/%s/' % (part, gender)
            for sessionName in os.listdir(loadpath):
                for emotionName in os.listdir(os.path.join(loadpath, sessionName)):
                    os.makedirs(os.path.join(savepath, sessionName, emotionName))
                    for filename in os.listdir(os.path.join(loadpath, sessionName, emotionName)):
                        data = numpy.genfromtxt(fname=os.path.join(loadpath, sessionName, emotionName, filename),
                                                dtype=float, delimiter=',')
                        writeData = totalData[startPosition:startPosition + len(data)]

                        with open(os.path.join(savepath, sessionName, emotionName, filename), 'w') as file:
                            for indexX in range(numpy.shape(writeData)[0]):
                                for indexY in range(numpy.shape(writeData)[1]):
                                    if indexY != 0: file.write(',')
                                    file.write(str(writeData[indexX][indexY]))
                                file.write('\n')

                        startPosition += len(data)
                    print('Writing', sessionName, emotionName, startPosition)
