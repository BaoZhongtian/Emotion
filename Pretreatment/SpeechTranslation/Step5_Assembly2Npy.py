import os
import numpy

if __name__ == '__main__':
    dataPath = 'D:/PythonProjects_Data/IEMOCAP/Step2_Normalization/'
    labelPath = 'D:/PythonProjects_Data/IEMOCAP/Text/Step3_Digitalize/'
    savePath = 'D:/PythonProjects_Data/IEMOCAP/Text/SER-CTC-EmotionAmplify/'
    os.makedirs(savePath)

    for partName in ['impro', 'script']:
        for genderName in os.listdir(os.path.join(dataPath, partName)):
            for sessionName in os.listdir(os.path.join(dataPath, partName, genderName)):
                totalData, totalLabel = [], []
                for emotionName in os.listdir(os.path.join(dataPath, partName, genderName, sessionName)):
                    for fileName in os.listdir(os.path.join(dataPath, partName, genderName, sessionName, emotionName)):
                        # print(partName, sessionName, genderName, emotionName, fileName)
                        # exit()
                        # print(genderName)
                        # exit()
                        currentData = numpy.genfromtxt(
                            fname=os.path.join(dataPath, partName, genderName, sessionName, emotionName, fileName),
                            dtype=float, delimiter=',')
                        currentLabel = numpy.reshape(numpy.genfromtxt(
                            fname=os.path.join(labelPath, partName, '%s-%s' % (sessionName, genderName[0]),
                                               emotionName if emotionName != 'exc' else 'hap', fileName), dtype=float,
                            delimiter=','), [-1])[0:-1]
                        print(currentLabel)
                        if len(currentLabel) == 0: continue
                        totalData.append(currentData)
                        totalLabel.append(currentLabel)
                numpy.save(file=os.path.join(savePath, '%s-%s-%s-Data.npy' % (partName, sessionName, genderName)),
                           arr=totalData)
                numpy.save(file=os.path.join(savePath, '%s-%s-%s-Label.npy' % (partName, sessionName, genderName)),
                           arr=totalLabel)
