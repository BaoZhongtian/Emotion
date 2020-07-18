import os
import shutil


def FoldSearcher(loadPath):
    if os.path.isfile(loadPath): return [loadPath]

    filePaths = []
    for sample in os.listdir(loadPath):
        filePaths.extend(FoldSearcher(os.path.join(loadPath, sample)))
    return filePaths


if __name__ == '__main__':
    textPath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step1_TextGeneration/'
    textSavePath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step2_TextChoosed/'
    audioPath = 'D:/PythonProjects_Data/IEMOCAP/IEMOCAP-Voices-Choosed/'
    audioSavePath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step2_AudioChoosed/'

    for filePath in FoldSearcher(audioPath):
        fileName = filePath[-filePath[::-1].find('\\'):]
        if not os.path.exists(os.path.join(textPath, fileName.replace('wav', 'txt'))): continue
        saveFoldPath = filePath[0:filePath.find(fileName)].replace(audioPath, audioSavePath)
        if not os.path.exists(saveFoldPath): os.makedirs(saveFoldPath)
        saveFoldPath = filePath[0:filePath.find(fileName)].replace(audioPath, textSavePath)
        if not os.path.exists(saveFoldPath): os.makedirs(saveFoldPath)
        shutil.copy(src=filePath, dst=filePath.replace(audioPath, audioSavePath))
        shutil.copy(src=os.path.join(textPath, fileName.replace('wav', 'txt')),
                    dst=filePath.replace(audioPath, textSavePath).replace('wav', 'txt'))
        # exit()
