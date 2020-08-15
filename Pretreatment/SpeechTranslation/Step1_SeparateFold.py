import os
import shutil

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/IEMOCAP_full_release/Session%d/dialog/'
    savePath = 'D:/PythonProjects_Data/IEMOCAP/Text/Step1_SeparateFold/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    for sessionNumber in range(1, 6):
        for fileName in os.listdir(os.path.join(loadPath % sessionNumber, 'EmoEvaluation')):
            if fileName.find('._') != -1: continue
            if fileName[-3:] != 'txt': continue
            print(fileName)
            emotionDictionary = {}

            with open(os.path.join(loadPath % sessionNumber, 'EmoEvaluation', fileName), 'r') as file:
                emotionData = file.readlines()
            with open(os.path.join(loadPath % sessionNumber, 'transcriptions', fileName), 'r') as file:
                transcriptData = file.readlines()

            for readLine in emotionData:
                if readLine[0] != '[': continue
                readLine = readLine.split('\t')
                if readLine[2] not in ['ang', 'hap', 'neu', 'sad', 'exc']: continue
                emotionDictionary[readLine[1]] = readLine[2]

            for readLine in transcriptData:
                # print(readLine)
                try:
                    fileName = readLine.split(' [')[0]
                    transcript = readLine[0:-1].split(']: ')[1]
                    if not os.path.exists(os.path.join(savePath, 'Session0%s-%s' % (fileName[4], fileName[-4]),
                                                       emotionDictionary[fileName])):
                        os.makedirs(os.path.join(savePath, 'Session0%s-%s' % (fileName[4], fileName[-4]),
                                                 emotionDictionary[fileName]))
                except:
                    continue
                with open(os.path.join(os.path.join(savePath, 'Session0%s-%s' % (fileName[4], fileName[-4]),
                                                    emotionDictionary[fileName], fileName + '.txt')), 'w') as file:
                    file.write(transcript)

            # exit()
