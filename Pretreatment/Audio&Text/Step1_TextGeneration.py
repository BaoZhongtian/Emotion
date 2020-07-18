import os
import numpy

if __name__ == '__main__':
    savePath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step1_TextGeneration/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    for sessionIndex in range(1, 6):
        loadPath = 'D:/PythonProjects_Data/IEMOCAP_full_release/Session%d/dialog/transcriptions/' % sessionIndex
        for fileName in os.listdir(loadPath):
            with open(os.path.join(loadPath, fileName), 'r') as file:
                data = file.readlines()
            for sample in data:
                print(sample)
                if sample.find(' [') == -1: continue
                if len(sample[sample.find(']: ') + 3:-1]) == 0: continue
                try:
                    with open(os.path.join(savePath, sample[0:sample.find(' [')] + '.txt'), 'w') as file:
                        file.write(sample[sample.find(']: ') + 3:-1])
                except:
                    pass
