import os
from Auxiliary.Tools import SearchFold

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/IEMOCAP/Text/Step2_CMULabel'
    savePath = 'D:/PythonProjects_Data/IEMOCAP/Text/Step3_Digitalize'

    dictionary = {}

    for filePath in SearchFold(loadPath):
        with open(filePath, 'r') as file:
            data = file.read()
        data = data[0:-1].replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '')
        data = data.split(' ')
        for sample in data:
            if sample in dictionary.keys(): continue
            dictionary[sample] = len(dictionary.keys()) + 1

    print(dictionary)
    exit()

    for filePath in SearchFold(loadPath):
        print(filePath)
        if filePath.find('impro') != -1: part = 'impro'
        if filePath.find('script') != -1: part = 'script'

        saveFoldPath = filePath[:-filePath[::-1].find('\\')].replace(loadPath, savePath + '\\' + part). \
            replace('exc', 'hap').replace('Session0', 'Session')
        if not os.path.exists(saveFoldPath): os.makedirs(saveFoldPath)
        with open(filePath, 'r') as file:
            data = file.read()
        data = data[0:-1].replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '')
        data = data.split(' ')

        with open(
                filePath.replace(loadPath, savePath + '\\' + part).replace('txt', 'csv').replace('exc', 'hap').replace(
                    'Session0', 'Session'), 'w') as file:
            for sample in data:
                if sample == '': continue
                file.write(str(dictionary[sample]) + ',')
