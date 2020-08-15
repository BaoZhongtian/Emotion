import os
from Auxiliary.Tools import SearchFold

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/IEMOCAP/Text/Step1_SeparateFold/'
    savePath = 'D:/PythonProjects_Data/IEMOCAP/Text/Step2_CMULabel/'
    dictionary = {}

    with open('Dictionary.txt', 'r') as file:
        data = file.readlines()
        for sample in data:
            sample = sample[0:-1].split('  ')
            dictionary[sample[0]] = sample[1]
    # print(dictionary)

    for filePath in SearchFold(loadPath):
        print(filePath)
        saveFoldPath = filePath[:-filePath[::-1].find('\\')].replace(loadPath, savePath)
        if not os.path.exists(saveFoldPath): os.makedirs(saveFoldPath)

        with open(filePath, 'r') as file:
            data = file.read()
        data = data.upper()
        replacedData = ''
        for sample in data:
            if sample == "'" or sample == ' ' or 'A' <= sample <= 'Z':
                replacedData += sample
        replacedData = replacedData.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        replacedData = replacedData.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        replacedData = replacedData.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        replacedData = replacedData.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        replacedData = replacedData.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')

        replacedData = replacedData.split(' ')

        with open(filePath.replace(loadPath, savePath), 'w') as file:
            for sample in replacedData:
                if sample in dictionary.keys(): file.write(dictionary[sample] + ' ')

        # exit()
