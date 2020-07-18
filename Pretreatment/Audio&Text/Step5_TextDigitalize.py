import os
import numpy

if __name__ == '__main__':
    loadPath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step2_TextChoosed/'
    savePath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step5_TextDigitalize/'
    dictionary = {'<PAD>': 0}

    for partName in os.listdir(loadPath):
        for genderName in os.listdir(os.path.join(loadPath, partName)):
            for sessionName in os.listdir(os.path.join(loadPath, partName, genderName)):
                for emotionName in os.listdir(os.path.join(loadPath, partName, genderName, sessionName)):
                    if not os.path.exists(os.path.join(savePath, partName, genderName, sessionName, emotionName)):
                        os.makedirs(os.path.join(savePath, partName, genderName, sessionName, emotionName))

                    for fileName in os.listdir(os.path.join(loadPath, partName, genderName, sessionName, emotionName)):
                        data = numpy.genfromtxt(
                            fname=os.path.join(loadPath, partName, genderName, sessionName, emotionName, fileName),
                            dtype=str, delimiter=' ').reshape([-1])

                        with open(os.path.join(savePath, partName, genderName, sessionName, emotionName,
                                               fileName.replace('txt', 'csv')), 'w') as file:
                            maxLen = max([len(word) for word in data])
                            for word in data:
                                for character in word:
                                    character = character.lower()
                                    if character not in dictionary.keys():
                                        dictionary[character] = len(dictionary.keys())
                                    file.write(str(dictionary[character]) + ',')
                                for index in range(len(word), maxLen):
                                    file.write('0,')
                                file.write('\n')
    with open('Dictionary.csv', 'w') as file:
        for sample in dictionary.keys():
            file.write(str(sample) + '#####' + str(dictionary[sample]) + '\n')
