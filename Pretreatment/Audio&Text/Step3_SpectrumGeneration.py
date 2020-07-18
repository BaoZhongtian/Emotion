import os
import librosa
from scipy import signal
import numpy

if __name__ == '__main__':
    m_bands = 40
    for part in ['improve', 'script']:
        for gender in ['Female', 'Male']:
            loadpath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step2_AudioChoosed/%s/%s/' % (part, gender)
            savepath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step3_SpectrumGeneration/%s/%s/' % (part, gender)

            s_rate = 16000
            win_length = int(0.025 * s_rate)  # Window length 15ms, 25ms, 50ms, 100ms, 200ms
            hop_length = int(0.010 * s_rate)  # Window shift  10ms
            n_fft = win_length

            for sessionName in os.listdir(loadpath):
                for emotionName in os.listdir(os.path.join(loadpath, sessionName)):
                    if os.path.exists(os.path.join(savepath, sessionName, emotionName)): continue
                    os.makedirs(os.path.join(savepath, sessionName, emotionName))
                    for filename in os.listdir(os.path.join(loadpath, sessionName, emotionName)):
                        print('Treating', sessionName, emotionName, filename)
                        y, sr = librosa.load(path=os.path.join(loadpath, sessionName, emotionName, filename), sr=s_rate)
                        D = numpy.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                                   window=signal.hamming, center=False)) ** 2
                        S = librosa.feature.melspectrogram(S=D, n_mels=m_bands)
                        gram = librosa.power_to_db(S, ref=numpy.max)
                        gram = numpy.transpose(gram, (1, 0))

                        with open(os.path.join(savepath, sessionName, emotionName, filename.replace('.wav', '.csv')),
                                  'w') as file:
                            for indexX in range(len(gram)):
                                for indexY in range(len(gram[indexX])):
                                    if indexY != 0: file.write(',')
                                    file.write(str(gram[indexX][indexY]))
                                file.write('\n')
