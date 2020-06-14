import os
import shutil

if __name__ == '__main__':
    loadPath = '/home/bztbztbzt/IEMOCAP/BLSTMwAttention/StandardAttention/'
    savePath = '/home/bztbztbzt/IEMOCAP/BLSTMwAttention/StandardAttention-Result/'
    os.makedirs(savePath)
    for part in os.listdir(loadPath):
        shutil.copytree(src=os.path.join(loadPath, part, '-TestResult'),
                        dst=os.path.join(savePath, part + '-TestResult'))
