import os
import numpy
from Model.HierarchyNetwork import HierarchyNetwork
from Auxiliary.Loader import Loader_IEMOCAP_OnlyText
from Auxiliary.TrainTemplate import Template_FluctuateSize

if __name__ == '__main__':
    cudaFlag = True
    # metaFlag, multiFlag = False, False
    for attentionName in ['StandardAttention', 'LocalAttention', 'ComponentAttention', 'MonotonicAttention']:
        for appointGender in ['Female', 'Male']:
            for appointSession in range(1, 6):
                Model = HierarchyNetwork(attentionName=attentionName, attentionScope=10, cudaFlag=cudaFlag)
                trainDataset, testDataset = Loader_IEMOCAP_OnlyText(appointGender=appointGender,
                                                                    appointSession=appointSession)
                savePath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/OnlyText/%s/%s-%d' % (
                    attentionName, appointGender, appointSession)
                Template_FluctuateSize(Model=Model, trainDataset=trainDataset, testDataset=testDataset,
                                       cudaFlag=cudaFlag, savePath=savePath)
