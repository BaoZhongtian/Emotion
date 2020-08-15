from Model.TCN import TCN_Multi
from Auxiliary.Loader import Loader_IEMOCAP
from Auxiliary.TrainTemplate import Template_FluctuateSize

if __name__ == '__main__':
    cudaFlag = True
    headNumber = 2
    # metaFlag, multiFlag = False, False
    for attentionName in ['SelfAttention']:
        for appointGender in ['Female', 'Male']:
            for appointSession in range(1, 6):
                Model = TCN_Multi(attentionName=[attentionName for _ in range(headNumber)],
                                  attentionScope=[0 for _ in range(headNumber)],
                                  attentionParameter=['Head%02d' % index for index in range(headNumber)],
                                  cudaFlag=cudaFlag)
                trainDataset, testDataset = Loader_IEMOCAP(appointGender=appointGender, appointSession=appointSession,
                                                           multiFlag=True, batchSize=16, appointShape=500)
                savePath = 'D:/PythonProjects_Data/IEMOCAP_Result_Test/TCN-Head%02d-500/%s/%s-%d' % (
                    headNumber, attentionName, appointGender, appointSession)

                Template_FluctuateSize(Model=Model, trainDataset=trainDataset, testDataset=testDataset,
                                       cudaFlag=cudaFlag, savePath=savePath)
