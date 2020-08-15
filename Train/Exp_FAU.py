from Model.TCN import TCN, TCN_Multi
from Auxiliary.Loader import Loader_FAUAEC
from Auxiliary.TrainTemplate import Template_FluctuateSize

if __name__ == '__main__':
    cudaFlag = True
    headNumber = 2
    for attentionName in ['SelfAttention']:
        # Model = TCN(attentionName=attentionName, attentionScope=0, cudaFlag=cudaFlag, labelNumber=5)
        Model = TCN_Multi(attentionName=[attentionName for _ in range(headNumber)],
                          attentionScope=[0 for _ in range(headNumber)],
                          attentionParameter=['Head%02d' % index for index in range(headNumber)],
                          cudaFlag=cudaFlag, labelNumber=5)
        trainDataset, testDataset = Loader_FAUAEC(multiFlag=True, batchSize=32, appointShape=500)
        savePath = 'D:/PythonProjects_Data/FAU-AEC/TCN-Head%d/%s' % (headNumber, attentionName)

        Template_FluctuateSize(Model=Model, trainDataset=trainDataset, testDataset=testDataset,
                               cudaFlag=cudaFlag, savePath=savePath, labelWeight=[1.1, 0.5, 0.2, 1.5, 1.4])
