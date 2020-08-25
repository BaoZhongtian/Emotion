from Model.TCN import TCN, TCN_DropOut
from Auxiliary.Loader import Loader_IEMOCAP
from Auxiliary.TrainTemplate import Template_FluctuateSize

if __name__ == '__main__':
    cudaFlag = True
    # metaFlag, multiFlag = False, False
    for attentionName in ['SelfAttention']:
        for appointGender in ['Female', 'Male']:
            for appointSession in range(1, 6):
                # Model = TCN(attentionName=attentionName, attentionScope=0, cudaFlag=cudaFlag)
                Model = TCN_DropOut(attentionName=attentionName, attentionScope=0, cudaFlag=True, labelNumber=4)
                trainDataset, testDataset = Loader_IEMOCAP(appointGender=appointGender, appointSession=appointSession,
                                                           multiFlag=True, batchSize=16)
                savePath = 'D:/PythonProjects_Data/IEMOCAP_Result_Test/TCN-DropOut/%s/%s-%d' % (
                    attentionName, appointGender, appointSession)

                Template_FluctuateSize(Model=Model, trainDataset=trainDataset, testDataset=testDataset,
                                       cudaFlag=cudaFlag, savePath=savePath)
