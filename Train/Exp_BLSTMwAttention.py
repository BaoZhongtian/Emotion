from Model.BLSTMwAttention import BLSTMwAttention
from Auxiliary.TrainTemplate import Template_FluctuateSize
from Auxiliary.Loader import Loader_IEMOCAP

if __name__ == '__main__':
    cudaFlag = True
    # metaFlag, multiFlag = False, False
    for attentionName in ['StandardAttention']:
        for appointGender in ['Female', 'Male']:
            for appointSession in range(1, 6):
                Model = BLSTMwAttention(attentionName=attentionName, attentionScope=10, featuresNumber=120,
                                        classNumber=4, cudaFlag=cudaFlag)
                # savePath = 'D:/PythonProjects_Data/IEMOCAP_Result_Test/BLSTMwAttention%s%s/%s/%s-%d' % (
                #     '_Meta' if metaFlag else '', '_Multi' if multiFlag else '', attentionName, appointGender,
                #     appointSession)
                savePath = 'D:/PythonProjects_Data/IEMOCAP_Result_Test/BLSTMwAttention/%s/%s-%d' % (
                    attentionName, appointGender, appointSession)
                # trainDataset, testDataset = Loader_IEMOCAP_UnsupervisedFeatures(
                #     appoingGender=appointGender, appoingSession=appointSession, metaFlag=metaFlag, multiFlag=multiFlag)
                trainDataset, testDataset = Loader_IEMOCAP(appointGender=appointGender, appointSession=appointSession,
                                                           batchSize=16, multiFlag=True)

                Template_FluctuateSize(Model=Model, trainDataset=trainDataset, testDataset=testDataset,
                                       cudaFlag=cudaFlag, savePath=savePath)
