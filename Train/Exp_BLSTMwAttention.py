from Model.BLSTMwAttention import BLSTMwAttention
from Auxiliary.TrainTemplate import Template_FluctuateSize, Template_Fluctuate_BestChoose
from Auxiliary.Loader import Loader_IEMOCAP_UnsupervisedFeatures, Loader_IEMOCAP

if __name__ == '__main__':
    cudaFlag = True
    # metaFlag, multiFlag = False, False
    for attentionName in ['StandardAttention', 'LocalAttention', 'ComponentAttention', 'MonotonicAttention']:
        for appointGender in ['Female', 'Male']:
            for appointSession in range(1, 6):
                Model = BLSTMwAttention(attentionName=attentionName, attentionScope=10, featuresNumber=40,
                                        classNumber=4, cudaFlag=cudaFlag)
                # savePath = 'D:/PythonProjects_Data/IEMOCAP_Result_Test/BLSTMwAttention%s%s/%s/%s-%d' % (
                #     '_Meta' if metaFlag else '', '_Multi' if multiFlag else '', attentionName, appointGender,
                #     appointSession)
                savePath = 'D:/PythonProjects_Data/IEMOCAP_Result_Test/BLSTMwAttention_Single/%s/%s-%d' % (
                    attentionName, appointGender, appointSession)
                # trainDataset, testDataset = Loader_IEMOCAP_UnsupervisedFeatures(
                #     appoingGender=appointGender, appoingSession=appointSession, metaFlag=metaFlag, multiFlag=multiFlag)
                trainDataset, testDataset = Loader_IEMOCAP(appointGender=appointGender, appointSession=appointSession,
                                                           batchSize=32)

                Template_Fluctuate_BestChoose(Model=Model, trainDataset=trainDataset, testDataset=testDataset,
                                              cudaFlag=cudaFlag, savePath=savePath)
