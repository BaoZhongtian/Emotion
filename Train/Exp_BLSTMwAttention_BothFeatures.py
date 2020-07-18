from Auxiliary.Loader import Loader_IEMOCAP_Both
from Model.BLSTMwMultiAttention import BLSTMwMultiAttention
from Auxiliary.TrainTemplate import Template_FluctuateSize_BothFeatures

if __name__ == '__main__':
    cudaFlag = True
    metaFlag, multiFlag = False, True
    for attentionName in ['StandardAttention', 'LocalAttention', 'ComponentAttention', 'MonotonicAttention']:
        for appointGender in ['Female', 'Male']:
            for appointSession in range(1, 6):
                Model = BLSTMwMultiAttention(
                    attentionName=[attentionName, attentionName], attentionScope=[10, 10],
                    attentionParameter=['LeftAttention', 'RightAttention'], featuresNumber=120 if multiFlag else 40,
                    cudaFlag=cudaFlag)
                savePath = 'D:/PythonProjects_Data/IEMOCAP_Result_Test/BLSTMwBothAttention%s%s/%s/%s-%d' % (
                    '_Meta' if metaFlag else '', '_Multi' if multiFlag else '', attentionName, appointGender,
                    appointSession)
                trainDataset, testDataset = Loader_IEMOCAP_Both(
                    appointGender=appointGender, appointSession=appointSession, batchSize=32, metaFlag=metaFlag,
                    multiFlag=multiFlag)

                Template_FluctuateSize_BothFeatures(Model=Model, trainDataset=trainDataset, testDataset=testDataset,
                                                    cudaFlag=cudaFlag, savePath=savePath)
