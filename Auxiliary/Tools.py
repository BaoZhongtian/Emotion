import os
import torch


def SaveNetwork(model, optimizer, savePath):
    torch.save(obj={'ModelStateDict': model.state_dict(), 'OptimizerStateDict': optimizer.state_dict()},
               f=savePath + '-Parameter.pkl')
    torch.save(obj=model, f=savePath + '-Network.pkl')


def LoadNetwork(model, optimizer, loadPath):
    checkpoint = torch.load(loadPath)
    model.load_state_dict(checkpoint['ModelStateDict'])
    optimizer.load_state_dict(checkpoint['OptimizerStateDict'])
    return model, optimizer


def SearchFold(loadPath):
    totalPath = []
    for fileName in os.listdir(loadPath):
        if os.path.isfile(os.path.join(loadPath, fileName)):
            totalPath.append(os.path.join(loadPath, fileName))
        else:
            totalPath.extend(SearchFold(os.path.join(loadPath, fileName)))
    return totalPath
