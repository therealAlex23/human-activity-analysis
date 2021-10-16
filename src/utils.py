import numpy as np
import matplotlib.pyplot as plt


def extractPartData(dir, numPart):
    """
    Read participant data from .csv files
    All 5 device files belonging to a participant are joined
    into a single array.
    """
    fullDir = dir + str(numPart) + "/part" + str(numPart) + "dev"
    partData = np.genfromtxt(fullDir + "1.csv", delimiter=',')
    for sens in range(2, 6):
        sensData = np.genfromtxt(fullDir + str(sens) + ".csv", delimiter=',')
        partData = np.append(partData, sensData, axis=0)
    return partData


def getAllPartData(dir, maxPart):
    allData = extractPartData(dir, 0)
    for part in range(1, maxPart):
        data = extractPartData(dir, part)
        allData = np.append(allData, data, axis=0)
    return allData


# def getSensorModuleArray(data, startIndex):
#    return np.linalg.norm(data[:, startIndex:startIndex + 3], axis=1)

def getActivityMod(data, startIndex, activityIndex):
    activityVector = np.where(data[:, -1] == activityIndex)
    print(activityVector)
    # return np.linalg.norm(activityVector[:, startIndex:startIndex + 3], axis=1)


def drawBoxPlot(data):
    _, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    plt.show()
