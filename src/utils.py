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

def getActivityMod(data, startIndex, activityIndex, sensorID):
    filterArrAct = data[:,11] == activityIndex
    activityData = data[:][filterArrAct] 
    filterArrSens = activityData[:,0] == sensorID
    activityData = activityData[:][filterArrSens] 
    
    """
    Nao da para fazer da maneira --> data[:,filterArr] 
    Se usar dessa maneira gera-me este erro --> IndexError: boolean index did not match indexed array along dimension 1; dimension is 12 but corresponding boolean dimension is 3930873
    Perguntar ao prof só para saber o porque 
    """

    return np.linalg.norm(activityData[:, startIndex:startIndex + 3], axis=1)

def getDensityOutliers():
    pass

def getBoxPlotModuleActivity(moduleName,allData,activities,startIndex,sensorID):

    """
    Draws a figure with multiples boxplots 
    Each one contains a dataset of a variable module by activity from one sensor
    """

    plt.figure()
    ticks = activities.values()
    positionsBox = range(0, len(ticks) * 5, 5)
    for act in activities.keys():
        modActData = getActivityMod(allData,startIndex,act,sensorID)
        bp = plt.boxplot(modActData, positions = [positionsBox[act-1]], widths=0.6)
        setBoxColors(bp,"red")
    plt.xticks(range(0, len(ticks) * 5, 5), ticks,rotation ='vertical')
    plt.xlim(-2, len(ticks)*5)
    plt.ylim(0,35)
    plt.show()
    #plt.savefig('boxcompare.png')
    pass

def setBoxColors(bp,color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)



def drawBoxPlot(data):
    _, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    plt.show()
