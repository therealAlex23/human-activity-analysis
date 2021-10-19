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
    filterArrAct = data[:, 11] == activityIndex
    activityData = data[:][filterArrAct]
    filterArrSens = activityData[:, 0] == sensorID
    activityData = activityData[:][filterArrSens]

    return np.linalg.norm(activityData[:, startIndex:startIndex + 3], axis=1)


def getDensityOutliers():
    pass


def getBoxPlotModuleActivity(moduleName, allData, activities, startIndex, sensorID):
    """
    Draws a figure with multiples boxplots
    Each one contains a dataset of a variable module by activity from one sensor
    """
    plt.figure()
    ticks = activities.values()
    positionsBox = range(0, len(ticks) * 5, 5)
    for act in activities.keys():
        modActData = getActivityMod(allData, startIndex, act, sensorID)
        bp = plt.boxplot(modActData, positions=[
                         positionsBox[act-1]], widths=0.6)
        setBoxColors(bp, "red")
    plt.xticks(range(0, len(ticks) * 5, 5), ticks, rotation='vertical')
    plt.xlim(-2, len(ticks)*5)
    plt.ylim(0, 35)
    plt.show()
    # plt.savefig('boxcompare.png')
    pass


def setBoxColors(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def drawBoxPlot(data):
    _, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    plt.show()


def zscore(arr, k):
    """
    Devolve os indices dos outliers
    """
    zscore = (arr - np.mean(arr)) / np.std(arr)
    filter = np.abs(zscore) > k
    return np.where(filter)[0]


def plotScatter(ax, x, y, title):
    ax.plot(y, '.', color='blue')
    ax.plot(x, y[x], '.', color='red')
    ax.set_title(title)


# deviceId, acc[x, y, z], gyro[x, y, z], mag[x, y, z], timestamp, activityIndex
def plotOutliers(data, k, activity_index, sensor_id, axs):
    acc_mod = getActivityMod(data, 1, activity_index, sensor_id)
    gyro_mod = getActivityMod(data, 4, activity_index, sensor_id)
    mag_mod = getActivityMod(data, 7, activity_index, sensor_id)

    print(zscore(mag_mod, k))
    print(gyro_mod[zscore(mag_mod, k)])

    plotScatter(axs[0], zscore(acc_mod, k), acc_mod, "ACC")
    plotScatter(axs[1], zscore(gyro_mod, k), gyro_mod, "GYRO")
    plotScatter(axs[2], zscore(mag_mod, k), mag_mod, "MAG")
