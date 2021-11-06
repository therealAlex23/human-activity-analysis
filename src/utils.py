import numpy as np
from mpl_toolkits import mplot3d
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


def getVectorModule(data, startIndex):
    return np.linalg.norm(data[:, startIndex:startIndex + 3], axis=1)


def getActivityData(data, activityIndex, sensorID):
    """
    Returns measurements (x, y, z) for acc, gyro and mag
    """
    filterArrAct = data[:, 11] == activityIndex
    activityData = data[:][filterArrAct]
    if sensorID is not None:
        filterArrSens = activityData[:, 0] == sensorID
        activityData = activityData[:][filterArrSens]
    return activityData


def getDensityOutliers(allData, activities, sensorID):
    getBoxPlotModuleActivity("Aceleração", allData, activities, 1, sensorID)
    getBoxPlotModuleActivity("Giroscopio", allData, activities, 4, sensorID)
    getBoxPlotModuleActivity("Magnetometro", allData, activities, 7, sensorID)
    plt.show()


def getBoxPlotModuleActivity(moduleName, allData, activities, startIndex, sensorID):
    """
    Draws figures with multiples boxplots
    Each one contains a dataset of the acceleration/gyroscope/magnetometer module by activity
    Detected by one sensor identified by the sensorID
    """

    fig = plt.figure()
    fig.suptitle("Modulo " + moduleName)
    ticks = activities.values()
    positionsBox = range(0, len(ticks) * 5, 5)

    print("-----------------MODULO DE " +
          moduleName+"-----------------------\n")
    yMaxLimit = 0
    yMinLImit = 0

    for act in activities.keys():
        print("ACTIVITY: " + activities[act]+"\n")
        modActData = getVectorModule(getActivityData(
            allData, act, sensorID), startIndex)

        maxValueArr = np.max(modActData)
        minValueArr = np.min(modActData)
        if maxValueArr > yMaxLimit:
            yMaxLimit = maxValueArr
        if minValueArr < yMinLImit:
            yMinLImit = minValueArr

        totalPoints = len(modActData)

        print("Numero total de pontos: " + str(totalPoints))
        bp = plt.boxplot(modActData, positions=[
                         positionsBox[act-1]], widths=0.6)

        totalOutliers = len(bp["fliers"][0].get_data()[1])

        print("Numero de Outliers: " + str(totalOutliers))
        dens = (totalOutliers/totalPoints) * 100

        print("Densidade de outliers: " + str(dens)+"\n")

        setBoxColors(bp, "red")

    plt.xticks(range(0, len(ticks) * 5, 5), ticks, rotation='vertical')
    plt.xlim(-2, len(ticks)*5)
    plt.ylim(yMinLImit, yMaxLimit+1)
    print("\n\n")
    # plt.savefig('boxcompare.png')

def outliersInsertion(sampleData, densPer,k):

    mean = np.mean(sampleData)
    std = np.std(sampleData)

    totPoints = len(sampleData)

    limMin = mean - k* std
    limMax = mean + k* std

    indicesInliers = np.where((sampleData >= limMin) & (sampleData <= limMax))[0] 
    numOut = totPoints - len(indicesInliers)
    dataDens = (numOut/totPoints) * 100

    print("Numero total de pontos: "+ str(totPoints))
    print("Numero total de outliers: "+ str(numOut))
    print("Densidade: " + str(dataDens))

    if dataDens < densPer:
        perChosen = (densPer - dataDens)/100

        numChosenPoints = round(perChosen * len(indicesInliers))
        print("Numero de inliers escolhidos aleatoriamente: " + str(numChosenPoints))

        indicesChosenPoints = np.random.choice(indicesInliers,numChosenPoints,replace = False)
        
        for ind in indicesChosenPoints:

            s = np.random.choice([-1,1],1,replace = False)[0]

            maxVal = np.max(np.abs(sampleData))
            z1 = maxVal - limMin
            z2 = maxVal - limMax
            if z1 > z2:
                z = z1
            else:
                z = z2

            q = np.random.uniform(low = 0.0, high = z)
            sampleData[ind] = mean + s * k*(std + q)

        newNumInliers = np.where((sampleData >= limMin) & (sampleData <= limMax))[0]
        newNumOut = totPoints - len(newNumInliers)
        newDataDens = (newNumOut/totPoints) * 100

        print("NOVA DENSIDADE DA AMOSTRA: " + str(newDataDens))
    
    return sampleData


def linearModule(xData,yData,p):
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
    acc_mod = getVectorModule(
        getActivityData(
            data,
            activity_index,
            sensor_id
        ),
        1
    )
    gyro_mod = getVectorModule(
        getActivityData(
            data,
            activity_index,
            sensor_id
        ),
        4
    )
    mag_mod = getVectorModule(
        getActivityData(
            data,
            activity_index,
            sensor_id
        ),
        7
    )

    plotScatter(axs[0], zscore(acc_mod, k), acc_mod, "ACC")
    plotScatter(axs[1], zscore(gyro_mod, k), gyro_mod, "GYRO")
    plotScatter(axs[2], zscore(mag_mod, k), mag_mod, "MAG")


def initClusters(arr, n):
    """
    Initializa 'n' clusters a partir de
    pontos aleatorios do dataset.
    """
    return arr[np.random.randint(arr.shape[0], size=n)]


def calcDist(centroid, arr):
    """
    Returns euclidean distance.
    """
    return np.linalg.norm(arr - centroid, axis=1)


def kmeans1(arr, n, iters):
    # inicializar 'n' clusters com centroides escolhidos ao acaso
    # centroide estao smp no inicio da fila
    clusters = initClusters(arr, n)
    for j in range(iters):
        for i in range(len(clusters)):
            cluster, centroid = clusters[i], clusters[i][0]
            print("BEFORE:")
            print("Cluster: ", cluster)
            print("Centroid: ", centroid)
            print("-----------------------------------")

            # assign point to its closest centroid
            dist = calcDist(arr, centroid)
            new_point = arr[dist == np.argmin(dist, axis=0)]
            cluster = np.append(cluster, new_point, axis=0)
            print(
                f"Adding: {np.array(new_point)} with a distance of {dist[dist == np.nanmin(dist)]} to nearest centroid")
            print("Updated cluster: ", cluster)

            # compute mean and update centroid
            new_centroid = np.mean(cluster, axis=0)
            print("Updated centroid: ", new_centroid)
            if np.all(new_centroid) != np.all(centroid):
                cluster[0] = new_centroid
                clusters[i] = cluster
                print("Updated cluster: ", cluster)
            else:
                break

            # debug
            """
            print("Centroid: ", centroid)
            print("Adding: ", np.array([new_point]))
            print("Updated cluster: ", cluster)
            """

    print("after\n", clusters)


def kmeans2(arr, n, iters):
    centroids = initClusters(arr, n)

    # para n=3, cada linha = [d(c1->p1) d(c2->p1) d(c3->p1)]
    distances = np.zeros([arr.shape[0], n])

    for i in range(iters):
        # distancia de cada centroide a cada ponto
        for i, c in enumerate(centroids):
            distances[:, i] = calcDist(c, arr)

        # atualiza cada centroide
        # com a media dos ptos que lhe sao mais proximos
        groups = np.argmin(distances, axis=1)
        for j in range(n):
            centroids[j] = np.mean(arr[j == groups], 0)
    return centroids, groups


def plotKmeans(arr, centroids, groups):
    _ = plt.figure()
    ax = plt.axes(projection='3d')

    group_colors = np.random.rand(len(centroids), 3)
    colors = [group_colors[j] for j in groups]

    # plot clusters
    ax.scatter3D(arr[:, 0], arr[:, 1], arr[:, 2], color=colors, alpha=0.5)

    # plot centroids
    ax.scatter3D(centroids[:, 0], centroids[:, 1],
                 centroids[:, 2], color='black', marker='x', lw=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
