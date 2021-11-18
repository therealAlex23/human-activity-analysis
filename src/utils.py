from numpy.lib.type_check import real
from mpl_toolkits import mplot3d
import math
import matplotlib
from matplotlib import legend
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, fft, signal
from plotly.express import scatter_3d, scatter
from sklearn.decomposition import PCA


# -- globals
sFreq, windowDuration = 51.2, 2


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


def outliersInsertion(sampleData, densPer, k):
    """
    Injects outliers till the density = densPer

    An outlier is considered to be the points outside the interval [mean+k*std,mean -k*std]

    """
    mean = np.mean(sampleData)
    std = np.std(sampleData)

    totPoints = len(sampleData)

    # Set limits for outlier filtering
    limMin = mean - k * std
    limMax = mean + k * std

    # Calculate indices of the inliers to be able to select and transform into outliers
    indicesInliers = np.where((sampleData >= limMin)
                              & (sampleData <= limMax))[0]

    numOut = totPoints - len(indicesInliers)
    dataDens = (numOut/totPoints) * 100

    # if the data density of the data is already bigger than that we want
    # Then it isnt necessary to inject more outliers
    if dataDens < densPer:
        perChosen = (densPer - dataDens)/100

        # Random choice of the indices of the values to transform into outliers
        numChosenPoints = round(perChosen * len(indicesInliers))
        indicesChosenPoints = np.random.choice(
            indicesInliers, numChosenPoints, replace=False)

        z = max(np.abs(mean + k * std), np.abs(mean - k * std))

        for ind in indicesChosenPoints:

            s = np.random.choice([-1, 1], 1, replace=False)[0]
            q = np.random.uniform(low=0.0, high=z)
            sampleData[ind] = mean + s * k*(std + q)

        newNumInliers = np.where(
            (sampleData >= limMin) & (sampleData <= limMax))[0]
        newNumOut = totPoints - len(newNumInliers)
        newDataDens = (newNumOut/totPoints) * 100

    return indicesChosenPoints, sampleData


def trainLinearModelPrevNext(mainData, nSamp, p):
    janela = math.floor(p/2)

    # if p = 7 then window = 3
    # Then the previous samples window will be window + 1 = 4 and the following ones with window = 3
    # It has to be the previous ones that are favored because it is worth more to have a past value than a future one

    if p % 2 == 0:
        janelaPrev = janela

    else:
        janelaPrev = janela + 1

    # Matriz n x p+1 --> +1 row of 1s
    matrixX = np.zeros((nSamp, p+1))

    # p=8 ; janela = p/2 = 4;
    # matrixX in column 4 is only values = 1 and contains the following values from position 0 to 3
    matrixX[:, janela] = 1

    # To have correct Y matrix you have to start from the element where you have windowPrev previous values
    matrixY = mainData[janelaPrev:janelaPrev + nSamp]

    # matrix in each row contains the previous values of the value you want to predict
    # Ex: MatrizX [Yi+3 Yi+2 Yi+1 1 Yi-1 Yi-2 Yi-3]

    for i in range(nSamp):

        sampValPrev = mainData[i:janelaPrev+i][::-1]
        sampValNext = mainData[janelaPrev+i+1:janela+janelaPrev+i+1][::-1]

        # Before the column of 1 are the following
        matrixX[i, :janela] = sampValNext
        matrixX[i, janela+1:] = sampValPrev  # Then it's the former values

    # B = PseudoInv(X)*Y to calculate the Weight vector
    pseudoInvX = np.linalg.pinv(matrixX)
    slopeVec = np.dot(pseudoInvX, matrixY)
    return slopeVec


def trainLinearModelPrev(mainData, n, p):

    matrixX = np.zeros((n, p+1))
    matrixX[:, 0] = 1
    matrixY = mainData[p:p+n]

    # matrix in each row contains the previous values of the value you want to predict
    # Ex: MatrizX [1 Yi-1 Yi-2 ... Yi-p]
    for i in range(n):
        sampVal = mainData[i:p+i]
        matrixX[i, 1:] = sampVal[::-1]

    # B = PseudoInv(X)*Y to calculate the Weight vector
    pseudoInvX = np.linalg.pinv(matrixX)
    slopeVec = np.dot(pseudoInvX, matrixY)
    return slopeVec


def testLinearModelPrevNextVal(mainData, indOut, p):

    # Para os plots
    errorVec = []
    realValues = []
    predValues = []

    # Retirar os valores com os indices de outliers para o treino do modelo
    dataWithoutOut = np.delete(np.copy(mainData), indOut)

    slopeVec = trainLinearModelPrevNext(
        dataWithoutOut, len(dataWithoutOut)-p, p)

    janela = math.floor(p/2)

    if p % 2 == 0:
        janelaPrev = janela

    else:
        janelaPrev = janela + 1

    for outlier in indOut:
        # se o indice for 5 e a janela for 6
        # E se o indice for 1000 e a janela ultrapassar o tamanho dos dados

        if outlier >= janelaPrev and outlier < len(mainData)-janela:

            # Buscar os janelaPrev valoes anteriores e janela valores seguintes
            # Depois poe se no formato [Yi+3 Yi+2 .. 1 Yi-1 ...] para se multiplicar com o vetor de Pesos
            sampleNextPrev = np.ones((p+1))
            sampleNextPrev[:janela] = mainData[outlier +
                                               1: janela + outlier+1][::-1]
            sampleNextPrev[janela+1:] = mainData[outlier -
                                                 janelaPrev: outlier][::-1]

            if len(sampleNextPrev) == p+1:

                predVal = np.dot(sampleNextPrev, slopeVec)
                newError = pow((predVal - mainData[outlier]), 2)

                errorVec.append(newError)
                realValues.append(mainData[outlier])
                predValues.append(pow(newError, 0.5) + mainData[outlier])

    # Devolve somatorio dos erros quadraticos demonstra a eficiencia do modelo de uma certa janela de amostra
    return np.sum(np.array(errorVec)), np.array(realValues), np.array(predValues)


def testLinearModelPrevVal(mainData, indOut, p):

    errorVec = []
    realValues = []
    predValues = []

    # Remove the values with outlier indices for model training
    dataWithoutOut = np.delete(np.copy(mainData), indOut)

    slopeVec = trainLinearModelPrev(dataWithoutOut, len(dataWithoutOut)-p, p)

    for outlier in indOut:
        if outlier >= p:  # if index is 5 and window is 6

            # sample P previous values relative to the outlier with the first column having the value 1
            prevValOut = np.ones((p+1))
            prevValOut[1:] = mainData[outlier - p: outlier][::-1]

            if len(prevValOut) == p+1:

                # Yprev = B0 + B1*Yi-1 .... Bp * Yi-p
                predVal = np.dot(prevValOut, slopeVec)
                newError = pow((predVal - mainData[outlier]), 2)

                errorVec.append(newError)
                realValues.append(mainData[outlier])

                #Yprev = realVal + erro
                predValues.append(pow(newError, 0.5) + mainData[outlier])

    return np.sum(np.array(errorVec)), np.array(realValues), np.array(predValues)


def plotModelsQuadError(arrPs, arrModels, moduleName):
    fig = plt.figure()
    fig.suptitle("Modulo de " + moduleName +
                 "\nSomatorio erro quadrático em função do tamanho da janela")
    plt.subplot(2, 1, 1)
    plt.plot(arrPs, arrModels[0])
    plt.scatter(arrModels[0].index(min(arrModels[0])) + 5,
                min(arrModels[0]), color="red", label="Menor Erro")
    plt.legend()
    plt.title("MODELO p valores aneriores")

    plt.subplot(2, 1, 2)
    plt.plot(arrPs, arrModels[1])
    plt.scatter(arrModels[1].index(min(arrModels[1])) + 5,
                min(arrModels[1]), color="red", label="Menor Erro")
    plt.legend()
    plt.title("MODELO p/2 valores anteriores e p/2 valores seguintes")
    pass


def testPVals(arrPs, mainData, indOutliers):
    arrMeanSquareModelPrev = []
    arrMeanSquareModelPrevNext = []
    for p in arrPs:

        meanSquareModelPrev, arrRealModelPrev, arrPredictedModelPrev = testLinearModelPrevVal(
            mainData, indOutliers, p)
        meanSquareModelPrevNext, arrRealModelPrevNext, arrPredictedModelPrevNext = testLinearModelPrevNextVal(
            mainData, indOutliers, p)

        arrMeanSquareModelPrev.append(meanSquareModelPrev)
        arrMeanSquareModelPrevNext.append(meanSquareModelPrevNext)
        print("Valor do p para janela: " + str(p)+"\n")
        print("MODELO P valores aneriores")
        print("Somatorio Erro quadrático: " + str(meanSquareModelPrev))
        print("\n-----------------------------\n")
        print("MODELO p/2 valores anteriores e p/2 valores seguintes")
        print("Somatorio erro quadrático: " + str(meanSquareModelPrevNext))
        print("\n\n")
    # Tuples with array of predicted values and array of actual values by the p-valued prior linear model
    valModelPrev = (arrRealModelPrev, arrPredictedModelPrev)

    # Tuplo with array of predicted values and array of actual values by the linear model of p/2 preceding and following values
    valModelPrevNext = (arrRealModelPrevNext, arrPredictedModelPrevNext)

    return arrMeanSquareModelPrev, arrMeanSquareModelPrevNext, valModelPrev, valModelPrevNext


def scatterRealLinePred(real, predicted):
    plt.plot(range(len(predicted)), predicted, "-r")
    plt.scatter(range(len(real)), real, color="blue", label=" Valor Real")
    plt.title("Regression line for prediction model Compared to real values")
    plt.grid()
    plt.show()


def scatterPlotRealPredicted(real, predicted, moduleName, model):
    plt.scatter(range(len(predicted)), predicted,
                color="red", label=" Valor Previsto")
    plt.scatter(range(len(real)), real, color="blue", label=" Valor Real")
    plt.legend()
    plt.title(model + " de Modulo de "+str(moduleName) +
              "\nComparação entre valores previstos e reais")


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
    z = np.abs((arr - np.mean(arr)) / np.std(arr))
    return np.where(z > k)[0]


def plotScatter(ax, x, y, title):
    ax.plot(y, '.', color='blue')
    ax.plot(x, y[x], '.', color='red')
    ax.set_title(title)


# deviceId, acc[x, y, z], gyro[x, y, z], mag[x, y, z], timestamp, activityIndex
def plotOutliers(sensorData, k, axs):
    labels = ["ACC", "GYRO", "MOD"]
    for i, sensorMod in enumerate(sensorData):
        plotScatter(axs[i], zscore(sensorMod, k), sensorMod, labels[i])


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
    """
    # esta mal
    """
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


def kmeans2(arr, n):
    centroids = initClusters(arr, n)

    # para n=3, cada linha = [d(c1->p1) d(c2->p1) d(c3->p1)]
    distances = np.zeros([arr.shape[0], n])
    while True:
        # distancia de cada centroide a cada ponto
        for i, c in enumerate(centroids):
            distances[:, i] = calcDist(c, arr)

        # groups contem o indice do centroide do cluster
        # mais proximos
        groups = np.argmin(distances, axis=1)

        # para comparar
        old_centroids = centroids
        for j in range(n):
            centroids[j] = np.mean(arr[j == groups], 0)
        if (old_centroids == centroids).all():
            break
    return centroids, groups


def getOutliers(arr, k):
    return [
        [
            arr[zscore(arr[:, i], k)][:, j] for j in range(3)
        ] for i in range(3)
    ]


def plotKmeans(ax, arr, outliers, centroids, groups):
    group_colors = np.random.rand(len(centroids), 3)
    colors = [group_colors[j] for j in groups]

    # plot clusters
    ax.scatter3D(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        color=colors,
        alpha=0.5
    )

    # plot acc outliers
    ax.scatter3D(
        outliers[0][0],
        outliers[0][1],
        outliers[0][2],
        color='black',
        alpha=1
    )

    # plot gyro outliers
    ax.scatter3D(
        outliers[1][0],
        outliers[1][1],
        outliers[1][2],
        color='black',
        alpha=1
    )

    # plot mag outliers
    ax.scatter3D(
        outliers[2][0],
        outliers[2][1],
        outliers[2][2],
        color='black',
        alpha=1
    )

    # plot centroids
    ax.scatter3D(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        color='black',
        marker='X',
        lw=5
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def normalizeCurve(arr):
    """
    Adjust data to fit common scale
    """
    return (arr - np.mean(arr)) / np.std(arr)


def sturge(n):
    """
    Get correct number of histogram bins
    according to Sturge's Rule.
    'n' is the length of the data.
    """
    return int(round(1 + 3.322 * np.log(n)))


def ksTest(arr):
    """
    Perform Kolmogorov-Smirnov test
    on 1D arrays.
    Utilized to compare the distribution of
    our data with the normal distribution.
    Note: stats.kstest returns 2 values: statistic and p-value
    """
    return stats.kstest(arr, 'norm')


def makeGraphTitle(part, act, sensor, devId):
    return f"[Part{part}] '{act}' - <{sensor}> {devId}"


def plotDistribution(arr, ax, title):
    ax.hist(arr, sturge(len(arr)))
    ax.set_title(title)


def plotCdfFit(ax, arr, x, title):
    """
    Plot CDF (i.e normal(0, 1))
    against ECDF (our data)
    """
    ax.step(
        arr,
        np.arange(len(arr)) / len(arr),
        where='post',
        label='Empirical CDF'
    )
    ax.plot(
        x,
        stats.norm(0, 1).cdf(x),
        label='CDF for N(0, 1)'
    )
    plt.legend()
    ax.set_title(title)
    plt.xlim([x[0], x[len(x) - 1]])


def getWindows(windows, winsize, inc, data):
    """
    Split data stream into fixed-size windows
    Returns a dictionary with
    (activity, array of intervals where a given
    activity is performed) pairs
    """
    for i in range(0, len(data) - winsize, inc):
        if data[i, -1] == data[i + winsize, -1]:
            if len(windows[data[i, -1]]) == 0:
                windows[data[i, -1]] = [data[i:i+winsize, :]]
            else:
                windows[data[i, -1]] = np.append(
                    windows[data[i, -1]],
                    [data[i:i+winsize, :]],
                    axis=0
                )
    return windows


def zcr(w):
    """
    Computes Zero-Crossing rate.
    """
    return np.nonzero(np.diff(w > 0))[0].shape[0] / len(w)


def dft(w):
    """
    Returns Discrete Fourier Transform and frequencies.
    """
    return fft.rfftfreq(len(w)), fft.rfft(w)
    # return signal.welch(w, sFreq, nperseg=len(w))


def energy(w):
    """
    Returns energy of signal.
    Taken from https://stackoverflow.com/questions/29429733/cant-find-the-right-energy-using-scipy-signal-welch
    """
    # -- welch
    # _, pxx = dft(w)
    # return sum(pxx ** 2 / len(w))
    _, fft = dft(w)
    return np.sum(np.abs(fft) ** 2) / len(w)


def df(w):
    """
    Returns energy of dominant freq of the signal.
    """
    freqs, fft = dft(w)
    return freqs[np.argmax(fft ** 2)]


def cagh(xyz):
    """
    Compute Correlation between Acceleration
    along Gravity and Heading Directions.
    Gravity direction is parallel
    to the 'x' axis, and heading direction
    is a combo of the 'y' and 'z' axes.

    Receives 3 columns as input: x, y, and z.
    Outputs correlation coefficient.
    """
    return np.corrcoef(
        np.sqrt(xyz[:, 1] ** 2 + xyz[:, 2] ** 2), xyz[:, 0]
    )[0, 1]


def avgd(xyz, col='x', windur=windowDuration, t=1/sFreq):
    """
    Computes Averaged Velocity along
    given Direction.
    Default is 'x'-axis' direction.
    """
    if col == 'x':
        return np.trapz(xyz[:, 0], np.arange(0, windur - t, t)) / windur
    elif col == 'y':
        return np.trapz(xyz[:, 1], np.arange(0, windur - t, t)) / windur
    elif col == 'z':
        return np.trapz(xyz[:, 2], np.arange(0, windur - t, t)) / windur


def avhd(xyz, windur=windowDuration, t=1/sFreq):
    """
    Computes Averaged Velocity along
    Heading Direction (y + z)
    """
    return np.sqrt(
        avgd(xyz, 'y', windur, t) ** 2 +
        avgd(xyz, 'z', windur, t) ** 2
    )


def ai(xyz):
    """
    Computes Average Movement Intensity.
    """
    mi = np.linalg.norm(xyz, axis=1)
    return np.sum(mi) / len(xyz)


def sma(xyz):
    """
    Computes Normalized Signal Magnitude
    Area.
    Receives 3 columns (x, y, z) as input
    and outputs a single column.
    """
    return np.sum(np.abs(xyz) / len(xyz))


def ae(xyz):
    """
    Pass acc (to get 'aae') or
    gyro (to get 'are') vector to
    this function!
    """
    return np.sum(energy(xyz)) / 3


def eva(xyz):
    """
    Returns the top two eigenvalues,
    which correspond to heading direction
    and the vertical direction.
    """
    covm = np.cov(xyz.T, bias=True)
    return round(covm[0, 0], 6), round(covm[2, 2], 6)


def aratg(xyz):
    """
    Computes the Averaged Rotation Angles related
    to Gravity Direction.
    Captures the rotation movement of
    the human torso around the x-axis.
    Pass gyro data to this function!
    """
    pass


def getFeature(f, w, si, fi, method='single'):
    """
    Returns an array of length 3 (x, y, z),
    with values of 'w' computed by 'f'.
    method = 'single' computes value for a
    single axis.
    If method = 'all', computes value for
    x, y and z.
    """
    if method == 'all':
        out = f(w[:, si:fi+1])
        if isinstance(out, tuple):
            return out
        return round(out, 6)
    return [round(f(w[:, i]), 6) for i in range(si, fi+1)]


def getFeatureName(f):
    """
    Extract feature name as string
    from array of feature-extracting functions
    """
    if isinstance(f, str):
        return f
    return str(f.__name__).rsplit('.', 1)[-1]


def getFeatures(funcs, win, si, fi, method='single'):
    """
    Given a window array 'win',
    a list of functions to compute values 'funcs',
    and starting (index of x coordinate) and finish indexes
    (index of y coordinate), inclusive - respectively 'si', 'fi' -,
    this function returns a dictionary with:
    key = function name;
    value = [function(x), function(y), function(z)]
    """
    if method == 'all':
        return {getFeatureName(f): getFeature(f, win, si, fi, method='all') for f in funcs}
    return {getFeatureName(f): getFeature(f, win, si, fi) for f in funcs}


def flatten(t):
    """
    Receives list of lists
    and returns a list.
    """
    out = []
    for sublist in t:
        for item in sublist:
            out.append(item)
    return out


def fetchFeatures(w, features, method):
    """
    Wrapper function for
    retrieving statistical feature values
    """
    return (
        getFeatures(features, w, 1, 3, method=method),
        getFeatures(features, w, 4, 6, method=method),
        getFeatures(features, w, 7, 9, method=method)
    )


def getColumns(labels, stats, phys):
    """
    Returns a 1d array of feature labels
    in the dataset.
    """
    return [
        f'{sensor.lower()}_{ax}_{getFeatureName(stat)}'
        for sensor in labels
        for stat in stats
        for ax in ['x', 'y', 'z']
    ] + [
        f'{sensor.lower()}_{getFeatureName(ft)}'
        for sensor in labels
        for ft in [ai, sma]
    ] + [
        f'{getFeatureName(pf)}'
        for pf in phys + ['evah', 'evag', 'aae', 'are']
    ] + ['act']


def getWindowData(w, stats, phys):
    """
    Given a window 'w', returns info
    for all sensors, in every axis.
    """
    # statistical features ({key:[x, y, z]})
    accStats, gyroStats, magStats = fetchFeatures(w, stats, 'single')

    # physical features ({key:[x, y, z]})
    accPhys, gyroPhys, magPhys = fetchFeatures(w, [ai, sma], 'all')

    # general physical features ({key:value})
    physFeatures = getFeatures(phys, w, 1, 3, method='all')
    physFeatures['evah'], physFeatures['evag'] = eva(w[:, 1:4])
    physFeatures['aae'] = round(ae(w[:, 1:4]), 6)
    physFeatures['are'] = round(ae(w[:, 4:7]), 6)

    # all info for a given window
    return flatten(list(accStats.values())) + list(accPhys.values()) + flatten(list(gyroStats.values())) + list(
        gyroPhys.values()) + flatten(list(magStats.values())) + list(magPhys.values()) + list(physFeatures.values())


def normalize(a):
    return (a - a.mean()) / a.std()


def hexCode():
    """
    Generates random hex code.
    """
    return f"#{format(randint(0, 16777215), 'x')}"


def comparisonPlot(data, ftname, colors, sensor=None, ftname2=None, mode='3d'):
    """
    Replicates the graphs on
    the cited paper and some more.
    """
    if mode == '2d' and ftname2:
        fig = scatter(
            data,
            ftname,
            ftname2,
            color_discrete_sequence=colors,
            color='act',
            title=f'{ftname}-{ftname2} comparison'
        )
    else:
        fig = scatter_3d(
            data,
            f'{sensor}_x_{ftname}',
            f'{sensor}_y_{ftname}',
            f'{sensor}_z_{ftname}',
            color_discrete_sequence=colors,
            color='act',
            title=f'{sensor} {ftname} comparison'
        )
    fig.write_html('fig.html', auto_open=True)


def doPca(dataset, n_components):
    pca = PCA(n_components=n_components)
    pca_dataset = pca.fit_transform(dataset)
    return pca, pca_dataset


def getEvrs(dataset):
    """
    Returns EVRs for a different
    number of PC's
    """
    evratios = []
    for i in range(len(dataset.columns)):
        pca, _ = doPca(dataset, i)
        evratios.append(sum(pca.explained_variance_ratio_) * 100)
    return evratios


def findPCs(evratios, accuracy):
    """
    Finds index of number
    of pc's to include in our pca
    by intersecting the evr % 
    and the evr's of pca.
    """
    return np.argwhere(np.diff(np.sign(
        evratios - np.repeat(accuracy, len(evratios))
    ))).flatten()[0]


def plotEvrPc(evratios, pc):
    _, ax = plt.subplots()
    x = np.arange(len(evratios))
    ax.plot(x, evratios)
    ax.plot(x[pc], evratios[pc], 'x', markersize=12,
            label=f'# of PCs: {x[pc]}')
    ax.axhline(evratios[pc], color='r', ls='--',
               label=f'{round(evratios[pc], 2)} % EVR')
    ax.legend(loc='lower right')
    ax.set(
        xlabel='number of PCs',
        ylabel='explained variance ratio (%)',
        title='# of PCs vs EVR'
    )
    ax.grid()
    plt.show()
