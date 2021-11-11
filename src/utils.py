from matplotlib import legend
import matplotlib
import numpy as np
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numpy.lib.type_check import real


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

    #Cacular limites para filtração de outliers
    #Outlier considerado se estiver fora do intervalo [mean - k* std,mean + k* std]
    limMin = mean - k* std
    limMax = mean + k* std

    #Calcular indices dos inliers para poder selecionar e transformar em outliers
    indicesInliers = np.where((sampleData >= limMin) & (sampleData <= limMax))[0] 

    #Density Percentage --> (totalOutliers/totalPoints) * 100
    numOut = totPoints - len(indicesInliers)
    dataDens = (numOut/totPoints) * 100


    if dataDens < densPer:
        perChosen = (densPer - dataDens)/100

        #Escolha random dos indices dos valores a transformar em outliers
        numChosenPoints = round(perChosen * len(indicesInliers))
        indicesChosenPoints = np.random.choice(indicesInliers,numChosenPoints,replace = False)

        z = max(np.abs(mean + k * std), np.abs(mean - k * std))

        for ind in indicesChosenPoints:

            s = np.random.choice([-1,1],1,replace = False)[0]
            q = np.random.uniform(low = 0.0, high = z)
            sampleData[ind] = mean + s * k*(std + q)

        newNumInliers = np.where((sampleData >= limMin) & (sampleData <= limMax))[0]
        newNumOut = totPoints - len(newNumInliers)
        newDataDens = (newNumOut/totPoints) * 100
    
    return indicesChosenPoints,sampleData

def trainLinearModelPrevNext(mainData,nSamp,p):
    janela = math.floor(p/2)

    #se for p = 7 então fica janela = 3
    #Logo a janela de amostras anteriores vai ser janela + 1 = 4 e os seguintes com a janela = 3
    #Tem que ser os anteriores os favorecidos pois vale mais a pena ter um valor passado do que um futuro

    if p % 2 == 0:
        janelaPrev = janela
        
    else:
        janelaPrev = janela + 1
    
    #Matriz n x p+1 --> +1 coluna de 1s
    matrixX = np.zeros((nSamp,p+1))

    #p=8 ; janela = p/2 = 4; matrixX na coluna 4 é so 1 e contem os valores seguintes da posiçao 0 a 3 
    matrixX[:,janela] = 1

    #Para ter matriz Y correta tem começar a partir do elemento em que tem janelaPrev valores anteriores
    matrixY = mainData[janelaPrev:janelaPrev +nSamp]

    #matriz em cada linha contem os valores anteriores do valor que se quer prever
    #Ex: MatrizX [Yi+3 Yi+2 Yi+1 1 Yi-1 Yi-2 Yi-3]

    for i in range(nSamp):

        sampValPrev = mainData[i:janelaPrev+i][::-1]
        sampValNext = mainData[janelaPrev+i+1:janela+janelaPrev+i+1][::-1]

        matrixX[i,:janela] = sampValNext #Antes da coluna do 1 são os seguintes
        matrixX[i,janela+1:] = sampValPrev #Depois é que são os anteriores
    
    # B = PseudoInv(X)*Y para se calcular o vetor de Pesos
    pseudoInvX = np.linalg.pinv(matrixX)
    slopeVec = np.dot(pseudoInvX,matrixY)
    return slopeVec
    

def trainLinearModelPrev(mainData,n,p):
    
    matrixX = np.zeros((n,p+1))
    matrixX[:,0] = 1
    matrixY = mainData[p:p+n]

    #matriz em cada linha contem os valores anteriores do valor que se quer prever
    #Ex: MatrizX [1 Yi-1 Yi-2 ... Yi-p]
    for i in range(n):
        sampVal = mainData[i:p+i]
        matrixX[i,1:] = sampVal[::-1]
    
    # B = PseudoInv(X)*Y para se calcular o vetor de Pesos
    pseudoInvX = np.linalg.pinv(matrixX)
    slopeVec = np.dot(pseudoInvX,matrixY)
    return slopeVec

def testLinearModelPrevNextVal(mainData,indOut,p):

    #Para os plots
    errorVec = []
    realValues = []
    predValues = []

    #Retirar os valores com os indices de outliers para o treino do modelo
    dataWithoutOut = np.delete(np.copy(mainData),indOut)

    slopeVec = trainLinearModelPrevNext(dataWithoutOut,len(dataWithoutOut)-p,p)

    janela = math.floor(p/2)

    if p % 2 == 0:
        janelaPrev = janela
        
    else:
        janelaPrev = janela + 1
    
    for outlier in indOut:
        #se o indice for 5 e a janela for 6
        #E se o indice for 1000 e a janela ultrapassar o tamanho dos dados

        if outlier >= janelaPrev and outlier < len(mainData)-janela: 

            #Buscar os janelaPrev valoes anteriores e janela valores seguintes
            #Depois poe se no formato [Yi+3 Yi+2 .. 1 Yi-1 ...] para se multiplicar com o vetor de Pesos
            sampleNextPrev = np.ones((p+1))
            sampleNextPrev[:janela] = mainData[outlier+1: janela + outlier+1][::-1]
            sampleNextPrev[janela+1:] = mainData[outlier-janelaPrev: outlier][::-1]
            
            if len(sampleNextPrev) == p+1:
                
                predVal  = np.dot(sampleNextPrev,slopeVec)
                newError = pow((predVal - mainData[outlier]), 2)

                errorVec.append(newError)
                realValues.append(mainData[outlier])
                predValues.append(pow(newError,0.5) + mainData[outlier])

    #Devolve somatorio dos erros quadraticos demonstra a eficiencia do modelo de uma certa janela de amostra
    return np.sum(np.array(errorVec)) ,np.array(realValues),np.array(predValues)

def testLinearModelPrevVal(mainData,indOut,p):

    errorVec = []
    realValues = []
    predValues = []

    #Retirar os valores com os indices de outliers para o treino do modelo
    dataWithoutOut = np.delete(np.copy(mainData),indOut)
    
    slopeVec = trainLinearModelPrev(dataWithoutOut,len(dataWithoutOut)-p,p)

    for outlier in indOut:
        if outlier >= p: #se o indice for 5 e a janela for 6
            
            #amostra P valores anteriores em relacao ao outlier com a primeira coluna com o valor 1
            prevValOut = np.ones((p+1))
            prevValOut[1:] = mainData[outlier - p: outlier][::-1]
            
            if len(prevValOut) == p+1:

                #Yprev = B0 + B1*Yi-1 .... Bp * Yi-p   
                predVal = np.dot(prevValOut,slopeVec)
                newError = pow((predVal - mainData[outlier]), 2)

                errorVec.append(newError)
                realValues.append(mainData[outlier])

                #Yprev = realVal + erro
                predValues.append(pow(newError,0.5) + mainData[outlier])
    
    return np.sum(np.array(errorVec)) ,np.array(realValues),np.array(predValues)

def plotModelsQuadError(arrPs, arrModels, moduleName):
    fig = plt.figure()
    fig.suptitle("Modulo de "+ moduleName + "\nSomatorio erro quadrático em função do tamanho da janela")
    plt.subplot(2,1,1)
    plt.plot(arrPs ,arrModels[0])
    plt.scatter(arrModels[0].index(min(arrModels[0])) + 5, min(arrModels[0]), color="red", label="Menor Erro")
    plt.legend()
    plt.title("MODELO p valores aneriores")

    plt.subplot(2,1,2)
    plt.plot(arrPs ,arrModels[1])
    plt.scatter(arrModels[1].index(min(arrModels[1])) + 5, min(arrModels[1]), color="red", label="Menor Erro")
    plt.legend()
    plt.title("MODELO p/2 valores anteriores e p/2 valores seguintes")
    pass

def testPVals(arrPs,mainData,indOutliers):
    arrMeanSquareModelPrev = []
    arrMeanSquareModelPrevNext = []
    for p in arrPs:

        meanSquareModelPrev, arrRealModelPrev, arrPredictedModelPrev = testLinearModelPrevVal(mainData,indOutliers,p)
        meanSquareModelPrevNext, arrRealModelPrevNext, arrPredictedModelPrevNext = testLinearModelPrevNextVal(mainData,indOutliers,p)

        arrMeanSquareModelPrev.append(meanSquareModelPrev)
        arrMeanSquareModelPrevNext.append(meanSquareModelPrevNext)
        print("Valor do p para janela: " +  str(p)+"\n")
        print("MODELO P valores aneriores")
        print("Somatorio Erro quadrático: " + str(meanSquareModelPrev))
        print("\n-----------------------------\n")
        print("MODELO p/2 valores anteriores e p/2 valores seguintes")
        print("Somatorio erro quadrático: " + str(meanSquareModelPrevNext))
        print("\n\n")
    #Tuplo com array de valores previstos e array de valores reais pelo modelo linear de p valores anteriores
    valModelPrev = (arrRealModelPrev,arrPredictedModelPrev)

    #Tuplo com array de valores previstos e array de valores reais pelo modelo linear de p/2 valores anteriores e seguintes
    valModelPrevNext = (arrRealModelPrevNext,arrPredictedModelPrevNext)

    return arrMeanSquareModelPrev,arrMeanSquareModelPrevNext,valModelPrev,valModelPrevNext

def scatterRealLinePred(real, predicted):
    plt.plot(range(len(predicted)),predicted,"-r")
    plt.scatter(range(len(real)),real, color="blue",label=" Valor Real")
    plt.title("Regression line for prediction model Compared to real values")
    plt.grid()
    plt.show()


def scatterPlotRealPredicted(real, predicted,moduleName,model):
    plt.scatter(range(len(predicted)),predicted, color="red",label=" Valor Previsto")
    plt.scatter(range(len(real)),real, color="blue",label=" Valor Real")
    plt.legend()
    plt.title(model + " de Modulo de "+str(moduleName) +  "\nComparação entre valores previstos e reais")


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
