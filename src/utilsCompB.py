from os import name
from joblib.logger import PrintTime
from matplotlib.pyplot import title
from dependecies import *
from globals import distMetrics,randomState,figSizeGen

def TTDataSplit(dataX,dataY, trainRatio):
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size = 1 - trainRatio, random_state = randomState)

    return (X_train,y_train), (X_test,y_test)

def TVTDataSplit(dataX,dataY,trainRatio,valRatio,testRatio):

    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size = 1- trainRatio, random_state = randomState)
    stepRatio = testRatio/(testRatio+valRatio)

    X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size=stepRatio,shuffle=True)

    return (X_train,y_train),(X_val,y_val) ,(X_test,y_test)


def kFoldDataSplit(data, nSplits):
    dataSize = data.shape
    dataInd = np.arange(dataSize[0])
    splitsSets = []

    kf = KFold(n_splits=nSplits,shuffle=True)

    for train_index, test_index in kf.split(data):
        
        if len(dataSize) > 1:
            X_train, X_test = data[train_index,:], data[test_index,:]
        else:
            X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = dataInd[train_index],dataInd[test_index]
        
        trainSet, testSet = (X_train,y_train),(X_test,y_test)

        splitsSets.append((trainSet,testSet))

    return splitsSets

def modelMetrics(model,testX,testY,avgMetrics = "weighted"):

    pred = model.predict(testX)
    metricsKnn = getEvalMetrics(testY,pred,avgMetrics)
    return metricsKnn,pred


def getEvalMetrics(real,predicted,avg):

    precision = precision_score(real,predicted,average=avg)
    recall = recall_score(real,predicted, average=avg)
    f1Score = f1_score(real,predicted,average= avg)
    matrixConf = confusion_matrix(real,predicted)
    numDec = 4
    
    return (matrixConf,np.around(precision,decimals=numDec),np.around(recall,decimals=numDec),np.around(f1Score,decimals=numDec))
    
def showMetrics(title,classesLab,metrics,dirPlot):
    fig,axs = plt.subplots(2)

    fig.suptitle(title+"\nModel Test Results")
    colLabs = ["Precision","Recall","F1-Score"]
    dataVals = [list(metrics[1:])]

    axs[0].table(cellText=dataVals,colLabels=colLabs,loc="center")
    axs[0].axis('off')
    axs[0].axis('tight')

    axs[1].set_title("Matrix Confusion")
    confMatrix =  pd.DataFrame(data = metrics[0],columns=classesLab,index=classesLab)
    sb.heatmap(confMatrix , annot = True ,cmap='viridis', ax=axs[1])
    axs[1].axis('tight')
    saveFig(fig,dirPlot,"/"+title+"ModelTestReults.png")
    plt.show()


def KnnTrainParamAnalysis(dataX,dataY,trainRatio,folds,classes):

    kNeigh = 1

    #Train-Only
    plotTitle = "Parameter Testing using Train-Only"
    KnnParamAnalysis("Knn (k=",kNeigh,dataX,dataY,dataX,dataY,distMetrics,classes)

    #TT 70-30
    trainSet,testSet = TTDataSplit(dataX,np.arange(len(dataX)),trainRatio)
    trainY,testY = dataY[trainSet[1]], dataY[testSet[1]]

    plotTitle = "Parameter Testing using TT " +str(trainRatio) + "-" +str(round(1-trainRatio,2))
    KnnParamAnalysis(plotTitle,kNeigh, trainSet[0],trainY,testSet[0],testY,distMetrics,classes)

    #K-Fold 10x10(10CV)
    splits = kFoldDataSplit(dataX,folds)

    numFold = 1
    for trainSet,testSet in splits:
        trainY,testY = dataY[trainSet[1]], dataY[testSet[1]]
        plotTitle = "Parameter Testing using K-Fold\n\nFold "+str(numFold)
        KnnParamAnalysis(plotTitle,kNeigh, trainSet[0],trainY,testSet[0],testY,distMetrics,classes)
        numFold += 1

def KnnParamAnalysis(plotTitle,kNeigh,trainX,trainY,testX,testY,distMetrics,classes):

    distScores = []
    confMatrixArr = []
    colLabs = ["Precision","Recall","F1-Score"]

    for dist in distMetrics:
        neighbor = KNeighborsClassifier(n_neighbors=kNeigh,metric = dist)
        neighbor.fit(trainX,trainY)
        metrics,pred = modelMetrics(neighbor,testX,testY)
        distScores.append(list(metrics[1:]))
        confMatrixArr.append(metrics[0])

    fig = plt.figure(figsize = figSizeGen)
    fig.suptitle(plotTitle)    
    G = gridspec.GridSpec(2,4)
    axsTable = plt.subplot(G[0,:])
    axsTable.axis('off')
    axsTable.axis('tight')
    table = axsTable.table(cellText=distScores,colLabels=colLabs,rowLabels=distMetrics,loc="center")
    table.set_fontsize(15)
    col = 0
    labFlag = 0
    for confMat in confMatrixArr:
        axs = plt.subplot(G[1,col])
        axs.set_title(distMetrics[col])
        try:
            confMat =  pd.DataFrame(data = confMat,columns=classes,index=classes)
            if labFlag == 0:
                hm = sb.heatmap(data =confMat , annot = True ,cmap='viridis', ax=axs)
                labFlag = 1
            else:
                sb.heatmap(data=confMat, annot = True,cmap='viridis', ax=axs,yticklabels=False,xticklabels=False)
        except:
            table = axs.table(cellText = confMat,colLabels = range(len(confMat)),rowLabels = range(len(confMat)))
            axsTable.axis('off')
            axsTable.axis('tight')
            table.set_fontsize(14)
        col+= 1
    
    plt.show()


def KFoldAnalysis(model,modelName,labClasses,numSplits,dataX,dataY):
    
    splits = kFoldDataSplit(dataX,numSplits)

    numFold = 1

    for trainSet,testSet in splits:
        model.fit(trainSet[0],dataY[trainSet[1]])

        metricsDum,predVal = modelMetrics(model,testSet[0],dataY[testSet[1]])

        titlePlot = modelName + " using K-Fold("+str(numSplits)+"CV)\n"+"Fold "+str(numFold)
        showMetrics(titlePlot,labClasses,metricsDum)
        numFold+=1
    

def KnnKFoldAnalysis(folds,dataX,dataY,dist,kVals,classes,dirPlots):
    splits = kFoldDataSplit(dataX,folds)

    numFold = 1
    indexFolds = ["Fold " + str(i) for i in range(1,folds +1)]
    f1ScoreFolds = []
    for trainSet,testSet in splits:
        dataKfold = []
        matConfArr = []
        f1ScoreFold = []
        for k in kVals:
            neighbor = KNeighborsClassifier(n_neighbors=k,metric = dist)
            neighbor.fit(trainSet[0],dataY[trainSet[1]])

            metricsKnn,predVal = modelMetrics(neighbor,testSet[0],dataY[testSet[1]])
            f1ScoreFold.append(metricsKnn[3])
            dataKfold.append(list(metricsKnn[1:]))
            matConfArr.append(metricsKnn[0])

        f1ScoreFolds.append(f1ScoreFold)

        dataKfold = np.array(dataKfold)
        precScores = dataKfold[:,0]
        recScores = dataKfold[:,1]
        f1Scores = dataKfold[:,2]

        indBest = np.where(f1Scores == np.amax(f1Scores))[0][0]
        best = list(dataKfold[indBest])
        best.append(kVals[indBest]) 

        bestMatConf = matConfArr[indBest]
        testDic = {"Precision": precScores, "Recall": recScores, "F1-Score": f1Scores,"Best": (best,bestMatConf)}
        
        plotTitle = "K Neighbour Value Testing using K-Fold ("+str(folds)+"CV)\n Fold "+str(numFold)
        numFold += 1
        plotKnnTestResults(plotTitle,kVals,testDic,classes,dirPlots,"Fold  " + str(numFold))

    f1ScoreFolds = pd.DataFrame(data = f1ScoreFolds, columns =kVals,index = indexFolds)
    fig,axs = plt.subplots(folds)
    fig.suptitle("F1-Score variation per Fold")
    for i in range(folds):
        fold = indexFolds[i]
        axs[i].set_title(fold)
        axs[i].plot(kVals,f1ScoreFolds.iloc[i],color = "blue",marker ='o')
        if i != folds -1:
            axs[i].xaxis.set_visible(False)
    plt.show()

def KnnTVTAnalysis(trainRat,valRat,testRat,dataX,dataY,dist,kVals,classes,dirPlots):

    trainSetTVT, valSetTVT, testSetTVT = TVTDataSplit(dataX,np.arange(len(dataY)),trainRat,valRat,testRat)
    dataTVTVal = []
    models = []
    for k in kVals:
        neighbor = KNeighborsClassifier(n_neighbors=k,metric = dist)
        models.append(neighbor)
        neighbor.fit(trainSetTVT[0],dataY[trainSetTVT[1]])
        
        metricsKnn,predVal = modelMetrics(neighbor, valSetTVT[0],dataY[valSetTVT[1]])
        dataTVTVal.append(list(metricsKnn[1:]))

    dataTVTVal = np.array(dataTVTVal)

    precScores = dataTVTVal[:,0]
    recScores = dataTVTVal[:,1]
    f1Scores = dataTVTVal[:,2]

    indBest = np.where(f1Scores == np.amax(f1Scores))[0][0]
    bestModel = models[indBest]
    metricsKnn,predVal = modelMetrics(bestModel,testSetTVT[0],dataY[testSetTVT[1]])

    best = list(metricsKnn[1:])
    best.append(kVals[indBest])

    testDic = {"Precision": precScores, "Recall": recScores, "F1-Score": f1Scores,"Best": (best,metricsKnn[0])}

    plotKnnTestResults("K value testing using TVT 40-30-30",kVals,testDic,classes,dirPlots,"TVT CV")

def KNNTrainOnlyAnalysis(dataX,dataY,dist,kVals,classes,dirPlots):
    dataTrainOnly = []

    matrixConfArr = []

    for k in kVals:
        neighbor = KNeighborsClassifier(n_neighbors=k,metric = dist)
        neighbor.fit(dataX,dataY)
        metricsKnn,predVal = modelMetrics(neighbor,dataX,dataY)
        dataTrainOnly.append(list(metricsKnn[1:]))
        matrixConfArr.append(metricsKnn[0])
    
    dataTrainOnly = np.array(dataTrainOnly)
    
    precScores = dataTrainOnly[:,0]
    recScores = dataTrainOnly[:,1]
    f1Scores = dataTrainOnly[:,2]

    indBest = np.where(f1Scores == np.amax(f1Scores))[0][0]
    best = list(dataTrainOnly[indBest])
    best.append(kVals[indBest])
    bestMatrixConf = matrixConfArr[indBest]

    testDic = {"Precision": precScores, "Recall": recScores, "F1-Score": f1Scores,"Best": (best,bestMatrixConf)}

    plotKnnTestResults("K Neighbour value testing using Train-Only",kVals,testDic,classes,dirPlots,"Train-Only")

def KnnNumFeatAnalysis(dataX,dataY,kVals,dist,featRanking,dirPlots):
    trainRatio, valRatio, testRatio = 0.4, 0.3, 0.3
    trainSetTVT, valSetTVT, testSetTVT = TVTDataSplit(dataX,np.arange(len(dataX)),trainRatio,valRatio,testRatio)

    trainX,trainY = trainSetTVT[0], dataY[trainSetTVT[1]]
    valX,valY = valSetTVT[0], dataY[valSetTVT[1]]
    testX,testY = testSetTVT[0], dataY[testSetTVT[1]]

    numFeats = len(featRanking)

    labels_nr_features = [str(i) + " features" for i in range(1, numFeats + 1)]
    list_of_multiple_features = [featRanking[:i] for i in range(1, 1 + numFeats)]

    scoresPerFeat = []
    combFeats = []
    allModels = []
    for ind in featRanking:
        combFeats.append(ind)
        scoresK = []
        modelsFeats = []
        for k in kVals:
            neighbour = KNeighborsClassifier(n_neighbors=k,metric = dist)
            neighbour.fit(trainX[:,combFeats],trainY)
            metricsKnn,predVal = modelMetrics(neighbour,valX[:,combFeats],valY)
            scoresK.append(metricsKnn[3])
            modelsFeats.append(neighbour)
        
        scoresPerFeat.append(scoresK)
        allModels.append(modelsFeats)

    results = pd.DataFrame(data = scoresPerFeat, columns = kVals, index = labels_nr_features)
    allModels = pd.DataFrame(data = allModels,columns =kVals,index =labels_nr_features)
    resultsPlot = np.array(results)
    plotPerFig = 5

    numFigs = math.ceil(resultsPlot.shape[0]/plotPerFig)
    if plotPerFig >= numFeats:
        fig,axs = plt.subplots(numFeats,sharex=True, sharey=True,figsize = figSizeGen)
        for i in range(numFeats):
            axs[i].set_title(str(i+1) + " features")
            axs[i].plot(kVals,resultsPlot[i],color = "blue",marker = 'o')
            if i != numFeats-1:
                axs[i].xaxis.set_visible(False)
        fig.savefig(dirPlots + "/Features 1-"+str(numFeats)+".png")
    else:
        axsArr = []
        figArr = []
        left = len(resultsPlot) % plotPerFig

        for i in range(numFigs):
            if left != 0 and i == numFigs-1:
                fig,axs = plt.subplots(left,sharex=True, sharey=True,figsize = figSizeGen)
                figArr.append(fig)
                fig.supxlabel('K Value',fontweight='bold')
                fig.supylabel('F1-Score',fontweight='bold')
            else:
                fig,axs = plt.subplots(plotPerFig,sharex=True, sharey=True,figsize = figSizeGen)
                figArr.append(fig)
                fig.supxlabel('K Value',fontweight='bold')
                fig.supylabel('F1-Score',fontweight='bold')

            axsArr = axsArr + list(axs)
        for i in range(numFeats):
            axsArr[i].set_title(str(i+1) + " features")
            axsArr[i].plot(kVals,resultsPlot[i],color = "blue",marker = 'o')
            axsArr[i].set_xticks(kVals)
            if (i+1) % plotPerFig != 0 :
                axsArr[i].xaxis.set_visible(False)
        for i in range(len(figArr)):
            figName = "/Features "+str(i+1)+"-"+str((i+1)*plotPerFig)+".png"
            figArr[i].savefig(dirPlots+figName)
    
    bestResults = results == results.max().max()
    fig,ax = plt.subplots(figsize = figSizeGen)
    fig.suptitle("Best Results Table")
    plotBestResults = []
    for col in bestResults.columns:
        rows = list(bestResults[col][bestResults[col] == True].index)
        for row in rows:
            combFeats = list_of_multiple_features[labels_nr_features.index(row)]
            X_test = testX[:,list_of_multiple_features[labels_nr_features.index(row)]]
            model = allModels.loc[(row,col)]
            metrics,predVal = modelMetrics(model,X_test,testY)
            plotBestResults.append([int(col),len(combFeats),results.loc[(row,col)],metrics[3]])
    
    table = ax.table(cellText = plotBestResults ,colLabels = ["K","Features","Val(idation F1-Score","Best F1-Score"], loc = "center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    ax.axis('tight')
    ax.axis('off')
    fig.savefig(dirPlots+"/bestResultsTable.png")
    plt.show()


def mlpParamAnalysis(figTitle,input,target,classes,numNeurons,funcAct,learnRate,momentum,dirPlots,typeLearnRate="constant"):
    #TVT 40-30-30
    trainRatio, valRatio, testRatio = 0.4, 0.3, 0.3
    trainSetTVT, valSetTVT, testSetTVT = TVTDataSplit(input,np.arange(len(input)),trainRatio,valRatio,testRatio)

    #Questão 4.1
    scores = []
    models = []

    trainY,valY,testY = target[trainSetTVT[1]], target[valSetTVT[1]], target[testSetTVT[1]]

    for neurons in numNeurons:
        model = MLPClassifier(hidden_layer_sizes=(neurons,),activation=funcAct,solver='sgd', learning_rate_init=learnRate,learning_rate=typeLearnRate,momentum=momentum)

        model.fit(trainSetTVT[0],trainY)

        metrics,predVal = modelMetrics(model,valSetTVT[0],valY)
        scores.append(metrics[3])
        models.append(model)

    fig = plt.figure()
    fig.suptitle(figTitle + " Score per num neutrons")
    plt.plot(numNeurons,scores,color = "indigo")
    plt.scatter(numNeurons[scores.index(max(scores))], max(scores), color='r')
    plt.ylabel("F1-Score")
    plt.xlabel("Number of Neurons")
    figCop = figTitle.replace(" ","")
    saveFig(fig,dirPlots,figCop)
    bestModel = models[scores.index(max(scores))]
    metrics,predVal = modelMetrics(bestModel,testSetTVT[0],testY)
    showMetrics(figTitle,classes,metrics,dirPlots)

# Implementação do ReliefF para retornar os índices e não os nomes, como na implementação acima
def reliefFrank(x, y):
    fs = ReliefF(n_neighbors=100, n_features_to_keep=4)
    fs.fit_transform(x, y)
    return fs.top_features


def forward_feature_selection(trainX, trainY, testX,testY, n):
    feature_set = []
    
    for num_features in range(n):
        metric_list = [] # Choose appropriate metric based on business problem
        model = KNeighborsClassifier() # You can choose any model you like, this technique is model agnostic
        for feature in trainX.columns:
            if feature not in feature_set:
                f_set = feature_set.copy()
                f_set.append(feature)
                model.fit(trainX[f_set], trainY)
                metricsKnn,predVal = modelMetrics(model,testX[f_set],testY)
                metric_list.append((metricsKnn[3], feature))
        metric_list.sort(key=lambda x : x[0], reverse = True) # In case metric follows "the more, the merrier"
        feature_set.append(metric_list[0][1])
    return feature_set


def plotKnnTestResults(title,kVals,scoresDic,classes,dirPlot,figTitle):

    colLabs = ["Precision","Recall","F1-Score","K"]
    
    fig = plt.figure(figsize = figSizeGen)
    fig.suptitle(title)

    G = gridspec.GridSpec(2,2)

    axsPlot = plt.subplot(G[:,0])
    axsTable = plt.subplot(G[0,1])
    axsMatConf = plt.subplot(G[1,1])

    axsPlot.plot(kVals,scoresDic["Precision"],label= "Precision",color = "red")
    axsPlot.plot(kVals,scoresDic["Recall"],label = "Recall",color = "blue")
    axsPlot.plot(kVals,scoresDic["F1-Score"],label = "F1-Score",color = "orange")
    axsPlot.set_xlabel("K-Neighbours",fontweight= "bold")
    axsPlot.set_ylabel("Score",fontweight= "bold")
    axsPlot.legend()

    axsTable.set_title("Test Results of the best model")
    table = axsTable.table(cellText = [scoresDic["Best"][0]],colLabels=colLabs,loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    axsTable.axis('off')
    axsTable.axis('tight')
    

    axsMatConf.set_title("Matrix Confusion")
    confMatrix =  pd.DataFrame(data = scoresDic["Best"][1] ,columns=classes,index=classes)
    sb.heatmap(confMatrix , annot = True ,cmap='viridis', ax=axsMatConf)
    axsMatConf.axis('tight')
    saveFig(fig,dirPlot,figTitle)

def saveFig(fig,dirPlot,figTitle):
    if not path.exists(dirPlot):
        mkdir(dirPlot)
    fig.savefig(dirPlot+"/"+figTitle+".png")

