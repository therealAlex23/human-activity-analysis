from utilsCompB import *
from utilsCompA import * 
from globals import *

#------------------- COMPONENTE B --------------


#Importing data with features
chosenSensorId = 1

sFreq = 51.2  # as described in dataset
windowDuration = 2  # seconds
windowSize = round(windowDuration * sFreq)
overlap = windowSize // 2  # 50% overlap as described in https://bit.ly/tcd-paper

# df and energy are not
# statistical features but are
# computed for each axis, like
# the other metrics in this list.

stats = [
    np.mean, np.median, np.std,
    stats.skew, stats.kurtosis,
    stats.iqr, np.var, zcr, df,
    energy
]

# list of general physical features
phys = [
    cagh, avgd,
    avhd,  # aratg
]

#dataset = genData(chosenSensorId,windowSize,overlap,phys,stats)
dataset = pd.read_csv(path.join(dirOutput, f'dev{chosenSensorId}.csv'))
actLab = list(activityLabels.values())  
datasetFeats = dataset.drop(columns = ["act"])

dataX = np.array(datasetFeats)
target = np.array(list(map(lambda x: actLab.index(x), dataset["act"])))

#Questão 4.5
#https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/

if flag4_5A == True:
    print("4.5A running...\n")
    if not path.exists(dir4_5A):
        mkdir(dir4_5A)
    
    #Fisher Score
    figSizeRank = (25,35)
    featRankFishScr = fisher_score(dataX,target)
    featImp = pd.DataFrame(data = featRankFishScr, index = datasetFeats.columns,columns = ["Score"])
    featImp.plot(kind = "barh",color = "teal",figsize = figSizeRank)
    fisherBest = featImp.nlargest(10,"Score")
    print(fisherBest)
    fig = plt.gcf()
    plt.xticks(fontsize = 40)
    plt.yticks(fontsize = 20)
    saveFig(fig,dir4_5A,"Fisher Score")

    #ReliefF

    featRankRelief = reliefFrank(dataX,target)
    featRankRelief = pd.DataFrame(data = featRankRelief,index = datasetFeats.columns,columns= ["Score"])
    reliefFbest = featRankRelief.nlargest(10,"Score")
    print(reliefFbest)
    featRankRelief.plot(kind = "barh",color = "teal",figsize = figSizeRank)
    fig = plt.gcf()
    plt.xticks(fontsize = 40)
    plt.yticks(fontsize = 20)
    saveFig(fig,dir4_5A,"ReliefF Score")
    print("\n4.5A Done")

#Questão 4.6

#QUESTÃO 1.1

#1.1.1 Train-Test (TT) e Train-Validation-Test(TVT) data split
lin, col = 20, 2
sampX,sampY = np.arange(20,20+lin),np.arange(20)

#TT
testRatio = 0.6
trainSet, testSet = TTDataSplit(sampX,sampY,testRatio)

#TVT
trainRatio, valRatio, testRatio = 0.4, 0.3, 0.3
trainSet, valSet, testSet   = TVTDataSplit(sampX,sampY,trainRatio,valRatio,testRatio)

"""
print("X Train: " + str(trainSet[0]))
print("Y Train: " + str(trainSet[1]))

print("\nX Val: " + str(valSet[0]))
print("Y Val: " + str(valSet[1]))

print("\nX Test: " + str(testSet[0]))
print("Y Test: " + str(testSet[1]))
"""

#1.1.2 K-Fold Data Split ( encontrar como definir o maximo de numero de splits)

nSplits = 3
trainRatio = 0.7

splits = kFoldDataSplit(sampX,nSplits)

"""
numSplit= 1
for trainS, testS in splits:
    print("SPLIT "+str(numSplit))
    print("\nTrainSet\nX: "+str(trainS[0]) +"\nY: "+str(trainS[1])+"\n")
    print("TestSet\nX: "+str(testS[0]) +"\nY: "+str(testS[1])+"\n")
    numSplit+=1
"""


#1.2 EVALUATION METRICS 

"""

    Classes: True Positives (tp), False Positives(fp), True Negatives(tn), False Negatives(fn)
             The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

    Metrics:

        - Precision = tp / (tp + fp) where tp is the number of true positives and fp the number of false positives

        - Recall = ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives;
                   The recall is intuitively the ability of the classifier to find all the positive samples.

        - F-Measure = F1 = 2 * (precision * recall) / (precision + recall). 
                      The F1 score is the harmonic mean of the precision and recall. 

        - Confusion Matrix = A confusion matrix is a technique for summarizing the performance of a classification algorithm.
                             The number of correct and incorrect predictions are summarized with count values and broken down by each class.
                            
    
    If average = None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data

    - Weighted = weighted average of a metric for individual classes
    - Macro = arithmetic average of a mretric for individual classes
    - Micro = Metric of the entire dataset (i.e., without considering individual classes) = accuracy

    https://stackoverflow.com/questions/68708610/micro-metrics-vs-macro-metrics
    https://en.wikipedia.org/wiki/F-score

"""

Y_true = [1,5,3,3,4,2,1,5,1]
Y_pred = [1,5,9,3,2,2,0,2,7]

metricsVal = getEvalMetrics(Y_true,Y_pred,None)

#/////////////// QUESTÃO 2 ///////////////

#Dataset Iris to experiment the functions
folds = 10
iris = datasets.load_iris()

irisData = iris.data
irisTarget = iris.target
irisClasses = iris.target_names
irisFeats = iris.feature_names

#Type of metrics
# ----> Questão 2.1

if flag2_1 == True:
    print("2.1 running...")

    kNeigh = 1
    strategyDummy = "uniform"

    for d in dirList:
        if not path.exists(d):
            mkdir(d)

    dirDummy2_1 = dir2_1+dirDummyClf
    if not path.exists(dirDummy2_1):
        mkdir(dirDummy2_1)

    dirOneRClf2_1 = dir2_1+dirOneRClf
    if not path.exists(dirOneRClf2_1):
        mkdir(dirOneRClf2_1)

    # Train-only

    dummy_clf = DummyClassifier(strategy=strategyDummy)
    clf = DecisionTreeClassifier(random_state=0,max_depth=1)

    dummy_clf.fit(irisData, irisTarget)
    metricsDummy,predVal = modelMetrics(dummy_clf,irisData, irisTarget)

    clf.fit(irisData, irisTarget)
    metricsOneClf,predVal = modelMetrics(dummy_clf,irisData, irisTarget)

    showMetrics("Dummy Classifier using Train-Only",irisClasses,metricsDummy,dirDummy2_1)
    showMetrics("OneR Classifier using Train-Only",irisClasses,metricsOneClf,dirOneRClf2_1)

    #TT train ratio/test ratio
    trainRatio = 0.7
    trainSet,testSet = TTDataSplit(irisData,np.arange(len(irisData)),trainRatio)
    trainY,testY = irisTarget[trainSet[1]], irisTarget[testSet[1]]

    #Dummy Classifier
    dummy_clf.fit(trainSet[0],trainY)
    metricsDummy,predVal = modelMetrics(dummy_clf,testSet[0],testY)

    #One-R Classifier
    clf.fit(trainSet[0],trainY)
    metricsOneClf,predVal = modelMetrics(clf,testSet[0], testY)

    showMetrics("Dummy Classifier TT 70-30",irisClasses,metricsDummy,dirDummy2_1)
    showMetrics("One-R Classifier TT 70-30",irisClasses,metricsOneClf,dirOneRClf2_1)

    # ------- K FOLD CV (10 CV) --------
    #Dummy Classifier

    dirDumFolds = dirDummy2_1+dirFolds
    if not path.exists(dirDumFolds):
        mkdir(dirDumFolds)

    KFoldAnalysis(dummy_clf,"Dummy Classifier",irisClasses,folds,irisData,irisTarget,dirDumFolds)

    dirOneRFolds = dirOneRClf2_1+dirFolds
    if not path.exists(dirOneRFolds):
        mkdir(dirOneRFolds)
    #OneR Classifier
    KFoldAnalysis(clf,"One-R Classifier",irisClasses,folds,irisData,irisTarget,dirOneRFolds)
    print("2.1 Done")

#--------- QUESTÃO 2.2.1 ----------

#Train-Test 70-30
if flag2_2_1 == True:
    print("2.2.1 running...")

    trainRatio = 0.7
    #K-Fold 10x10(10CV)
    folds = 10
    KnnTrainParamAnalysis(irisData,irisTarget,trainRatio,folds,irisClasses,dir2_2_1)
    print("2.2.1 Done")

#--------- QUESTÃO 2.2.2 ----------
#With different k values of neighbors k = {1,3,5,...,15}
kVals = range(1,16,2)
distMetricChosen = "euclidean"

if flag2_2_2 == True:
    print("2.2.2 running...")

    folds = 10
    trainRatio,valRatio, testRatio = 0.4,0.3,0.3

    #Train-Only
    KNNTrainOnlyAnalysis(irisData,irisTarget,distMetricChosen,kVals,irisClasses,dir2_2_2)

    #TVT 40-30-30
    KnnTVTAnalysis(trainRatio,valRatio,testRatio,irisData,irisTarget,distMetricChosen,kVals,irisClasses,dir2_2_2)

    #K-Fold 10x10 (10CV) - Se calhar falta avg e std
    dirFolds2_2_2 = dir2_2_2+"/Kfold CV(" + str(folds) + ")"
    if not path.exists(dirFolds2_2_2):
        mkdir(dirFolds2_2_2)

    KnnKFoldAnalysis(folds,irisData,irisTarget,distMetricChosen,kVals,irisClasses,dirFolds2_2_2)
    print("2.2.2 Done")


"""
    --> Underfitting:

        - When the model has a high error rate in the training data
        - We can say the model is underfitting.
        - This usually occurs when the number of training samples is too low. 
    
    --> Overfitting:

        - When the model has a low error rate in training data but a high error rate in testing data, we can say the model is overfitting. 
        - This usually occurs when the number of training samples is too high
    
    --> Bias:

        - Assuming we have trained the model and are trying to predict values with input ‘x_train’. 
        - The predicted values are y_predicted. Bias is the error rate of y_predicted and y_train.
        - Error rate High = High Bias 
        - Error rate Low = Low Bias
    
    --> Variance:

        - Assume we have trained the model and this time we are trying to predict values with input ‘x_test’.
        - Variance is the error rate of the y_predicted and y_test
        - Error rate High = High Variance
        - Error rate Low = Low Variance
    
    NOTES:

        - Parametric or linear machine learning algorithms often have a high bias but a low variance.
        - Non-parametric or non-linear machine learning algorithms often have low bias but high variance.


"""

#------- EX 2.3 ---------
#https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/similarity_based/reliefF.py
#https://towardsdatascience.com/feature-importance-and-forward-feature-selection-752638849962

if flag2_3 == True:
    print("2.3 running...")

    #Questão 2.3.1
    #get feature scores and rank them

    featureRanking = reliefFrank(irisData,irisTarget)

    #TVT for data-splitting
    #Questão 2.3.2 e 2.3.3

    KnnNumFeatAnalysis(irisData,irisTarget,kVals,distMetricChosen,featureRanking,dir2_3)
    print("2.3 Done")

    #Questão 2.3.4

#QUESTÃO 2.4
if flag2_4 == True:
    print("2.4 running...")

    knnFeatSelectionAnalysis(irisData,irisTarget,kVals,irisFeats,irisClasses,dir2_4)
    print("2.4 Done")


#QUESTÃO 2.5
if flag2_5 == True:
    print("2.5 running...")

    irisDf = pd.DataFrame(irisData,columns=irisFeats)

    #Grouping the samples
    irisDf["Target"] = irisTarget
    Setosa = irisDf[irisDf["Target"] == 0]
    Versicolour = irisDf[irisDf["Target"] == 1].sample(n=30)
    Virginica = irisDf[irisDf["Target"] == 2].sample(n=10)

    #Building the data to further analysing
    sampsData = Setosa.append(Versicolour)
    sampsData = sampsData.append(Virginica)

    #For the data splitting
    sampsDataX = np.array(sampsData.drop(columns=["Target"]))
    sampsDataY = np.array(sampsData["Target"])

    #TVT 40-30-30
    trainRatio, valRatio, testRatio = 0.4, 0.3, 0.3
    trainSetTVT, valSetTVT, testSetTVT = TVTDataSplit(sampsDataX,np.arange(len(sampsDataX)),trainRatio,valRatio,testRatio)

    featureScores = reliefF(sampsDataX,sampsDataY)
    featureRanking = feature_ranking(featureScores)

    KnnNumFeatAnalysis(sampsDataX,sampsDataY,kVals,distMetricChosen,np.array(featureRanking),dir2_5)
    print("2.5 Done")

#---> QUESTÃO 3.1

if flag3_1 == True:
    print("3.1 running...")

    plotParams = {"titleFont": 40, "figSize": (20,15), "matScale": (1,2.5),
                "matFont": 18,"testScale":(1,3), "testTitleFont" : 25,
                "testFont":18, "matTitleFont" : 20}

    #Train-Only
    figSize = (18,15)
    oneRclf = DecisionTreeClassifier(random_state=0,max_depth=1)
    oneRclf.fit(dataX,target)
    metricsDummy,predVal = modelMetrics(oneRclf,dataX,target)
    showMetrics("OneRule Train-Only",actLab,metricsDummy,dir3_1,plotParams)

    #TT 70-30
    trainRatio = 0.7
    trainSet,testSet = TTDataSplit(dataX,np.arange(len(dataX)),trainRatio)
    trainY,testY = target[trainSet[1]], target[testSet[1]]

    oneRclf.fit(trainSet[0],trainY)
    metricsDummy,predVal = modelMetrics(oneRclf,testSet[0],testY)
    showMetrics("OneRule TT 70-30",actLab, metricsDummy,dir3_1,plotParams)

    #K-Fold
    dirFolds3_1 = dir3_1+dirFolds
    if not path.exists(dirFolds3_1):
        mkdir(dirFolds3_1)
    
    KFoldAnalysis(oneRclf,"OneRule",actLab,folds,dataX,target,dirFolds3_1,plotParams)

    print("3.1 Done")
    
#Questão 3.2.1
#Train-Only
if flag3_2_1 == True:
    print("3.2.1 running...")

    plotParams = {"figSize": (35,20), "matScale": (1.25,2.5),"matFont": 16,"tabScale":(1,3), "testFont":20,"metricFont": 20}

    KnnTrainParamAnalysis(dataX,target,trainRatio,folds,actLab,dir3_2_1,plotParams)

    print("3.2.1 Done")


#Questão 3.2.2
if flag3_2_2 == True:
    print("3.2.2 running...")

    trainRatio,valRatio, testRatio = 0.4,0.3,0.3
    distMetricChosen = "euclidean"

    #Train-Only
    KNNTrainOnlyAnalysis(dataX,target,distMetricChosen,kVals,actLab,dir3_2_2)

    #TVT 40-30-30
    KnnTVTAnalysis(trainRatio,valRatio,testRatio,dataX,target,distMetricChosen,kVals,actLab,dir3_2_2)

    #K-Fold 10x10 (10CV)
    
    KnnKFoldAnalysis(folds,dataX,target,distMetricChosen,kVals,actLab,dir3_2_2)
    print("3.2.2 Done")

#Questão 3.2.3

#Questão 3.3
#https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/similarity_based/reliefF.py

if flag3_3 == True:
    print("3.3 running...")
    fsNames = np.array(datasetFeats.columns.values)

    featureScores = reliefFrank(dataX,target)

    KnnNumFeatAnalysis(dataX,target,kVals,distMetricChosen,featureScores[:30],dir3_3)
    print("3.3 Done")

"""
#------ QUESTÃO 4 -------

Layers = 3
Neurons = variable
Activation = Logistic

"""

featureScores = reliefFrank(dataX[:10000],target)
numNeurons = [i for i in range(4,16)]
funcAct ='logistic'
learningRate = 0.1
dataSelectFeats = dataX[:,featureScores[:40]]
momentum = 0.9

#Questão 4.1

plotParams = {"titleFont": 40, "figSize": (20,15), "matScale": (1,2.5),
                "matFont": 18,"testScale":(1,3), "testTitleFont" : 25,
                "testFont":18, "matTitleFont" : 20}

if flag4_1 == True:
    print("4.1 running...")

    mlpParamAnalysis("MLP Constant Learn Rate",dataSelectFeats,target,
                        actLab,numNeurons,funcAct,learningRate,momentum,dir4_1,plotParams= plotParams)
    print("4.1 Done")
    
#Questão 4.2
if flag4_2 == True:
    print("4.2 running...")
    typeLearnRate = "invscaling"
    mlpParamAnalysis("MLP Variable Learn Rate",dataSelectFeats,target,actLab,numNeurons,funcAct,0.2,momentum,dir4_2,typeLearnRate,plotParams)
    print("4.2 Done")

#Questão 4.3
if flag4_3 == True:
    print("4.3 running...")
    momentum = 0.4
    mlpParamAnalysis("MLP Momentum Change",dataSelectFeats,target,
                            actLab,numNeurons,funcAct,learningRate,momentum,dir4_3,plotParams = plotParams)
    print("4.3 Done")

#Questão 4.4

print("Done")

