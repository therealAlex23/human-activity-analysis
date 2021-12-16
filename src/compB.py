from utilsCompB import *
from utilsCompA import * 
from globals import *

#------------------- COMPONENTE B --------------


#Importing data with features
chosenSensorId = 1
chosenParticipant = 0
debug = False

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

#Questão 4.5


#Questão 4.6



#QUESTÃO 1.1

#1.1.1 Train-Test (TT) e Train-Validation-Test(TVT) data split
#TT

lin, col = 20, 2

sampX,sampY = np.arange(20,20+lin),np.arange(20)

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

iris = datasets.load_iris()

irisData = iris.data
irisTarget = iris.target
irisClasses = iris.target_names
irisFeats = iris.feature_names

#Type of metrics
avgMetrics = "weighted"

# ----> Questão 2.1

kNeigh = 1
strategyDummy = "uniform"
# Train-only

dummy_clf = DummyClassifier(strategy=strategyDummy)
clf = DecisionTreeClassifier(random_state=0,max_depth=1)

dummy_clf.fit(irisData, irisTarget)
metricsDummy,predVal = modelMetrics(dummy_clf,irisData, irisTarget, avgMetrics)

clf.fit(irisData, irisTarget)
metricsOneClf,predVal = modelMetrics(dummy_clf,irisData, irisTarget, avgMetrics)

#showMetrics("Dummy Classifier using Train-Only",irisClasses,metricsDummy)
#showMetrics("OneR Classifier using Train-Only",irisClasses,metricsOneClf)

#TT train ratio/test ratio

trainRatio = 0.7
trainSet,testSet = TTDataSplit(irisData,np.arange(len(irisData)),trainRatio)
trainY,testY = irisTarget[trainSet[1]], irisTarget[testSet[1]]

#Dummy Classifier
dummy_clf.fit(trainSet[0],trainY)
metricsDummy,predVal = modelMetrics(dummy_clf,testSet[0],testY,avgMetrics)

#One-R Classifier
clf.fit(trainSet[0],trainY)
metricsOneClf,predVal = modelMetrics(clf,testSet[0], testY,avgMetrics)

#showMetrics("Dummy Classifier using Train-Test 70-30",irisClasses,metricsDummy)
#showMetrics("One-R Classifier using Train-Test 70-30",irisClasses,metricsOneClf)

# ------- K FOLD CV (10 CV) --------

folds = 10
#Dummy Classifier
#KFoldAnalysis(dummy_clf,"Dummy Classifier",irisClasses,folds,irisData,irisTarget,avgMetrics)

#OneR Classifier
#KFoldAnalysis(clf,"One-R Classifier",irisClasses,folds,irisData,irisTarget,avgMetrics)

#--------- QUESTÃO 2.2.1 ----------

#Train-Test 70-30

trainRatio = 0.7
#K-Fold 10x10(10CV)
folds = 10
#KnnTrainParamAnalysis(irisData,irisTarget,trainRatio,folds,irisClasses)

#--------- QUESTÃO 2.2.2 ----------
#With different k values of neighbors k = {1,3,5,...,15}

kVals = range(1,16,2)
folds = 10
trainRatio,valRatio, testRatio = 0.4,0.3,0.3
distMetricChosen = "euclidean"

#Train-Only
#KNNTrainOnlyAnalysis(irisData,irisTarget,distMetricChosen,kVals,irisClasses,dir2_2_2)

#TVT 40-30-30
#KnnTVTAnalysis(trainRatio,valRatio,testRatio,irisData,irisTarget,distMetricChosen,kVals,irisClasses,dir2_2_2)

#K-Fold 10x10 (10CV) - Se calhar falta avg e std
#KnnKFoldAnalysis(folds,irisData,irisTarget,distMetricChosen,kVals,irisClasses,dir2_2_2+"/Kfold CV(" + str(folds) + ")")

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

    -------- QUESTÃO 2.2.3 POR FAZER 

    Train-Only: Low Bias --> it uses the whole data as training set, so there's no bias
                Low Variance --> the pred
    TT 70-30: Low bias and

"""

#------- QUESRÃO 2.3 ---------
#Questão 2.3.1

fsNames = np.array(irisFeats)

#get feature scores and rank them
#https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/similarity_based/reliefF.py
#https://towardsdatascience.com/feature-importance-and-forward-feature-selection-752638849962

#featureRanking = reliefFrank(irisData,irisTarget)
#print(featureRanking)

#TVT for data-splitting
#Questão 2.3.2 e 2.3.3
#KnnNumFeatAnalysis(irisData,irisTarget,kVals,distMetricChosen,featureRanking,dir2_3)

#Questão 2.3.4


#QUESTÃO 2.4

"""
trainRatio= 0.7
trainSet, testSet = TTDataSplit(irisData,np.arange(len(irisData)),trainRatio)

trainX,trainY = trainSet[0], irisTarget[trainSet[1]]
testX,testY = testSet[0], irisTarget[testSet[1]]

trainXdf = pd.DataFrame(trainX, columns=fsNames)
testXdf = pd.DataFrame(testX, columns=fsNames)
featRank = forward_feature_selection(trainXdf,trainY,testXdf,testY,4)
featRankSort = list(map(lambda x: irisFeats.index(x), featRank))

featRankSort = np.array(featRankSort)

combFeats = []
distMetricChosen = "euclidean"
trainTTRatio = 0.7
trainRatio,valRatio, testRatio = 0.4,0.3,0.3

for ind in featRankSort:

    combFeats.append(ind)
    print(fsNames[combFeats])
    dataX = irisData[:,combFeats]

    #2.2.1
    folds = 10
    KnnTrainParamAnalysis(dataX,irisTarget,trainTTRatio,folds,irisClasses)

    #2.2.2
    #Train-Only
    KNNTrainOnlyAnalysis(dataX,irisTarget,distMetricChosen,kVals,irisClasses)

    #TVT 40-30-30
    KnnTVTAnalysis(trainRatio,valRatio,testRatio,dataX,irisTarget,distMetricChosen,kVals,irisClasses)

    #K-Fold 10x10 (10CV)
    KnnKFoldAnalysis(folds,dataX,irisTarget,distMetricChosen,kVals,irisClasses)

"""


#QUESTÃO 2.5
irisDf = pd.DataFrame(irisData,columns=fsNames)

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

#featureScores = reliefF(sampsDataX,sampsDataY)
#featureRanking = feature_ranking(featureScores)

#KnnNumFeatAnalysis(sampsDataX,sampsDataY,kVals,distMetricChosen,np.array(featureRanking))

#---> QUESTÃO 3.1

actLab = list(activityLabels.values())  
datasetFeats = dataset.drop(columns = ["act"])

dataX = np.array(datasetFeats)
target = np.array(list(map(lambda x: actLab.index(x), dataset["act"])))

#Train-Only
dummyClf = DummyClassifier()
dummy_clf.fit(dataX,target)
metricsDummy,predVal = modelMetrics(dummy_clf,dataX,target)
#showMetrics("Dummy Classifier using Train-Only",actLab,metricsDummy)

#TT 70-30
trainRatio = 0.7
trainSet,testSet = TTDataSplit(dataX,np.arange(len(dataX)),trainRatio)
trainY,testY = target[trainSet[1]], target[testSet[1]]

dummy_clf.fit(trainSet[0],trainY)
metricsDummy,predVal = modelMetrics(dummy_clf,testSet[0],testY)
#showMetrics("Dummy Classifier using Train-Test 70-30",actLab, metricsDummy)

#K-Fold
#KFoldAnalysis(dummy_clf,"Random Classifier",actLab,folds,dataX,dataY)

#Questão 3.2.1
#Train-Only
#KnnTrainParamAnalysis(dataX,dataY,trainRatio,folds,actLab)

#Questão 3.2.2
trainRatio,valRatio, testRatio = 0.4,0.3,0.3
distMetricChosen = "euclidean"

#Train-Only
#KNNTrainOnlyAnalysis(dataX,dataY,distMetricChosen,kVals,actLab)

#TVT 40-30-30
#KnnTVTAnalysis(trainRatio,valRatio,testRatio,dataX,datasX,distMetricChosen,kVals,actLab)

#K-Fold 10x10 (10CV)
#KnnKFoldAnalysis(folds,dataX,dataY,distMetricChosen,kVals,actLab)

#Questão 3.2.3

#Questão 3.3
fsNames = np.array(datasetFeats.columns.values)

#get feature scores and rank them
#https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/similarity_based/reliefF.py
#https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis

#featureScores = reliefFrank(dataX[:,:30],target)
#print(featureScores)

#KnnNumFeatAnalysis(dataX[:3000],target[:3000],kVals,distMetricChosen,featureScores,dir3_3)

"""
#------ QUESTÃO 4 -------

Layers = 3
Neurons = variable
Activation = Logistic

"""
numNeurons = [i for i in range(4,16)]
funcAct ='logistic'
learningRate = 0.1
dataSelectFeats = dataX[:,:40]
momentum = 0.9

#Questão 4.1
mlpParamAnalysis("MLP Constant Learn Rate",dataSelectFeats,target,
                        actLab,numNeurons,funcAct,learningRate,momentum,dir4_1)
#Questão 4.2

typeLearnRate = "invscaling"
mlpParamAnalysis("MLP Variable Learn Rate",dataSelectFeats,target,actLab,numNeurons,funcAct,0.2,momentum,dir4_2,typeLearnRate)

#Questão 4.3

momentum = 0.4
mlpParamAnalysis("MLP Momentum Change",dataSelectFeats,target,
                        actLab,numNeurons,funcAct,learningRate,momentum,dir4_3)
                        
#Questão 4.4


print("Done")

