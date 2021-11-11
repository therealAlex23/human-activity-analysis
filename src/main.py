from os import access
from numpy.core.numeric import outer
from numpy.lib.function_base import select
from utils import *

import warnings
warnings.filterwarnings('ignore')

# Globals

strAcc,strGir,strMag = "Aceleração","Giroscopio","Magnometro"

dirParts = "../assets/part"
maxPart = 15
noOfSensors = 5

indexModule = {strAcc: 1, strGir: 4, strMag: 7}

activityLabels = {
    1: 'Stand', 2: 'Sit', 3: 'Sit and Talk', 4: 'Walk', 5: 'Walk and Talk',
    6: 'Climb Stair(up/down)', 7: 'Climb(up/down)', 8: 'Stand -> Sit',
    9: ' Sit -> Stand', 10: 'Stand -> Sit and Talk', 11: 'Sit -> Stand and talk',
    12: 'Stand -> Walk', 13: 'Walk -> Stand',
    14: 'Stand -> climb stairs (up/down), stand -> climb stairs (up/down) and talk',
    15: 'Climb stairs (up/down) -> walk',
    16: 'Climb stairs (up/down) and talk -> walk and talk'}

deviceID = {1: 'Pulso Esquerdo', 2: 'Pulso direito', 3: 'Peito',
            4: 'Perna superior direita', 5: 'Perna inferior esquerda'
            }


# Questao 2 - Importação de Dados
# allData = getAllPartData(dirParts + "part", maxPart)

# Questao 3.2 - Densidade dos outliers
# getDensityOutliers(allData,activityLabels, 1)

# Questao 3.1 e 3.2
# allData = getAllPartData(dirParts + "part", maxPart)

# Questao 3.3 e 3.4
"""
chosenParticipant = 0
chosenActivity = 8
data = extractPartData(dirParts, 0)
fig, axs = plt.subplots(3)
plotOutliers(data, 4, chosenActivity, 1, axs)
fig.suptitle(
    f"Part {chosenParticipant} - {activityLabels.get(chosenActivity)}")
plt.show()
"""

# Questão 3.6
# variables --
#chosenParticipant = 0
#chosenActivity = 2
#chosenSensorId = 1
# ------------
#data = extractPartData(dirParts, chosenParticipant)

# ignore activityId, sensorId and timestamp
#device_data = getActivityData(data, chosenActivity, chosenSensorId)[:, 1:-2]

# acc data for chosenActivity only
# kmeans1(device_data[:, :3], 3, 10)
# https://www.youtube.com/watch?v=_aWzGGNrcic
# centroids, groups = kmeans2(device_data[:, :3], 3, 1)
# plotKmeans(device_data[:, :3], centroids, groups)



#3.8 - Outlier Injection
"""

partData = extractPartData(dirParts, chosenParticipant)
actSensData = getActivityData(partData, chosenActivity,chosenSensorId)
sampleData = getVectorModule(actSensData,indexModule[strAcc])

#Devolve Tuplo com (indices dos outliers, data com outliers)
outliers = outliersInsertion(sampleData,5,3)

"""

#3.9 - Linear modulus based on samples of p previous values

"""
sample = np.array([4,3,7,8,2,6,7,9,1,5,10,32,9])
p = 3
print(trainLinearModel(sample,len(sample)-p,p))

"""

#3.10 e 3.11 - Linear Module applied to the moduli of each variable of a single activity

#Neste caso vamoos aplicar o LOOCV(Leave one out Cross Validation) -
#https://www.youtube.com/watch?v=fSytzGwwBVw&ab_channel=StatQuestwithJoshStarmer

chosenAct = 1
allData  = getAllPartData(dirParts,maxPart)        
ActData = getActivityData(allData,chosenAct,None)

#Modules of the different variables

AccModData = getVectorModule(ActData,indexModule[strAcc])
MagModData = getVectorModule(ActData,indexModule[strMag])
GirModData = getVectorModule(ActData,indexModule[strGir])

k=3.5
perOut = 10
janelasVal = range(5,20)
numPoints = 50

#--------------- ACCELERATION MODULUS  --------------------------

#inject outliers to replace them with the predictive linear model of each module
#return tuples (indicesOutliers, dataWithOutliers)

outliersInfoAcc = outliersInsertion(np.copy(AccModData),perOut,k)
indicesOutliersAcc = outliersInfoAcc[0]

# Test the models for different window sizes
# Returns the array of the sum of quadratic errors for each window of each model
# And the array of predicted values for each model along with the actual values

infArrAcc = testPVals(janelasVal,AccModData,indicesOutliersAcc)

arrMeanSquareModels = (infArrAcc[0],infArrAcc[1])
realPredValModelPrev = infArrAcc[2]
realPredValModelPrevNext  = infArrAcc[3]

#Plots of the models in relation to the quadratic error per window 
plotModelsQuadError(janelasVal,arrMeanSquareModels,strAcc)
plt.figure()
plt.subplot(2,1,1)

#Scatter of predicted and actual values for each model
scatterPlotRealPredicted(realPredValModelPrev[0][:numPoints],realPredValModelPrev[1][:numPoints],
                        strAcc,"Modelo Linear de p valores anteriores")
plt.subplot(2,1,2)
scatterPlotRealPredicted(realPredValModelPrevNext[0][:numPoints],realPredValModelPrevNext[1][:numPoints],
                        strAcc,"Modelo Linear p/2 valores anteriores e seguintes")

#--------------- MAGNOMETER MODULE --------------------------

outliersInfoMag = outliersInsertion(np.copy(MagModData),perOut,k)
indicesOutliersMag = outliersInfoMag[0]

infArrMag = testPVals(janelasVal,MagModData,indicesOutliersMag)

arrMeanSquareModels = (infArrMag[0],infArrMag[1])
realPredValModelPrev = infArrMag[2]
realPredValModelPrevNext  = infArrMag[3]

plotModelsQuadError(janelasVal,arrMeanSquareModels,strMag)
plt.figure()
plt.subplot(2,1,1)

scatterPlotRealPredicted(realPredValModelPrev[0][:numPoints],realPredValModelPrev[1][:numPoints],
                        strMag,"Modelo Linear de p valores anteriores")
plt.subplot(2,1,2)           
scatterPlotRealPredicted(realPredValModelPrevNext[0][:numPoints],realPredValModelPrevNext[1][:numPoints],
                        strMag,"Modelo Linear p/2 valores anteriores e seguintes")

#--------------- GYRO MODULE -------------------------

outliersInfoGir = outliersInsertion(np.copy(GirModData),perOut,k)
indicesOutliersGir = outliersInfoGir[0]

infArrGir = testPVals(janelasVal,GirModData,indicesOutliersGir)

arrMeanSquareModels = (infArrGir[0],infArrGir[1])
realPredValModelPrev = infArrGir[2]
realPredValModelPrevNext  = infArrGir[3]

plotModelsQuadError(janelasVal,arrMeanSquareModels,strGir)

plt.figure()
plt.subplot(2,1,1)
scatterPlotRealPredicted(realPredValModelPrev[0][:numPoints],realPredValModelPrev[1][:numPoints],
                        strGir,"Modelo Linear de p valores anteriores")
plt.subplot(2,1,2)
scatterPlotRealPredicted(realPredValModelPrevNext[0][:numPoints],realPredValModelPrevNext[1][:numPoints],
                        strGir,"Modelo Linear p/2 valores anteriores e seguintes")

"""

     Conclusion on the comparison of the 2 models

        The linear model based on p prior values shows good results however
    the linear model based on p/2 prior and subsequent values shows much better results
    and with smaller windows.
    So if I wanted to predict values for missing data I would use the second model

"""

plt.show()

print("Done")
