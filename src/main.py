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



#3.8 - Injeçao de Outliers
"""

partData = extractPartData(dirParts, chosenParticipant)
actSensData = getActivityData(partData, chosenActivity,chosenSensorId)
sampleData = getVectorModule(actSensData,indexModule[strAcc])

#Devolve Tuplo com (indices dos outliers, data com outliers)
outliers = outliersInsertion(sampleData,5,3)

"""

#3.9 - Modulo linear com base de amostras de p valores anteriores

"""
sample = np.array([4,3,7,8,2,6,7,9,1,5,10,32,9])
p = 3
print(trainLinearModel(sample,len(sample)-p,p))

"""

#3.10 e 3.11 - Modulos Lineares aplicados aos modulos de cada variavel de uma so atividade

#Neste caso vamoos aplicar o LOOCV(Leave one out Cross Validation) -
#https://www.youtube.com/watch?v=fSytzGwwBVw&ab_channel=StatQuestwithJoshStarmer

chosenAct = 1
allData  = getAllPartData(dirParts,maxPart)        
ActData = getActivityData(allData,chosenAct,None)

#Modulos das diferentes variaveis
AccModData = getVectorModule(ActData,indexModule[strAcc])
MagModData = getVectorModule(ActData,indexModule[strMag])
GirModData = getVectorModule(ActData,indexModule[strGir])

k=3.5
perOut = 10
janelasVal = range(5,20)
numPoints = 50

#---------------MODULO DE ACELERAÇÃO--------------------------

#injeçao de outliers para os substituir com o modelo linear preditivo de cada modulo
#devolve tuplo (indicesOutliers, dataWithOutliers)

outliersInfoAcc = outliersInsertion(np.copy(AccModData),perOut,k)
indicesOutliersAcc = outliersInfoAcc[0]

#Teste dos modelos para janelas de diferentes tamanhos
#Devolve tuplo com o array do somatorio de erros quadraticos para cada janela de cada modelo
#E com o array de valores previstos de cada modelo juntamente com os valores reais

infArrAcc = testPVals(janelasVal,AccModData,indicesOutliersAcc)

arrMeanSquareModels = (infArrAcc[0],infArrAcc[1])
realPredValModelPrev = infArrAcc[2]
realPredValModelPrevNext  = infArrAcc[3]

#Plots dos modelos em relação ao erro quadratico por janela
plotModelsQuadError(janelasVal,arrMeanSquareModels,strAcc)
plt.figure()
plt.subplot(2,1,1)

#Scatter dos valores previstos e reais de cada modelo
scatterPlotRealPredicted(realPredValModelPrev[0][:numPoints],realPredValModelPrev[1][:numPoints],
                        strAcc,"Modelo Linear de p valores anteriores")
plt.subplot(2,1,2)
scatterPlotRealPredicted(realPredValModelPrevNext[0][:numPoints],realPredValModelPrevNext[1][:numPoints],
                        strAcc,"Modelo Linear p/2 valores anteriores e seguintes")

#---------------MODULO DE MAGNOMETRO--------------------------

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

#---------------MODULO DE GIROSCOPIO-------------------------

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

    Conclusão sobre a comparação dos 2 modelos

        O modelo linear com base em p valores anteriores demonstra bons resultados no entanto
    o modelo linear com base em p/2 valores anteriores e seguintes apresenta bastante melhores resultados.
    Logo caso quisesse prever valores para missing data utilizaria o segundo modelo

"""

plt.show()

print("Done")
