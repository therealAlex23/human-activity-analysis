from os import access
from numpy.core.numeric import outer
from numpy.lib.function_base import select
from utils import *
from os import path, mkdir
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# Globals

strAcc, strGir, strMag = "Aceleração", "Giroscopio", "Magnometro"

dirParts = "../assets/part"
dirOutput = "../outputs"
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

labels = ["acc", "gyro", "mag"]

# Questao 2 - Importação de Dados
# allData = getAllPartData(dirParts + "part", maxPart)

# Questao 3.2 - Densidade dos outliers
# getDensityOutliers(allData,activityLabels, 1)

# Questao 3.1 e 3.2
# allData = getAllPartData(dirParts + "part", maxPart)

# Questao 3.3 e 3.4
# -- Variables
"""
chosenParticipant = 0
chosenActivity = 8
k = 3
chosenSensorId = 1

data = extractPartData(dirParts, chosenParticipant)
sensorData = [
    getVectorModule(
        getActivityData(
            data,
            chosenActivity,
            chosenSensorId),
        i)
    for i in range(1, 8, 3)]

fig, axs = plt.subplots(3)
plotOutliers(sensorData, k, axs)
fig.suptitle(
    makeGraphTitle(
        chosenParticipant,
        activityLabels.get(chosenActivity),
        None,
        deviceID.get(chosenSensorId)
    )
)
plt.show()
"""

# Questão 3.6
# variables --
"""
chosenParticipant = 0
chosenActivity = 1
chosenSensorId = 1
k = 3  # z-score param
n = 5  # number of clusters
# ------------

data = extractPartData(dirParts, chosenParticipant)
#chosenParticipant = 0
#chosenActivity = 2
#chosenSensorId = 1
# ------------
#data = extractPartData(dirParts, chosenParticipant)

# ignore activityId, sensorId and timestamp
#device_data = getActivityData(data, chosenActivity, chosenSensorId)[:, 1:-2]

# acc data for chosenActivity only
# https://www.youtube.com/watch?v=_aWzGGNrcic
centroids, groups = kmeans2(device_data[:, :3], n)

fig = plt.figure()
ax = plt.axes(projection='3d')
outliers = getOutliers(device_data[:, :3], k)
plotKmeans(ax, device_data[:, :3], outliers, centroids, groups)
ax.set_title(
    makeGraphTitle(
        chosenParticipant,
        activityLabels.get(chosenActivity),
        "ACC",
        deviceID.get(chosenSensorId)
    )
    f"Part {chosenParticipant}/{activityLabels.get(chosenActivity)}/{deviceID.get(chosenSensorId)}")
plt.show()
"""

# Questão 4.1
# NH = Normal distribution and
# our data's underlying distribution are the same
# variables --
"""
chosenActivity = 16
chosenSensorId = 1
default_pvalue = 0.05  # if pvalue <= 0.05, then NH is rejected
# ------------
rejected, accepted = 0, 0
for participant in range(maxPart):
    data = extractPartData(dirParts, participant)

    # get acc (sensorData[0]), gyro (sensorData[1]),
    # and mag (sensorData[1]) vector modules
    sensorData = [
        getVectorModule(
            getActivityData(
                data,
                chosenActivity,
                chosenSensorId),
            i)
        for i in range(1, 8, 3)]
    for i in range(len(sensorData)):
        title = makeGraphTitle(
            participant,
            activityLabels.get(chosenActivity),
            labels[i],
            deviceID.get(chosenSensorId)
        )
        statistic, measured_pvalue = ksTest(normalizeCurve(sensorData[i]))
        print(
            title +
            f": mean={np.mean(sensorData[i])}, statistic={statistic}, pvalue={measured_pvalue}"
        )
        if measured_pvalue <= default_pvalue:
            rejected += 1
        else:
            accepted += 1
            fig, axs = plt.subplots(2)
            plotDistribution(
                normalizeCurve(sensorData[i]),
                axs[0],
                title
            )
            plotCdfFit(
                axs[1],
                np.sort(normalizeCurve(sensorData[i])),
                np.linspace(-5, 5, 100),  # settings for 'norm' distribution
                title
            )
            plt.subplots_adjust(hspace=0.50)
    plt.show()
    print(
        title +
        f": accepted={accepted}, rejected={rejected}"
    )
"""

# Questão 4.2 --
# Note: uncomment code from this
# question to generate dataset

# variables --
"""
chosenSensorId = 1
chosenParticipant = 0
debug = False

sFreq = 51.2  # as described in dataset
windowDuration = 2  # seconds
windowSize = round(windowDuration * sFreq)
overlap = windowSize // 2  # 50% overlap as described in https://bit.ly/tcd-paper
"""
# ------------

"""
data = extractPartData(dirParts, chosenParticipant)
sensorData = data[data[:, 0] == chosenSensorId]

# store array of windows for each activity
windows = getWindows(
    {k: [] for k in activityLabels.keys()},
    windowSize,
    overlap,
    sensorData
)

for i in range(1, 17):
    print(len(windows[i]))
for w in windows[13]:
    accStats = getStatFeat(stats, w, 1, 3)
    print(accStats)
"""

# df and energy are not
# statistical features but are
# computed for each axis, like
# the other metrics in this list.
"""
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
"""

# generate dataset
"""
alldata = []
for participant in range(maxPart):
    data = extractPartData(dirParts, participant)
    sensorData = data[data[:, 0] == chosenSensorId]
    # store array of windows for each activity
    windows = getWindows(
        {k: [] for k in activityLabels.keys()},
        windowSize,
        overlap,
        sensorData
    )
    for act, win in windows.items():
        for w in win:
            alldata.append(
                getWindowData(w, stats, phys)
                + [activityLabels.get(act)]  # append activity name in the end
            )

dataset = pd.DataFrame(alldata, columns=getColumns(labels, stats, phys))

# normalize dataset
for column in dataset:
    if column != 'act':
        dataset[column] = normalize(dataset[column])

# save to csv for performance
if not path.exists(dirOutput):
    mkdir(dirOutput)
filepath = path.join(dirOutput, f'dev{chosenSensorId}.csv')
if not path.exists(filepath):
    dataset.to_csv(
        filepath,
        index=False
    )
"""

# -- variables
# chosenSensorId = 2

# ------------

# read csv and plot graphics
# dataset = pd.read_csv(path.join(dirOutput, f'dev{chosenSensorId}.csv'))

# generate 16 RGB values
# colors = [hexCode() for i in range(len(activityLabels.values()))]

# stat feature comparison
"""
# for ft in ['mean', 'median', 'std', 'skew', 'kurtosis']:
for s in labels:
    comparisonPlot(
        dataset,
        'mean',
        colors,
        sensor=s
    )
"""

# phys feature comparison
"""
comparisonPlot(
    dataset,
    'acc_ai',
    colors,
    ftname2='acc_sma',
    mode='2d'
)
comparisonPlot(
    dataset,
    'gyro_ai',
    colors,
    ftname2='gyro_sma',
    mode='2d'
)
comparisonPlot(
    dataset,
    'mag_ai',
    colors,
    ftname2='mag_sma',
    mode='2d'
)
comparisonPlot(
    dataset,
    'evag',
    colors,
    ftname2='evah',
    mode='2d'
)
"""

# Questão 4.3 + 4.4
# [1] https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
# [2] https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# [3] https://youtu.be/kApPBm1YsqU

# -- variables
chosenSensorId = 2
dataset = pd.read_csv(path.join(dirOutput, f'dev{chosenSensorId}.csv'))
evr = 75
# ------------
"""
- Explained variance ratio has to be > 85% (sum of array).
If its lower than that, a lot of data is lost, and
it is not a valid analysis [3].
- Every pca can have at most min(n_samples, n_features) PC's
In our case, we have n_samples > n_features = 102, hence,
we can have at most 102 PC's.
"""
# prepare data
target = dataset['act']  # save activities column
dataset = dataset.drop(['act'], axis=1)

# In the code below, we apply pca
# with increasing pc's
# to find the best tradeoff
# between performance and information

# for at least 95% EVR, the # of PCs is 47+. We've
# halved the dimensionality of the problem
# while only losing 5% of the information.
# In our opinion, this is a great trade-off
# between performance and information
# For at least 75% EVR, the # of PCs is 22+.
evratios = getEvrs(dataset)
pcs = findPCs(evratios, evr + 1)
plotEvrPc(evratios, pcs)


# Questão 4.4.2
# The first component is responsible for most
# of the variance in the data, and each subsequent
# component is responsible for increasingly less
# variance.

# pros and use cases:
# - pca is useful when there's a lot of features/variables
# to consider
# - denoising and data compression
# - essential for data processing in smaller devices with
# less computing power, as this technique helps identify
# the most important variables, and exclude the rest.
# Less variables -> smaller problem space ->
# -> less computing power needed -> faster ML data processing.

# cons:
# - Loss of information: Whenever PCA is used, there's an inherent
# compromise between performance enhancement and data loss.
# - Data interpretability: Analyzing PCA's output can be difficult
# if you don't fully grasp its concept and the data structures
# involved.
# centroids, groups = kmeans2(device_data[:, :3], 3, 1)
# plotKmeans(device_data[:, :3], centroids, groups)


# 3.8 - Outlier Injection
"""

partData = extractPartData(dirParts, chosenParticipant)
actSensData = getActivityData(partData, chosenActivity,chosenSensorId)
sampleData = getVectorModule(actSensData,indexModule[strAcc])

#Devolve Tuplo com (indices dos outliers, data com outliers)
outliers = outliersInsertion(sampleData,5,3)

"""

# 3.9 - Linear modulus based on samples of p previous values

"""
sample = np.array([4,3,7,8,2,6,7,9,1,5,10,32,9])
p = 3
print(trainLinearModel(sample,len(sample)-p,p))

"""

# 3.10 e 3.11 - Linear Module applied to the moduli of each variable of a single activity

# Neste caso vamoos aplicar o LOOCV(Leave one out Cross Validation) -
# https://www.youtube.com/watch?v=fSytzGwwBVw&ab_channel=StatQuestwithJoshStarmer

chosenAct = 1
allData = getAllPartData(dirParts, maxPart)
ActData = getActivityData(allData, chosenAct, None)

# Modules of the different variables

AccModData = getVectorModule(ActData, indexModule[strAcc])
MagModData = getVectorModule(ActData, indexModule[strMag])
GirModData = getVectorModule(ActData, indexModule[strGir])

k = 3.5
perOut = 10
janelasVal = range(5, 20)
numPoints = 50

# --------------- ACCELERATION MODULUS  --------------------------

# inject outliers to replace them with the predictive linear model of each module
# return tuples (indicesOutliers, dataWithOutliers)

outliersInfoAcc = outliersInsertion(np.copy(AccModData), perOut, k)
indicesOutliersAcc = outliersInfoAcc[0]

# Test the models for different window sizes
# Returns the array of the sum of quadratic errors for each window of each model
# And the array of predicted values for each model along with the actual values

infArrAcc = testPVals(janelasVal, AccModData, indicesOutliersAcc)

arrMeanSquareModels = (infArrAcc[0], infArrAcc[1])
realPredValModelPrev = infArrAcc[2]
realPredValModelPrevNext = infArrAcc[3]

# Plots of the models in relation to the quadratic error per window
plotModelsQuadError(janelasVal, arrMeanSquareModels, strAcc)
plt.figure()
plt.subplot(2, 1, 1)

# Scatter of predicted and actual values for each model
scatterPlotRealPredicted(realPredValModelPrev[0][:numPoints], realPredValModelPrev[1][:numPoints],
                         strAcc, "Modelo Linear de p valores anteriores")
plt.subplot(2, 1, 2)
scatterPlotRealPredicted(realPredValModelPrevNext[0][:numPoints], realPredValModelPrevNext[1][:numPoints],
                         strAcc, "Modelo Linear p/2 valores anteriores e seguintes")

# --------------- MAGNOMETER MODULE --------------------------

outliersInfoMag = outliersInsertion(np.copy(MagModData), perOut, k)
indicesOutliersMag = outliersInfoMag[0]

infArrMag = testPVals(janelasVal, MagModData, indicesOutliersMag)

arrMeanSquareModels = (infArrMag[0], infArrMag[1])
realPredValModelPrev = infArrMag[2]
realPredValModelPrevNext = infArrMag[3]

plotModelsQuadError(janelasVal, arrMeanSquareModels, strMag)
plt.figure()
plt.subplot(2, 1, 1)

scatterPlotRealPredicted(realPredValModelPrev[0][:numPoints], realPredValModelPrev[1][:numPoints],
                         strMag, "Modelo Linear de p valores anteriores")
plt.subplot(2, 1, 2)
scatterPlotRealPredicted(realPredValModelPrevNext[0][:numPoints], realPredValModelPrevNext[1][:numPoints],
                         strMag, "Modelo Linear p/2 valores anteriores e seguintes")

# --------------- GYRO MODULE -------------------------

outliersInfoGir = outliersInsertion(np.copy(GirModData), perOut, k)
indicesOutliersGir = outliersInfoGir[0]

infArrGir = testPVals(janelasVal, GirModData, indicesOutliersGir)

arrMeanSquareModels = (infArrGir[0], infArrGir[1])
realPredValModelPrev = infArrGir[2]
realPredValModelPrevNext = infArrGir[3]

plotModelsQuadError(janelasVal, arrMeanSquareModels, strGir)

plt.figure()
plt.subplot(2, 1, 1)
scatterPlotRealPredicted(realPredValModelPrev[0][:numPoints], realPredValModelPrev[1][:numPoints],
                         strGir, "Modelo Linear de p valores anteriores")
plt.subplot(2, 1, 2)
scatterPlotRealPredicted(realPredValModelPrevNext[0][:numPoints], realPredValModelPrevNext[1][:numPoints],
                         strGir, "Modelo Linear p/2 valores anteriores e seguintes")

"""

     Conclusion on the comparison of the 2 models

        The linear model based on p prior values shows good results however
    the linear model based on p/2 prior and subsequent values shows much better results
    and with smaller windows.
    So if I wanted to predict values for missing data I would use the second model

"""

plt.show()

print("Done")
