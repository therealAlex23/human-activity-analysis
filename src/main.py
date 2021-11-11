from utils import *
from os import path, mkdir
import pandas as pd

# Globals
dirParts = "../assets/part"
dirOutput = "../outputs"
maxPart = 15
noOfSensors = 5

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

# ignore activityId, sensorId and timestamp
device_data = getActivityData(data, chosenActivity, chosenSensorId)[:, 1:-2]

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

print("Done")
