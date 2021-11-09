from utils import *

# Globals
dirParts = "../assets/part"
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

labels = ["ACC", "GYRO", "MAG"]

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

# Questão 4.2
# variables --
chosenParticipant = 0
chosenSensorId = 4
debug = False

sFreq = 51.2  # as described in dataset
windowDuration = 2  # seconds
windowSize = round(windowDuration * sFreq)
overlap = windowSize // 2  # 50% overlap as described in https://bit.ly/tcd-paper
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

alldata = []

for participant in range(4, 5):
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
print(dataset)

# print(dataset)
# dataset = makeParticipantDf(dataset)  # , columns)

# -- todo: passar as features
# -- para uma df de pandas

print("Done")
