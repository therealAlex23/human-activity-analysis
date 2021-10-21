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
chosenParticipant = 0
chosenActivity = 1
chosenSensorId = 1
# ------------
data = extractPartData(dirParts, chosenParticipant)

# ignore activityId, sensorId and timestamp
device_data = getActivityData(data, chosenActivity, chosenSensorId)[:, 1:-2]

# acc data for chosenActivity only
# kmeans1(device_data[:, :3], 3, 10)
# https://www.youtube.com/watch?v=_aWzGGNrcic
centroids, groups = kmeans2(device_data[:, :3], 3, 1)
plotKmeans(device_data[:, :3], centroids, groups)

print("Done")
