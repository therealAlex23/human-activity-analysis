from scipy.stats.stats import zscore
from utils import *

# Globals
dirParts = "../assets/DatasetParts/part"
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


# Questao 3.1 e 3.2
# allData = getAllPartData(dirParts + "part", maxPart)
# getBoxPlotModuleActivity("cona", allData, activityLabels, 1, 2)

# Questao 3.3 e 3.4
# Escolhi o participante 0 --
chosenParticipant = 0
data = extractPartData(dirParts, 0)
fig, axs = plt.subplots(3)
plotOutliers(data, 3, 1, 1, axs)  # stand dev1
fig.suptitle(f"PART0 - STAND")
plt.show()

print("Done")
