from utils import *

# Globals
dirParts = "../assets/DatasetParts/"
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


# Questao 3.1
"""
for i in range(maxPart):
    print("Participant " + str(i))
    data = extractPartData(dirParts + "part", i)
    getActivityMod(data, 1, 1)  # 1 = Stand
    getActivityMod(data, 1, 2)  # 2 = Sit
    getActivityMod(data, 1, 3)  # 3 = Sit&Talk
    getActivityMod(data, 1, 4)  # etc...

    # accModule = getSensorModuleArray(data, 1)
    # gyroModule = getSensorModuleArray(data, 4)
    # magModule = getSensorModuleArray(data, 7)

    
    print("AccModule", accModule)
    print("Gyro Module", gyroModule)
    print("Mag Module", magModule)

    drawBoxPlot(accModule)
"""
allData = getAllPartData(dirParts + "part", maxPart)
for act in activityLabels.keys():

    accMod = getActivityMod(allData, 1, act)
    gyroMod = getActivityMod(allData, 4, act)
    magMod = getActivityMod(allData, 7, act)
    print("ACTIVITY -----> " + activityLabels[act]+"\n\n")
    print("AccModule: ", accMod)
    print("Gyro Module: ", gyroMod)
    print("Mag Module: ", magMod)

    # drawBoxPlot(accMod)


print("Done")
