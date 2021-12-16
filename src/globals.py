
randomState = 42

strAcc, strGir, strMag = "Aceleração", "Giroscopio", "Magnometro"

dirParts = "../assets/part"
dirOutput = "../outputs"
dirPlots = "../Plots"

figSizeGen = (18,15)

dir2_2_1 = dirPlots + "/2.2.1"
dir2_2_2 = dirPlots + "/2.2.2"
dir2_3 = dirPlots + "/2.3"
dir3_2_1 = dirPlots + "/3.2.1"
dir3_2_2 = dirPlots + "/3.2.2"
dir3_3 = dirPlots +"/3.3"
dir4_1 = dirPlots + "/4.1"
dir4_2 = dirPlots + "/4.2"
dir4_3 = dirPlots + "/4.3"

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

strategiesDumClf = ["stratified", "most_frequent", "prior", "uniform"]
distMetrics = ["euclidean","manhattan","chebyshev","minkowski"]