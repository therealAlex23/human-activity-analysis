
randomState = 42
figSizeGen = (15,10)

strAcc, strGir, strMag = "Aceleração", "Giroscopio", "Magnometro"

dirParts = "../assets/part"
dirOutput = "../outputs"
dirPlots = "../Plots"

dirDummyClf = "/Dummy Classifier" 
dirOneRClf = "/OneR Classifier"

dirParamAnalysis = "/Parameter Analysis"
dirKTest = "/K Value Analysis"
dirTrainOnly = "/Train Only"
dirTT = "/Train-Test"
dirKFoldCV = "/K Fold 10CV"
dirFolds = "/Folds"

dir4_5A = dirPlots + "/4.5CompA" #Perfect for analysis
dir2_1 = dirPlots +"/2.1"
dir2_2_1 = dirPlots + "/2.2.1"
dir2_2_2 = dirPlots + "/2.2.2"
dir2_3 = dirPlots + "/2.3"
dir2_4 = dirPlots + "/2.4"
dir2_5 = dirPlots + "/2.5"
dir3_1 = dirPlots + "/3.1"
dir3_2_1 = dirPlots + "/3.2.1"
dir3_2_2 = dirPlots + "/3.2.2"
dir3_3 = dirPlots +"/3.3"
dir4_1 = dirPlots + "/4.1"
dir4_2 = dirPlots + "/4.2"
dir4_3 = dirPlots + "/4.3"

dirList = [dir2_1,dir2_2_1,dir2_2_2,dir2_3,dir2_4,dir2_5,dir3_1,dir3_2_1,dir3_2_2,dir3_3,dir4_1,dir4_2,dir4_3]

#Put the flag at True if you want to run it

flag4_5A = False #Perfect for analysis
flag2_1 = False #Perfect for analysis
flag2_2_1 = False #Perfect for analysis
flag2_2_2 = False #Perfect for analysis
flag2_3 = False #Perfect for analysis
flag2_4 = False #Perfect for analysis
flag2_5 = False #Perfect for analysis
flag3_1  = False #Bigger picture and titles with bold font
flag3_2_1 = False
flag3_2_2 = False 
flag3_3 = False
flag4_1 = False
flag4_2 = False
flag4_3 = False

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