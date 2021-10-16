import csv
import numpy as np

def extractPartData(dir,numPart):
    fullDir = dir + str(numPart) + "/part"+str(numPart) +"dev"
    partData = np.genfromtxt(fullDir+ "1.csv",delimiter=',')
    for sens in range(2,6):
        sensData = np.genfromtxt(fullDir+str(sens)+ ".csv",delimiter=',')
        partData = np.append(partData,sensData,axis=0)
    return partData