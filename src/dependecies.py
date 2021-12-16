from numpy.lib.type_check import real
from mpl_toolkits import mplot3d
import math
import matplotlib
from matplotlib.pyplot import cla, ylabel
from matplotlib.text import OffsetFrom
from matplotlib import legend
import matplotlib.gridspec as gridspec
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, fft, signal
#from plotly.express import scatter_3d, scatter
from sklearn import datasets
from sklearn.decomposition import PCA
from ReliefF import ReliefF

from scipy.sparse import data
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score,recall_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from skfeature.function.similarity_based.reliefF import feature_ranking, reliefF
from itertools import combinations
from copy import deepcopy

import os
from os import access
from numpy import printoptions
from numpy.core.numeric import outer
from numpy.lib.function_base import select
from scipy.stats import zscore
from scipy.stats.stats import kendalltau
from sklearn import neighbors
from sklearn.utils import axis0_safe_slice
from utilsCompB import *
from os import path, mkdir
import pandas as pd
import seaborn as sb

from skfeature.function.similarity_based import reliefF as rf

import warnings
warnings.filterwarnings('ignore')
