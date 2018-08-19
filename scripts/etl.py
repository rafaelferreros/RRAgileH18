import pandas as dp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import algo

dataset = "../dataset/hackaton_training_v1.csv"

dp.options.display.max_rows = 5000
dp.options.display.max_columns = 50
dp.options.display.width = 1000

data = dp.read_csv(dataset, sep=",", usecols = [2,3,5,7,9,11])

outcome_var = 'v_10'
#model = LogisticRegression()
model = DecisionTreeClassifier()
predictor_var = ['v_1','v_2','v_4','v_6']
algo.classification_model(model, data, predictor_var, outcome_var)


#data = dp.read_csv(dataset, sep=",", usecols = [1,2,3,4,5,6,7,8,9,10,11,12,13])
#returns = data[[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64']]].pct_change()
#plot.figure(figsize=(10,10))
#plot.plot(data['v_10'])
#plot.show()
#corr = data.corr()
#corr.style.background_gradient()
#f = open("output.txt",'w')
#print(corr, file=f) # Python 3.x
#print(returns)