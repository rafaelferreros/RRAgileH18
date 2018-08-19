import numpy as np
import pandas as pd
import glob
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#from matplotlib import pyplot as plot
from sklearn.ensemble import RandomForestRegressor

def train_data():
    data_file = "../dataset/hackaton_training_v1.csv"

    # v_0  = ID
    # v_1  = dias morosidad
    # v_2  = estabilidad laboral
    # v_3  = Corregimiento
    # v_4  = edad
    # v_5  = sexo
    # v_6  = salario
    # v_7  = otros ingresos
    # v_8  = score de originalizacion
    # v_9  = lugar de trabajo
    # v_10 = score de comportamiento de cobranza
    # v_11 = Provincia
    # v_12 = Ciudad
    data = pd.read_csv(data_file, usecols=['v_0', 'v_1', 'v_2', 'v_3', 'v_4',
                                           'v_5', 'v_6', 'v_7', 'v_8', 'v_9',
                                           'v_10', 'v_11', 'v_12'])


    # print("Resultados del banco:")
    # resultado_banco = data['v_8']
    # resultado_real = data['v_10']
    # error = abs(resultado_real - resultado_banco)
    # err = pd.DataFrame(error)
    # print(err.describe())
    #return
    #print(data.describe())
    resultado_banco = data['v_8']
    #print(resultado_banco.describe())

    # data_crazyines = data[(data.v_10 == 0)]
    # print(data_crazyines.head())
    # return

#    print(data[(data.v_9 >= 0)]['v_9'].describe())
#    return
    data_perfilada = data[(data.v_4 > 0) & (data.v_6 <= 30000) & (data.v_1 <= 1000) & (data.v_9 <= 600)] # & (data.v_9 >= 0)]
    #data_dropped = data_perfilada.drop(['v_0','v_1', 'v_3', 'v_5', 'v_7', 'v_8', 'v_9', 'v_11', 'v_12'], axis=1)
    data_dropped = data_perfilada.drop(['v_0','v_1', 'v_3', 'v_5', 'v_7', 'v_9', 'v_11', 'v_12'], axis=1)
    #print(data_dropped.head())

    # data_dropped.loc[data_dropped['v_9'] < 0, 'v_9'] = 85;
    # print(data_dropped.head())

    goal = np.array(data_dropped['v_10'])
    data_ready = data_dropped.drop('v_10', axis=1)

    data_ready_column_name = list(data_ready.columns)

    train_data, test_data, train_goal, test_goal = train_test_split(data_ready, goal, test_size = 0.4, random_state = 100)

    banco_guest = test_data['v_8']
    train_data = train_data.drop('v_8', axis=1)
    test_data = test_data.drop('v_8', axis=1)

    print('Training Data Shape:', train_data.shape)
    print('Training Goals Shape:', train_goal.shape)
    print('Testing Data Shape:', test_data.shape)
    print('Testing Goals Shape:', test_goal.shape)

    print("Resultados banco:")

    errors_banco = abs(banco_guest - test_goal)
    eb = pd.DataFrame(errors_banco)
    print(eb.describe())

    rf = RandomForestRegressor(n_estimators = 1000, random_state = 100)
    rf.fit(train_data, train_goal)

    predictions = rf.predict(test_data)

    errors = abs(predictions - test_goal)

    err = pd.DataFrame(errors)
    print("Resultados nuestros:")
    print(err.describe())
    #print('Errors: ', round(np.mean(errors), 2), 'degrees')

    #mape = 100 * (errors / test_goal)
    #accuracy = 100 - np.mean(mape)
    #print(accuracy)

    #plot.figure(figsize=(100,100))
    #print(test_goal.size)


    #plot.plot(range(0, test_goal.size), test_goal, 'gs')
    #plot.plot(range(0, test_goal.size), predictions, 'bs')
    #plot.plot(range(0, test_goal.size), banco_guest, 'rs')

    #plot.show()
#
    #errors = abs(predictions - test_goal)

    #print(errors)
    #print('Errors: ', round(np.mean(errors), 2), 'degrees')

    #print(data.query('v_5 == 2'))
    #prov = list(data.groupby(['v_12']).groups.keys())
    #print(data['v_12'].isin(prov) )
    #print(data[(data.v_11 == 56)])

    #print(data.loc[:, 'v_11'].describe())
    #print(data.loc[:,'v_5'==0])

    # size = data['v_9'].count()
    # # print('Size: ', size)

    # array_number = range(0, size)

    # count = pd.Series(array_number)

    # plot.figure(figsize=(10,10))
    # plot.plot(count, data['v_6'], 'o')
    # plot.xlabel('v_11')
    # plot.ylabel('v_6')

    # plot.show()

    # column = 'v_12'
    # zeros = 0

    # for i in range(0, data[column].count()):
    #     if data[column][i] <= 0:
    #         zeros+=1

    # print('Zeros found: ', zeros)

def gob_data():
    path = "./Data/gobdata/"
    all_files = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()

    for file_ in all_files:
        df = pd.read_csv(file_, index_col=0, header=0)
        frame = frame.append(df, ignore_index=True)

    #print(frame.iloc[:,4])
    #print(frame.index)
    #print(frame.loc[5,"Fecha de Inicio"].apply(pd.to_datetime))
    frame.loc[:,"Fecha de Inicio"] = frame.loc[:,"Fecha de Inicio"].apply(pd.to_datetime)
    print(frame.loc[5,"Fecha de Inicio"])
    #print(frame.loc[:,"Salario"].apply(pd.to_numeric))

def gob_data_2():
    path = "./Data/gobdata/Defensoria_del_Pueblo_1.csv"

    data = pd.read_csv(path)
    print(data.describe())

def DTR_data():
    data_file = "../dataset/hackaton_training_v1.csv"

    # v_0  = ID
    # v_1  = dias morosidad
    # v_2  = estabilidad laboral
    # v_3  = Corregimiento
    # v_4  = edad
    # v_5  = sexo
    # v_6  = salario
    # v_7  = otros ingresos
    # v_8  = score de originalizacion
    # v_9  = lugar de trabajo
    # v_10 = score de comportamiento de cobranza
    # v_11 = Provincia
    # v_12 = Ciudad
    data = pd.read_csv(data_file, usecols=['v_0', 'v_1', 'v_2', 'v_3', 'v_4',
                                           'v_5', 'v_6', 'v_7', 'v_8', 'v_9',
                                           'v_10', 'v_11', 'v_12'])

    resultado_banco = data['v_8']

    data_chopped = data[(data.v_4 > 0) & (data.v_6 <= 30000) & (data.v_1 <= 1000) & (data.v_9 <= 600)]
    #data_dropped = data_chopped.drop(['v_0','v_1', 'v_3', 'v_5', 'v_7', 'v_8', 'v_9', 'v_11', 'v_12'], axis=1)
    data_dropped = data_chopped.drop(['v_0','v_1', 'v_3', 'v_5', 'v_7', 'v_9', 'v_11', 'v_12'], axis=1)
    print(data_dropped.head())

    # data_dropped.loc[data_dropped['v_9'] < 0, 'v_9'] = 85;
    # print(data_dropped.head())

    goal = np.array(data_dropped['v_10'])
    data_ready = data_dropped.drop('v_10', axis=1)

    data_ready_column_name = list(data_ready.columns)

    train_data, test_data, train_goal, test_goal = train_test_split(data_ready, goal, test_size = 0.4, random_state = 100)

    banco_guest = test_data['v_8']
    train_data = train_data.drop('v_8', axis=1)
    test_data = test_data.drop('v_8', axis=1)

    print('Training Data Shape:', train_data.shape)
    print('Training Goals Shape:', train_goal.shape)
    print('Testing Data Shape:', test_data.shape)
    print('Testing Goals Shape:', test_goal.shape)

    print("Resultados banco:")
    errors_banco = abs(banco_guest - test_goal)
    eb = pd.DataFrame(errors_banco)
    print(eb.describe())

    reg = DecisionTreeRegressor(max_depth=8)

    reg.fit(train_data, train_goal)

    predictions = reg.predict(test_data)

    errors = abs(predictions - test_goal)

    err = pd.DataFrame(errors)
    print("Resultados nuestros:")
    print(err.describe())

if __name__ == '__main__':
    print("--------------With Random Forest---------------")
    train_data()
    print("--------------With Decision Tree---------------")
    DTR_data()
    f = open("../result/output.txt",'w')
    print("Hola paco como estas", file=f) # Python 3.x