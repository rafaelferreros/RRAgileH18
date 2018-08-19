import pandas as dp
import os, fnmatch

dp.options.display.max_rows = 50000
dp.options.display.max_columns = 10
dp.options.display.width = 1000

gobDir = "../dataset/gobdata/"
datasets = []

listOfFiles = os.listdir(gobDir)
for dataFile in listOfFiles:

    full_path = gobDir+dataFile;
    print(full_path)
    #usecols = ['Nombre', 'Apellido', 'Cedula', 'Cargo', 'Salario', 'Gasto', 'Estado', 'Fecha de Inicio']
    datasets.append(dp.read_csv(full_path, sep=",",
                                usecols=['Cedula','Salario','Fecha de Inicio']))


result = dp.concat(datasets)

print(result)
