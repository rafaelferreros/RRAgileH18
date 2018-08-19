import pandas as dp
import os, fnmatch


gobDir = "../dataset/gobdata"


listOfFiles = os.listdir(gobDir)
for entry in listOfFiles:

    print(entry)