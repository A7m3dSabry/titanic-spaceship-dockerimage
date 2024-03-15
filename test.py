import json 
import pickle as pkl 
from sklearn import *
import numpy as np
import pandas as pd
import sys


path = ''
model = 0
model_path = './model.pkl'


def loadData(path):
    return pd.read_csv(path)
    
#### helper functions #####

def get_dummies(df,column):
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    return df

def change_type_to_bool(df,column):
    df[column] = df[column].astype('bool')
    return df
    
    
    

 
#### end of helper functions ####

def prepareData(df):
    df.columns = df.columns.str.lower()
    df = df.dropna()
    if 'name' in df.columns:
        df.drop(columns=['name'],inplace=True)        
    df[['deck','cabin_num','side']]=df['cabin'].str.split('/',expand=True)
    df[['passenger_number','passenger_group']]=df['passengerid'].str.split('_',expand=True)
    df.drop(columns=['cabin','cabin_num','passengerid','passenger_number'],inplace=True)
    df= get_dummies(df,'homeplanet')
    df= get_dummies(df,'destination')
    df= get_dummies(df,'deck')
    df= get_dummies(df,'passenger_group')
    df.drop(columns=['homeplanet','destination','deck','passenger_group'],inplace=True)
    df['side']= df.side.apply(lambda x: 1 if (x=='P') else 0)
    df= change_type_to_bool(df,'vip')
    df= change_type_to_bool(df,'cryosleep')
    if ('transported' in df.columns):
        df= change_type_to_bool(df,'transported')
        pass
    return df



    
def test_model(data):
    # load model
    model = pkl.load(open('model.pkl', 'rb'))
    
    # split data
    if 'transported' in data:
        y_test = data['transported']
        x_test = data.drop(columns=['transported'])
    else:
        x_test = data
    acc=0
    # test
    y_pred = model.predict(x_test)
    
    if 'transported' in data:
       a = (y_pred == y_test)
       acc = a.sum()/a.size
    return y_pred,acc


def writePrediction(path,pred):
    # write predection in txt
    with open(path,'w') as f:
        f.write(str(pred))




# load and prepare data

dataPath = sys.argv[1]

    #read data from file
inputData = loadData(dataPath)
    # process data
testData = prepareData(inputData)


# predict data
y_out,acc = test_model(testData)

# extract folder path

folderpath = dataPath[0 : dataPath.rfind('\\') + 1 ]
# writePrediction
writePrediction(f'{folderpath}test.txt', acc)
