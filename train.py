import pickle as pkl 
from sklearn import *
import numpy as np
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle as pkl

def saveModel(model, filename='titanic_model.pkl'):
    # save the model to disk
    with open(filename, 'wb') as file:
        pkl.dump(model, file)
 
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
 
 

def trainModel(X, y):
    # Create an instance of Logistic Regression Classifier and fit the data.
    logreg = LogisticRegression()
    logreg.fit(X, y)

    # Predict the labels of the training data
    y_pred = logreg.predict(X)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y, y_pred)

    return logreg, accuracy
 
def writePrediction(path,pred):
    # write predection in txt
    with open(path,'w') as f:
        f.write(str(pred))
 
 
 
 
# load and prepare data
 
dataPath = sys.argv[1]
 
inputData = loadData(dataPath)
 
testData = prepareData(inputData)
 
model ,acc = trainModel(testData.drop(columns=['transported']), testData['transported'])

folderpath = dataPath[0 : dataPath.rfind('\\') + 1 ]

writePrediction(f'{folderpath}train.txt', acc)

saveModel(model, 'model.pkl')