from glob import glob
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import sklearn
import pandas as pd
import numpy as np
from scipy.stats import skew
import sklearn.model_selection
from sklearn.datasets import load_breast_cancer, load_diabetes


data_path = '../flask-server-revised/json_files'


def create_dataset(dataset):
    

    if dataset == 'iris':
        
        data = sklearn.datasets.load_iris()
        df = pd.DataFrame(data.data)
        df.columns = data['feature_names']


    elif dataset == 'wine':
        data = sklearn.datasets.load_wine()
        df = pd.DataFrame(data.data)
        df.columns = data['feature_names']

    elif dataset == 'abalone':

        df = pd.read_csv('/Users/navid/Documents/rasa_practice/XAIBot_V1/chat-app/flask-server-revised/abalone.csv')
        nf = df.select_dtypes(include=[np.number]).columns
        cf = df.select_dtypes(include=[np.object]).columns

        # sending all numericalfeatures and omitting nan values
        skew_list = skew(df[nf], nan_policy='omit')
        skew_list_df = pd.concat([pd.DataFrame(nf, columns=['Features']), pd.DataFrame(
            skew_list, columns=['Skewness'])], axis=1)
        mv_df = df.isnull().sum().sort_values(ascending=False)
        pmv_df = (mv_df/len(df)) * 100
        missing_df = pd.concat([mv_df, pmv_df], axis=1, keys=[
                                'Missing Values', '% Missing'])

        df['Age'] = df['Class_number_of_rings'] + 1.5
        df['Sex'] = LabelEncoder().fit_transform(df['Sex'].tolist())
        df = df.drop(['Class_number_of_rings'], axis=1)
    
    elif dataset == 'breast_cancer':
        data = load_breast_cancer()
        df = pd.DataFrame(data['data'], columns=data['feature_names'])

    elif dataset == 'diabetes':
        data = load_diabetes()
        df = pd.DataFrame(data['data'], columns=data['feature_names'])


    elif dataset == 'bike_sharing':
        bike_sharing = pd.read_csv('/Users/navid/Documents/rasa_practice/XAIBot_V1/chat-app/flask-server-revised/datasets/bike_sharing.csv')
        df = bike_sharing.drop(['casual', 'count'], axis=1)

        

    return df
