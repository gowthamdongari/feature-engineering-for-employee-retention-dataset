from sklearn import preprocessing as pp
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt



def csv_to_dataframe(filepath):
    dataframe = pd.read_csv(filepath)
    return dataframe


def dtype_category(dataframe, column_list):


    for col_name in column_list:

        dataframe[col_name] = dataframe[col_name].astype("category")

    return dataframe

def centre_and_scale(dataframe, column_list):

    try:
        for col_name in column_list:
            dataframe[col_name] = pp.scale(dataframe[col_name], copy=False)
    except KeyError:
        raise 'Col does not exist'
    return dataframe

def label_encoder(dataframe, column_list):
    try:
        for col_name in column_list:
            le = pp.LabelEncoder()
            dataframe[col_name] = le.fit_transform(dataframe[col_name])

    except KeyError:
        raise 'column is not categorical or not exists'
    return dataframe


def one_hot_encoder(dataframe, column_list):
    try:
        return pd.get_dummies(dataframe, columns = column_list)
    except KeyError:
        raise 'column is not categorical or not exists'


def skewness(dataframe, column_list):
    try:
        list_of_skew = []
        for col_name in column_list:
            skew_data = dataframe[col_name].skew()
            list_of_skew.append(skew_data)
    except KeyError:
        raise 'column is not categorical or not exists'
    return list_of_skew


def sqrt_transform(dataframe, column_list):
    try:
        list_of_sqrt = []
        for col_name in column_list:
            sqrt_num = np.sqrt(dataframe[col_name])
            list_of_sqrt.append(sqrt_num)
    except KeyError:
        raise 'column is not categorical or not exists'
    return list_of_sqrt


def plots(dataframe, column_list):

    transformed_list = sqrt_transform(dataframe)
    transformed = pd.DataFrame(data=transformed_list).T

    for col in column_list:

        plt.figure(figsize=(20,10))
        plt.subplot(221)
        plt.title('Original distribution : {}'.format(col))
        sns.distplot(dataframe[col], fit=norm, kde=False)

        plt.subplot(222)
        plt.title('Transformed distribution : {}'.format(col))
        sns.distplot(transformed[col], fit=norm, kde=False)

        plt.subplot(223)
        plt.title('Original boxplot : {}'.format(col))
        sns.boxplot(x=col, data=dataframe)

        plt.subplot(224)
        plt.title('Transformed boxplot : {}'.format(col))
        sns.boxplot(x=col, data=transformed)
        plt.show()
