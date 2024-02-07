import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score
import warnings
import ast
import re

def number_df(df):
    num_df = df.select_dtypes(include='number')

    num_df['total_num_orders'] = num_df['order_num_total_ever_online'] + num_df['order_num_total_ever_offline']
    num_df['customer_total_value'] = num_df['customer_value_total_ever_offline'] + num_df['customer_value_total_ever_online']
    num_df = num_df.drop(['order_num_total_ever_online', 'order_num_total_ever_offline', 'customer_value_total_ever_offline',   'customer_value_total_ever_online'], axis=1)

    return num_df

def date_df_converter(df):
    date_df = df[['first_order_date', 'last_order_date', 'last_order_date_online', 'last_order_date_offline']]
    for i in date_df.keys():
        date_df[i] = pd.to_datetime(date_df[i])
    
    # Total customer lifetime (offline?)
    date_df['total_lifetime_days'] = pd.Timestamp.now() - date_df['first_order_date']
    date_df['total_lifetime_days'] = date_df['total_lifetime_days'].dt.days

    date_df['time_from_last_order_days'] = pd.Timestamp.now() - date_df['last_order_date']
    date_df['time_from_last_order_days'] = date_df['time_from_last_order_days'].dt.days
    date_df['active_days'] = date_df['last_order_date'] - date_df['first_order_date']
    date_df['active_days'] = date_df['active_days'].dt.days
    date_df['order_freq_per_month'] = number_df(df)['total_num_orders'] / (date_df['active_days'] / 30)
    date_df['order_freq_per_month'].fillna(0, inplace=True)  # Replace NaN with 0
    date_df['order_freq_per_month'] = date_df['order_freq_per_month'].replace([np.inf, -np.inf], np.nan).fillna(0)  

    # Convert to integer
    date_df['order_freq_per_month'] = date_df['order_freq_per_month'].astype(float)
    
    return date_df

def cat_encoding(df):
    cat_df = df['order_channel']
    
    # One Hot Encoding 
    cat_df = pd.get_dummies(cat_df)
    cat_df = cat_df.astype(int)
    
    # Editing column names
    cat_df.rename(columns=lambda i: i.lower().replace(" ", "_"), inplace=True)
    
    return cat_df
    

def dummies(df):
    df_intrest = df['interested_in_categories_12']
    df_intrest = df_intrest.apply(lambda x: x.strip('][').split(', '))

        # Items translation dict
    translations = {
        'AKTIFSPOR': 'SPORTS',
        'KADIN': 'WOMEN',
        'ERKEK': 'MEN',
        'COCUK': 'CHILDREN',
        'AKTIFCOCUK': 'CHILDREN_SPORTS',
        '':''
    }

    #Translate_items
    df_intrest = df_intrest.apply(lambda x: [translations[item] for item in x])
    # Display the after translation
    df_intrest.value_counts()
    
    df_encoded = df_intrest.str.join(sep='*').str.get_dummies(sep='*')
    df_encoded.rename(columns=lambda i: i.lower(), inplace=True)
    
    return df_encoded


# def combined_df(number_df(df), date_df(df), cat_encoding(df), dummies(df)):
#     combined_df = pd.concat([num_df, date_df, df_encoded, cat_df], axis=1)
    
#     return combined_df


############################################################################################
# Reuire extra setup 
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.1)
    quartile3 = dataframe[variable].quantile(0.9)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)
    
############################################################################################

# columns = ["total_num_orders",
#            "customer_total_value",
#            "total_lifetime_days",
#            "time_from_last_order_days"]
# for col in columns:
#     replace_with_thresholds(combined_df, col)
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    

