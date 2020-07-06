"""
Predicting Instacart product re-orders --
Script for data preprocessing
Used to create aggregated df / pickles to be used for model creation
This script is designed to be run on a remote cloud instance where postgresql database is 
set up and stored with Instacart data, 
either through a Jupyter notebook or the command line
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import patsy

import psycopg2 as pg
import pandas.io.sql as pd_sql

from helper_functions import plot_features, get_user_split_data, scale_transform

### Connect to database
# establish connection to instacart database
connection_args = {
    'host': 'localhost',  # enter PUBLIC IP address of AWS instance here - '3.16.165.24'
    #'user': 'ubuntu',    # enter username on AWS - 'ubuntu'
    'dbname': 'instacart',    # DB that we are connecting to
    'port': 5432,             # port we opened on AWS
}

connection = pg.connect(**connection_args)

# run below queries from instacart db and store output into dataframes
# query from order products prior view (joined on order products prior and orders table)
query_order_products_prior = """
    SELECT * FROM order_products_prior_sample_view 
""" # order_products_prior_view 

# query from order products train view (joined on order products train and orders table)
query_order_products_train = """
    SELECT * FROM order_products_train_sample_view 
""" # order_products_train_view

# query from products table
query_products = """
    SELECT * FROM products
"""

# query from departments table
query_departments = """
    SELECT * FROM departments
"""

# query from aisles table
query_aisles = """
    SELECT * FROM aisles 
"""

df_order_products_prior = pd_sql.read_sql(query_order_products_prior, connection)
df_order_products_train = pd_sql.read_sql(query_order_products_train, connection)
df_products = pd_sql.read_sql(query_products, connection)
df_departments = pd_sql.read_sql(query_departments, connection)
df_aisles = pd_sql.read_sql(query_aisles, connection)

# create a user-product aggregated version of the order_products_prior data, aggregating on total # of times a user ordered a product
df_user_product = (df_order_products_prior.groupby(['product_id','user_id'], as_index=False) 
                                          .agg({'order_id':'count'}) 
                                          .rename(columns={'order_id':'user_product_total_orders'}))

train_ids = df_order_products_train['user_id'].unique() 
df_X = df_user_product[df_user_product['user_id'].isin(train_ids)] #this excludes test users from kaggle

# add new column as the set of products that exist in the user's latest cart
train_carts = (df_order_products_train.groupby('user_id',as_index=False)
                                      .agg({'product_id':(lambda x: set(x))})
                                      .rename(columns={'product_id':'latest_cart'}))

df_X = df_X.merge(train_carts, on='user_id')

# create in cart as binary variable, 1 as product in cart and 0 as product not in cart
df_X['in_cart'] = (df_X.apply(lambda row: row['product_id'] in row['latest_cart'], axis=1).astype(int))

# pickle dataframes
df_X.to_pickle('data/df_X.pkl')
df_order_products_prior.to_pickle('data/df_order_products_prior.pkl')
df_order_products_train.to_pickle('data/df_order_products_train.pkl')
df_products.to_pickle('data/df_products.pkl') 
df_departments.to_pickle('data/df_departments.pkl') 
df_aisles.to_pickle('data/df_aisles.pkl')