import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# function to produce pairplot on features
def plot_features(df, sample_size=500):
    """
    Takes in a dataframe and sample size, returns pairplot of features
    """    
    sample = (df.drop(['product_id', 'user_id', 'latest_cart'], axis=1)
                .sample(1000, random_state=44)) 
    sns.pairplot(sample, hue='in_cart', plot_kws=dict(alpha=0.3, edgecolor='none')) 

# function to split train/val/test based on users
def get_user_split_data(df, test_val_size=0.4, seed=42): 
    """
    Takes in a dataframe, test/val size and seed, returns train, validation and test for X and y
    """
    rs = np.random.RandomState(seed)
    
    total_users = df['user_id'].unique() 
    test_val_size = int(total_users.shape[0] * test_val_size) # size of test / val sample
    test_val_users = rs.choice(total_users, 
                               size=(2, test_val_size), # array of two arrays, 1 for val and 1 for test
                               replace=False)

    df_train = df[~df['user_id'].isin(test_val_users.flatten())]
    df_test = df[df['user_id'].isin(test_val_users[0])]
    df_val = df[df['user_id'].isin(test_val_users[1])]

    y_train, y_test, y_val = df_train['in_cart'], df_test['in_cart'], df_val['in_cart']
    X_train = df_train.drop(['product_id','user_id','latest_cart','in_cart'], axis=1) 
    X_test = df_test.drop(['product_id','user_id','latest_cart','in_cart'], axis=1)
    X_val = df_val.drop(['product_id','user_id','latest_cart','in_cart'], axis=1)  

    return X_train, X_val, X_test, y_train, y_val, y_test

# function to scale and transform newly engineered features 
def scale_transform(features, X_train, X_val, X_test):#X_train_cols, X_val_cols, X_test_cols):
    """
    Takes in features of df_X to scale, X train, X val and X test, returns scaled X train, val and test
    """
    scaler = StandardScaler()

    scaler.fit(X_train[features]) 
    X_train[features] = scaler.transform(X_train[features])
    X_val[features] = scaler.transform(X_val[features]) 
    X_test[features] = scaler.transform(X_test[features])
    
    # if X_train_cols.shape[1] > 1:
    #     scaler.fit(X_train_cols)
    #     X_train_cols = scaler.transform(X_train_cols)
    #     X_val_cols = scaler.transform(X_val_cols)
    #     X_test_cols = scaler.transform(X_test_cols)
    
    # else:
    #     scaler.fit(X_train_cols)
    #     X_train_cols = scaler.transform(X_train_cols)
    #     X_val_cols = scaler.transform(X_val_cols)
    #     X_test_cols = scaler.transform(X_test_cols)    
    
    # else:
    #     scaler.fit(X_train_cols.reshape(-1,1))
    #     X_train_cols = scaler.transform(X_train_cols.reshape(-1,1))
    #     X_val_cols = scaler.transform(X_val_cols.reshape(-1,1))
    #     X_test_cols = scaler.transform(X_test_cols.reshape(-1,1))
    
    return X_train, X_val, X_test