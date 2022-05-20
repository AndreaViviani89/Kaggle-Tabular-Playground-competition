from matplotlib import axis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/train.csv').drop(['id'], axis = 1)
# print(df.shape)
# print(df.head)
# print(df.info)

#Check the columns
# print(df.columns)


#check for unique values
# for col in df:
#     print(col)
#     print(df[col].unique())


# Try to divide the float features
def float_data(float_features):
    float_features = [f for f in df.columns if df[f].dtype == 'float64']
    return float_features
# print(float_features)


# fig, axs = plt.subplots(4, 4, figsize=(16, 16))
# for f, ax in zip(float_features, axs.ravel()):
#     ax.hist(df[f], density=True, bins=100)
#     ax.set_title(f'Train {f}, std={df[f].std():.1f}')
# plt.suptitle('Histograms of the float features', y=0.93, fontsize=20)
# plt.show()

# Try to divide the int features
def int_data(int_features):
    int_features = [i for i in df.columns if df[i].dtype == 'int64']
    return int_features

# print(int_features)


# Create a function, the main goal is split f_27 out by each of the 10 letters. 
le = LabelEncoder()

def clean(data):
    
    data = df
    
    data['a1'] = data['f_27'].astype(str).str[0]
    data['a2'] = data['f_27'].astype(str).str[1]
    data['a3'] = data['f_27'].astype(str).str[2]
    data['a4'] = data['f_27'].astype(str).str[3]
    data['a5'] = data['f_27'].astype(str).str[4]
    data['a6'] = data['f_27'].astype(str).str[5]
    data['a7'] = data['f_27'].astype(str).str[6]
    data['a8'] = data['f_27'].astype(str).str[7]
    data['a9'] = data['f_27'].astype(str).str[8]
    data['a10'] = data['f_27'].astype(str).str[9]
 

    #map LabelEncoder for the new f_27 
    
    cols_to_map = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10']
    for col in cols_to_map:
        data[col] = le.fit_transform(data[col])

        
    data = data.drop(['f_27'], axis = 1)

    return data

# print(clean(df))    #[900000 rows x 41 columns] Now I've 44 columns



new_df = clean(df)

X = new_df.drop(['target'], axis = 1)
# print(X.shape)              # (900000, 40)
y = new_df['target']
# print(y.shape)                  # (900000,)

# Train test split
def test_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 0, stratify= y )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = test_split(X, y)

# print(X_train.shape)                    #(630000, 40)
# print(X_test.shape)                     # (270000, 40)
# print(y_train.shape)                    # (630000,)
# print(y_test.shape)                     # (270000,)
