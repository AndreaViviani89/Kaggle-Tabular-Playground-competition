import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


from feat_eng import int_data, float_data, new_df

df = pd.read_csv('data/train.csv').drop(['id'], axis = 1)

# print(df.shape)
# print(df.head)

#Check the columns
# print(df.columns)

cat_var = [int_data]
num_var = [float_data]

# print(cat_var)
print(num_var)


# Pipeline
num_preprocessing = Pipeline( [('imp', SimpleImputer(fill_value= -999, strategy='constant')) ] )
cat_preporcessing = Pipeline( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -999)),
                                ('imp', SimpleImputer(strategy='constant', fill_value= -999))] )


tree_prepr = ColumnTransformer( [('num', num_preprocessing, num_var), ('cat', cat_preporcessing, cat_var)], remainder = 'passthrough') 


models_dic = {
                      "Decision Tree": DecisionTreeClassifier(random_state=0),
                      "Extra Trees": ExtraTreesClassifier(random_state=0),
                      "AdaBoost": AdaBoostClassifier(random_state=0),
                      "Skl GBM": GradientBoostingClassifier(random_state=0),
                      "Random Forest": RandomForestClassifier(random_state=0)
                      }

models_dic = {name: make_pipeline(tree_prepr, model) for name, model in models_dic.items()}


X = new_df.drop(['target'], axis = 1)
y = new_df['target']

# Train test split
def test_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 0, stratify= y )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = test_split(X, y)