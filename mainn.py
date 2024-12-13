import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from xgboost import XGBRegressor


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
df_train = pd.read_csv("dataset/train.csv")
df_test = pd.read_csv("dataset/test.csv")
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df_train = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df_train, end="\n")

    if na_name:
        return na_columns


print("#########################Train missing table########################")
missing_values_table(df_train)
print("#########################Test missing table########################")
missing_values_table(df_test)

drop_list = ['Id','PoolQC','MiscFeature','Alley','Fence','MasVnrType']
df_train.drop(drop_list,axis=1,inplace=True)
df_test.drop(drop_list,axis=1,inplace=True)
print("ok")
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols,num_cols,cat_but_car = grab_col_names(df_train)
cat_cols= cat_cols+cat_but_car

na_cols_test = missing_values_table(df_test,na_name=True)
na_cols_train = missing_values_table(df_train,na_name=True)
def fill_missing_values(dataframe,na_col):
    if na_col in cat_cols:
        dataframe[na_col].fillna(dataframe[na_col].mode()[0],inplace=True)
    else:
        dataframe[na_col].fillna(dataframe[na_col].median(),inplace=True)
    return dataframe

for col in na_cols_train:
    fill_missing_values(df_train,col)

for col in na_cols_test:
    fill_missing_values(df_test,col)

main_df = pd.concat([df_train,df_test])
# rare encoding step
cat_cols,num_cols,cat_but_car = grab_col_names(main_df)
main_df.drop(['Neighborhood'],axis=1,inplace=True)
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df_train = dataframe.copy()

    rare_columns = [col for col in temp_df_train.columns if temp_df_train[col].dtypes == 'O'
                    and (temp_df_train[col].value_counts() / len(temp_df_train) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df_train[var].value_counts() / len(temp_df_train)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df_train[var] = np.where(temp_df_train[var].isin(rare_labels), 'Rare', temp_df_train[var])

    return temp_df_train

rare_encoder(main_df,0.01)
rare_analyser(main_df, "SalePrice", cat_cols)


cat_cols, cat_but_car, num_cols = grab_col_names(main_df)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in main_df.columns if main_df[col].dtypes == "O" and len(main_df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(main_df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

main_df = one_hot_encoder(main_df, cat_cols, drop_first=True)
bool_columns = main_df.select_dtypes(include='bool').columns
main_df[bool_columns] = main_df[bool_columns].astype(int)
main_df =main_df.loc[:,~main_df.columns.duplicated()]
df_Train=main_df.iloc[:1460,:]
df_Test=main_df.iloc[1460:,:]
df_Test.drop(['SalePrice'],axis=1,inplace=True)
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=np.log1p(df_Train['SalePrice'])
X_test = df_test
model = XGBRegressor()

n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
learning_rate=[0.1,0.2]
min_child_weight=[2,4]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    }

xgboost_best_gs = GridSearchCV(model,
                            hyperparameter_grid,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

final_model = model.set_params(**xgboost_best_gs.best_params_).fit(X_train, y_train)
predictions = final_model.predict(X_test)
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
print(f"last xgboost rmse : {rmse}")
