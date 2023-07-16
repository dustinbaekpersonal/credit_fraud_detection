from data.preprocess import data_preprocess
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer

import time
import pickle

def train_model(path:str, 
                model:str, 
                mode:str, 
                model_path:str, 
                grid_search:bool=False, 
                grid_params:dict=None) -> pickle:
    """
    Training model
    
    Parameters
    ----------
    path : str
        input csv file path
    
    model: str
        ML model type e.g. rf, lr, xgboost, svm

    model_path: str
        output model pickle path
    
    grid_search: bool, default=False
        whether random grid serach is going to be used
    
    grid_params: dict, default=None
        grid search parameters specified by user

    Returns
    -------
    model pickle file dumped to given model path
    """

    df_raw = pd.read_csv(path)
    data_prep = data_preprocess()
    original_Xtrain, original_Xtest, original_ytrain, original_ytest = data_prep.stratify_df(df_raw)

    if mode == 'base':
        ### Baesline Model ###
        new_df = pd.concat([original_Xtrain,original_ytrain],axis=1)
        new_df = data_prep._scaling(data_prep._cleaning(new_df))
        X_train=new_df.drop('Class',axis=1)
        y_train=new_df['Class']
    
    elif mode == 'under':
        ### Under-sampling Technique ###
        under_df=data_prep.subsample(pd.concat([original_Xtrain,original_ytrain],axis=1),mode='undersampling')
        # data_prep.check_imbalanced(under_df)
        under_df=data_prep._scaling(data_prep._cleaning(under_df))
        X_train=under_df.drop('Class',axis=1)
        y_train=under_df['Class']
    
    elif mode == 'over':
        ### Over-sampling Technique ###
        over_df=data_prep.subsample(pd.concat([original_Xtrain,original_ytrain],axis=1),mode='oversampling')
        # data_prep.check_imbalanced(under_df)
        over_df=data_prep._scaling(data_prep._cleaning(over_df)).sample(frac=0.5)
        X_train=over_df.drop('Class',axis=1)
        y_train=over_df['Class']

    ### Model Training ###
    start_time=time.time()
    if model == 'rf':
        rf = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', verbose=1)
        if grid_search:
            rf_params=grid_params
            rf_cv = GridSearchCV(estimator=rf, param_grid=rf_params, n_jobs=-1, verbose=1, cv=4)
            rf_cv.fit(X=X_train, y=y_train)
            best_model=rf_cv.best_estimator_
            print(pd.Series(data=best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False))

        else:
            rf.fit(X_train,y_train)
        # print(rf_cv.best_params_)
    print(f'time taken for training is {time.time() - start_time}')

    ### Saving trained Model ###
    pickle.dump(best_model, open(model_path, 'wb'))

if __name__ == '__main__':
    path = '../../data/raw/creditcard.csv'
    filename = f'rf_cv_under.pkl'
    model_path = f'../../data/trained_model/{filename}'
    rf_params = {'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [200, 400, 600, 800, 1000]}
    
    train_model(path=path,
                model='rf',
                mode='under',
                model_path=model_path,
                grid_search=True,
                grid_params=rf_params)



