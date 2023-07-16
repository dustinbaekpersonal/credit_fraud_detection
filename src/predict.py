import pickle
import pandas as pd

from data.preprocess import data_preprocess
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

filename = f'rf_cv_under.pkl'
model_path = f'../../data/trained_model/{filename}'
best_model= pickle.load(open(model_path, 'rb'))

feature_impt = best_model.feature_importances_

path = '../../data/raw/creditcard.csv'
df_raw = pd.read_csv(path)
data_prep = data_preprocess()
original_Xtrain, original_Xtest, original_ytrain, original_ytest = data_prep.stratify_df(df_raw)

# steps=[data_preprocess()]
# pipeline = Pipeline(steps=steps)

# pipeline.fit_transform(original_Xtrain)

test_df = data_prep._scaling(data_prep._cleaning(pd.concat([original_Xtest,original_ytest],axis=1)))
X_test = test_df.drop('Class',axis=1)
y_test = test_df['Class']

print(pd.Series(data=feature_impt, index=X_test.columns).sort_values(ascending=False))

y_pred=best_model.predict(X_test)
y_true=y_test

# y_prob=best_model.predict_proba(X_test)[:,1]

# PrecisionRecallDisplay.from_predictions(y_test, y_prob)
# plt.show()

target_names=['class 0', 'class 1']
print(classification_report(y_true=y_true, y_pred=y_pred,target_names=target_names))
print(confusion_matrix(y_true=y_true, y_pred=y_pred))
print(f'{average_precision_score(y_true=y_true, y_score=y_pred):.3f}')
df_eval=pd.DataFrame({'pred':y_pred,
                    'true':y_true})
print(df_eval[(df_eval['true']==1) & (df_eval['pred']==0) | (df_eval['true']==0) & (df_eval['pred']==1)].to_string())