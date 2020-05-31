import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle

df=pd.read_csv("kidney_disease.csv")

df = df.drop(['id'], axis = 1) 
  

col_names = df.columns 
  
for c in col_names: 
    df = df.replace("?", np.NaN) 
df = df.apply(lambda x:x.fillna(x.value_counts().index[0])) 




df.replace(['\tno', '\tyes',  
              'ckd\t'], 
             ['no', 'yes' , 'ckd'], inplace = True)

category_col =['rbc', 'pc', 'pcc', 'ba', 'htn', 
               'dm', 'cad', 'appet', 'pe' , 'ane' , 'classification']  
labelEncoder = preprocessing.LabelEncoder()

mapping_dict ={} 
for col in category_col: 
    df[col] = labelEncoder.fit_transform(df[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping 
    

print(mapping_dict) 



X=df.drop('classification',axis='columns')
Y=df['classification']

X=pd.get_dummies(X,drop_first=True)
print(X.info())
X=X.replace(['\tno', '\tyes'], ['no','yes'], inplace=True)
X=df.to_numpy()
print(Y.value_counts(dropna=False))
Y=Y.replace('ckd\t','ckd')

  
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=50,min_samples_leaf=1,random_state=24)
rf.fit(X,Y)


pickle.dump(rf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
