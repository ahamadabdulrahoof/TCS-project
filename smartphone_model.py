import pandas as pd 
import numpy as np
import pickle
data_train=pd.read_csv('MobileTrain.csv')
data_test=pd.read_csv('MobileTest.csv')
data = pd.concat([data_train.assign(ind="train"), data_test.assign(ind="test")])
data.drop(['id'],axis=1,inplace=True)
# outlier detection
ax=['fc','px_height']
def outlier_det(col_name,data):
    Q1=np.percentile(data[col_name],25,interpolation='midpoint')
    Q2=np.percentile(data[col_name],50,interpolation='midpoint')
    Q3=np.percentile(data[col_name],75,interpolation='midpoint')
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    new_df = data[(data[col_name] < upper) & (data[col_name] > lower)]
    return new_df
for i in ax:
    data=outlier_det(i,data)
#splitting as train and test
test,train=data[data['ind'].eq("test")],data[data['ind'].eq("train")]
test=test.drop(['ind','price_range'],axis=1)
train=train.drop('ind',axis=1)
# splitting as X and y
X=train.drop(['price_range','touch_screen','four_g','wifi','dual_sim','blue','fc','m_dep','n_cores','pc','three_g'],axis=1)
y=train['price_range']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
# modelling
from sklearn.svm import SVC
svm_linear=SVC(kernel='linear')
svm_linear.fit(X_train,y_train)
#Saving the model to disk
pickle.dump(svm_linear,open('smartphone_model.pkl','wb') )
