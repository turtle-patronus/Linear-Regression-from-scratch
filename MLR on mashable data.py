#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')



def predict(self,X_test,Y_test):
        ones=np.ones(X_test.shape)
        X_test=np.append(ones,X_test,axis=1)
        Y_pred=np.matmul(X_test,theta)
        error_percentage=(abs(Y_pred-y_test)/y_test)*100
        return Y_pred,error_percentage




def computeCostFunction(self):
        h=np.matmul(self.X,theta)
        J=(1/(2*self.m))*np.sum((h-self.Y)**2)
        return J



def predictUsingEquation(X_test,y_test):
        ones=np.ones(X_test.shape)
        X_test=np.append(ones,X_test,axis=1)
        X_cross = np.matmul(np.linalg.pinv(np.matmul(X, X.T)), X)
        w=np.matmul(X_cross, y)
        y_pred=np.matmul(X_test,w)
        return y_pred,(abs(y_test-y_pred)/y_test)*100

def inverse_transform_X(self,X):
        X_transformed=X.copy()
        num_of_features=X_transformed.shanp.linalg.invpe[1]
        for i in range(num_of_features):
            feature=X_transformed[:,i]
            Mean=self.minMax_X[i][0]
            Min=self.minMax_X[i][1]
            Max=self.minMax_X[i][2]
            feature=feature*(Max-Min)+Mean
            X_transformed[:,i]=feature
        return X_transformed



def inverse_transform_Y(self,y):
        y_transformed=y.copy()
        if y_transformed.ndim==1:
            y_transformed=np.reshape(y_transformed,(y_transformed.shape[0],1))
        num_of_features=y_transformed.shape[1]
        for i in range(num_of_features):
            feature=y_transformed[:,i]
            Mean=minMax_y[i][0]
            Min=minMax_y[i][1]
            Max=minMax_y[i][2]
            feature=feature*(Max-Min)+Mean
            y_transformed[:,i]=feature
        return np.reshape(y_transformed,y_transformed.shape[0])




data = pd.read_csv(r"FILE_PATH.csv")


data = data.drop(['url'],axis=1)


# In[15]:


Q1 = data.quantile(q=0.25) 
Q3 = data.quantile(q=0.75)
IQR = Q3-Q1
print(IQR)



sorted_shares = data.sort_values(' shares') 
median = sorted_shares[' shares'].median() 
q1 = sorted_shares[' shares'].quantile(q=0.25) 
q3 = sorted_shares[' shares'].quantile(q=0.75) 
iqr = q3-q1




Inner_bound1 = q1-(iqr*1.5) 
print(f'Inner Boundary 1 = {Inner_bound1}')
Inner_bound2 = q3+(iqr*1.5)  
print(f'Inner Boundary 2 = {Inner_bound2}')
Outer_bound1 = q1-(iqr*3)    
print(f'Outer Boundary 1 = {Outer_bound1}')
Outer_bound2 = q3+(iqr*3)   
print(f'Outer Boundary 2 = {Outer_bound2}')




data2 = data[data[' shares']<=Outer_bound2]




print(f'Data before Removing Outliers = {data.shape}')
print(f'Data after Removing Outliers = {data2.shape}')
print(f'Number of Outliers = {data.shape[0] - data2.shape[0]}')



X = data.iloc[:,0:59].values
print(X.shape)




y = data.iloc[:,-1].values
print(y.shape)


y=y.reshape(-1,1)
print(y.shape)




train_size=int(0.8*data2.shape[0])
test_size=int(0.2*data2.shape[0])
print("Training set size : "+ str(train_size))
print("Testing set size : "+str(test_size))



Data=data2.sample(frac=1)
X=data2.iloc[:,0:59].values
y=data2.iloc[:,59].values




X_train=X[0:train_size,:]
y_train=y[0:train_size]


print(X_train.shape)
print(y_train.shape)



X_test=X[train_size:,:]
y_test=y[train_size:]




print(X_test.shape)
print(y_test.shape)




y_train=y_train.reshape(-1,1)
print(y_train.shape)
y_test=y_test.reshape(-1,1)
print(y_test.shape)



theta=np.random.randn(X.shape[1])


y_pred,error_percentage=predictUsingEquation(X_test,y_test)
y_pred=inverse_transform_Y(y_pred)
print(error_percentage)




y_pred_train,error_percentage_train_normal=predictUsingEquation(X_train,y_train)
y_pred_train=inverse_transform_Y(y_pred_train)
print(computeCostFunction())



X_train=inverse_transform_X(X_train)
y_train=inverse_transform_Y(y_train)
X_test=inverse_transform_X(X_test)
y_test=inverse_transform_Y(y_test)



plt.scatter(X_train,y_train)
plt.plot(X_train,y_pred_train,'r')
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.title('Training set prediction using LR Equation')
plt.show()




plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,'r')
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.title('Test set prediction using LR Equation')
plt.show()

