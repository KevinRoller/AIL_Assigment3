from ast import excepthandler
import numpy as np
import pandas as pd
import re
import math
from  scipy.optimize import minimize as opt
import matplotlib.pyplot as plt
patternTime='^(?:[0-9]+)[:;,.](?:[0-9]+[:;,.])?(?:[0-9]+)$'
def loadData(file:str):
    df=pd.read_csv(file,delimiter=',',encoding_errors='ignore')
    return df
def rightBeginEnd(row):
    if (re.search(patternTime,str(row['start time']))!=None) and (re.search(patternTime,str(row['end time']))!=None):
        return True
    else:
        return False
def getSecond(s:str):
    s=s.replace(' ','')
    tempList=re.split("[:;,.]",s)
    power=1
    result=0
    try:
        for i in reversed(tempList):
            result+=int(i)*power
            power*=60
    except:
        print(s)
    return result
def getDuration(row):
    return getSecond(row['end time'])-getSecond(row['start time'])
def makeDuration(df:pd.DataFrame):
    count=0
    i=0
    df=df[[(rightBeginEnd(df.iloc[i,:]))for i in range(df.shape[0])]]
    df['duration']=df.apply(getDuration,axis=1)
    return df
#def cleanSpace(value):

def oneHot(df:pd.DataFrame):
    for i in ["venue","container"]:
        temp=pd.get_dummies(df[i])
        df.drop(i,axis=1,inplace=True)
        df=pd.concat([df,temp],axis=1)
    return df
def func(row):
    return str(row).replace(' ','').lower()
from sklearn.ensemble import RandomForestClassifier
def cleanData(df:pd.DataFrame):
    df=df[['start time','end time','number of in','venue','container','describe how to make it',"viewer feeling of youtuber's style "]]
    df['venue'][(df['venue']=='x')|(df['venue']=='home')]='other'
    df['venue'][df['venue']=='boat restaurant']='fine restaurant'
    df['container'][df['container']=='Bag']='bag'
    df['container'][(df['container']=='cup')|(df['container']=='plastic glass')]='glass'
    df['container'][(df['container']=='x')|(df['container']=='no')]='other'
    df['container'][df['container']=='hand']='hands-on'
    df['container'][df['container']=='clay bot']='pot'
    df['container'][df['container']=='tray ']='tray'
    df.drop(df[(df["viewer feeling of youtuber's style "]=='x')|(df["viewer feeling of youtuber's style "]=='0')].index, inplace=True)
    print(df["viewer feeling of youtuber's style "].unique())
    df=df.dropna()
    df=makeDuration(df)
    df['duration']=df['duration']/1000
    df['venue']=df['venue'].apply(func)
    df['container']=df['container'].apply(func)
    temp=df["viewer feeling of youtuber's style "]
    df.drop("viewer feeling of youtuber's style ",axis=1,inplace=True)
    df=oneHot(df)
    #print(df.shape)
    df.insert(loc=19,column="viewer feeling",value=temp)
    #print(df.columns)
    df['describe how to make it']=pd.to_numeric(df['describe how to make it'],errors='coerce')
    df['viewer feeling']=pd.to_numeric(df['viewer feeling'],errors='coerce')
    
    return df

df=loadData(r"D:\học python\AIL\Annotation_AllVideos_FPT_Ver1.csv")
df=cleanData(df)
#print(df.shape)
#print(df.isna().any())
npData=(df.iloc[:,2:]).to_numpy()
# print(df.columns)
# print(npData.shape)
np.random.shuffle(npData)
print(npData[:,17])
for i in range(1,6):
    npData=np.concatenate((npData,(np.array([npData[:,17]==i])).astype(int).T),axis=1)
print(npData[:,18:])
#print(npData1.shape)
# print(npData1[0:4])
# print('----------------')
X_train=npData[0:3000,0:17]
Y_train=npData[0:3000,18:]
# print(X[0:4])
# print('----------------')
# print(Y[0:4])
X_test=npData[3000:,0:17]
Y_test=npData[3000:,18:]
Y_predTest=npData[3000:,17]


from sklearn.decomposition import PCA
pca=PCA(n_components=16) #712 is 99% information
pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)


import matplotlib.pyplot as plt
info_rate=np.array([])
for i in range(16):
  info_rate=np.append(info_rate,np.sum(pca.explained_variance_ratio_[:i])*100)
plt.plot(range(16),info_rate)
plt.xlabel("K vector rieng")
plt.ylabel("Bao toan thong tin(%)")
plt.show()

#print(X_train.shape)
# rfc=RandomForestClassifier(n_estimators=100)
# rfc.fit(X_train,Y_train)
# y_pred=rfc.predict(X_test)
# from sklearn import metrics
# print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
"----------------------------------------------------------------"

# def g(z):
#     sigm = 1.0/(1.0+np.exp(-z))
#     return sigm
# print(in_theta.shape)
# def costFunction(theta, x, y):   
#     m = len(y)
#     h_theta = g(x.dot(theta))
#     J = (1.0/m)* (((-y).transpose()).dot(np.log(h_theta)) - (1.0 -y.transpose()).dot(np.log(1.0-h_theta)))
#     grad = grad = (1.0/m)* x.transpose().dot(h_theta - y)    
#     #return J, grad
#     print ('Cost at theta:', str(J[0,0]))
#     print ('Gradient at theta:','\n', str(grad[0,0]),'\n', str(grad[1,0]),'\n', str(grad[2,0]))
# def CostFunction(theta, x, y):
#     m = len(y)
#     h_theta = g(x.dot(theta))
#     J = (1.0/m)* (((-y).transpose()).dot(np.log(h_theta)) - (1.0 -y.transpose()).dot(np.log(1.0-h_theta)))
#     J = np.float64(J)
#     return J
# def Gradient(theta, x, y):
#     m = len(y)
#     n = x.shape[1]
#     theta = theta.reshape((n,1))
#     h_theta = g(x.dot(theta))
#     print(x.shape)
#     print(theta.shape)
#     print((h_theta-y).shape)
#     grad = (1.0/m)* (x.transpose().dot(h_theta - y)) 
#     print(grad.flatten().shape)
#     return grad.flatten()   
# Result = opt(fun = CostFunction, x0 = in_theta, args = (X, Y), method = 'TNC', jac = Gradient, options ={'maxiter':400})
# theta = Result.x
# print('Cost at theta:',Result.fun, '\n', 'Theta:', Result.x)
# def predict(theta, x):    
#     m = X.shape[0]
#     p = np.zeros((m,1))
#     n = X.shape[1]
#     theta = theta.reshape((n,1))
#     h_theta = g(X.dot(theta))    
#     for i in range(0, h_theta.shape[0]):
#         if h_theta[i] > 0.5:
#             p[i, 0] = 1
#         else:
#             p[i, 0] = 0
#     return p
# p = predict(theta, X_test)
# print ('Test Accuracy:', (Y_test[p == Y_test].size / float(Y_test.size)) * 100.0)


"----------------------------------------------------------"
class LogitRegression() :
    def __init__( self, learning_rate, iterations ) :        
        self.learning_rate = learning_rate        
        self.iterations = iterations
          
    # Function for model training    
    def fit( self, X, Y ) :        
        # no_of_training_examples, no_of_features        
        self.m, self.n = X.shape        
        # weight initialization        
        self.W = np.zeros( self.n )        
        self.b = 0        
        self.X = X        
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :            
            self.update_weights()            
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :           
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
          
        # calculate gradients        
        tmp = ( A - self.Y.T )        
        tmp = np.reshape( tmp, self.m )        
        dW = np.dot( self.X.T, tmp ) / self.m         
        db = np.sum( tmp ) / self.m 
          
        # update weights    
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db
          
        return self
      
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :    
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
        Y = np.where( Z > 0.5, 1, 0 )        
        return Y
    def prob( self, X ) :    
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )                
        return Z
  
# Driver code


#Model training
model=[1,1,1,1,1,1,1]
for i in range(1,6):    
    if i>=3:
        model[i] = LogitRegression( learning_rate = 0.1, iterations = 1000 )
    else:
        model[i] = LogitRegression( learning_rate = 0.01, iterations = 500 )
    #print(Y_train[i])
    model[i].fit( X_train, Y_train[:,i-1] )    
    # Prediction on test set
    Y_pred = model[i].predict(X_test)    
    # measure performance    
    correctly_classified = 0    
    # counter    
    count = 0    
    for count in range( np.size( Y_pred ) ) :  
        if Y_test[count,i-1] == Y_pred[count] :            
            correctly_classified = correctly_classified + 1
    print( "Accuracy on test set by our model       :  ", i , ( 
        correctly_classified / count ) * 100 )
correctly=0
for i in range(X_test.shape[0]):
    max=-10
    maxPred=0
    for j in range(3,6):
        if (model[j].prob(X_test[i]))>max:
            max=(model[j].prob(X_test[i]))
            maxPred=j
    if (maxPred==Y_predTest[i]):
        correctly+=1
print("Overall accuracy :",correctly/X_test.shape[0]*100)
