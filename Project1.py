from ast import excepthandler
import numpy as np
import pandas as pd
import re
import math
from  scipy.optimize import minimize as opt
import matplotlib.pyplot as plt
import seaborn as sns 
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
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
    #print(df["viewer feeling of youtuber's style "].unique())
    df=df.dropna()
    # print("------------------------------------------")
    # print("Values list of venue: "+str(df['venue'].unique()))
    # print("------------------------------------------")
    # print("Values list of container: "+str(df['container'].unique()))
    # print("------------------------------------------")
    # print("Values list of viewer's feeling: "+str(df["viewer feeling of youtuber's style "].unique()))
    # print("------------------------------------------")
    df=makeDuration(df)
    # df['duration']=(df['duration']-df['duration'].mean())/(df['duration'].max()-df['duration'].min())
    # df['number of in']=(df['number of in']-df['number of in'].mean())/(df['number of in'].max()-df['number of in'].min())
    df['duration']=df['duration']/2000
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
npData=(df.iloc[:,2:]).to_numpy()
#print(npData.shape)
np.random.shuffle(npData)
#print(npData[:,17])
for i in range(1,6):
    npData=np.concatenate((npData,(np.array([npData[:,17]==i])).astype(int).T),axis=1)
#print(npData[:,18:])
X_train=npData[0:3000,0:17]
Y_train=npData[0:3000,18:]
X_test=npData[3000:,0:17]
Y_test=npData[3000:,18:]
Y_predTest=npData[3000:,17]
Y_labelTest=npData[3000:,17]



pca=PCA(n_components=16) #5 is 95% information
pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)


import matplotlib.pyplot as plt
info_rate=np.array([])
for i in range(16):
  info_rate=np.append(info_rate,np.sum(pca.explained_variance_ratio_[:i])*100)
plt.plot(range(16),info_rate)
plt.xlabel("Number of principle components")
plt.ylabel("Information integrity after apply PCA(%)")
plt.show()

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
  

#Model training
model=[1,1,1,1,1,1,1]
for i in range(1,6):    
    if i>3:
        model[i] = LogitRegression( learning_rate = 0.1, iterations = 1000 )
    elif i==3:
        model[i] = LogitRegression( learning_rate = 0.05, iterations = 5000 )
    else:
        model[i] = LogitRegression( learning_rate = 0.01, iterations = 500 )
    #print(Y_train[i])
    model[i].fit( X_train[:,:5], Y_train[:,i-1] )    
    # Prediction on test set
    Y_pred = model[i].predict(X_test[:,:5])    
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
Y_testPredicted=[]
for i in range(X_test.shape[0]):
    max=-10
    maxPred=0
    for j in range(1,6):
        if (model[j].prob(X_test[i,:5]))>max:
            max=(model[j].prob(X_test[i,:5]))
            maxPred=j
    Y_testPredicted.append(maxPred)
Y_testPredicted=np.array([Y_testPredicted]).T
print(Y_labelTest.shape)
print(Y_testPredicted.shape)
print("Precision score: {:.4}".format(precision_score(Y_labelTest,Y_testPredicted, average='micro')))
print("Recall score: {:.4}".format(recall_score(Y_labelTest,Y_testPredicted, average='micro')))
print("Accuracy score: {:.4}".format(accuracy_score(Y_labelTest,Y_testPredicted)))
print("F1 score: {:.4}".format(f1_score(Y_labelTest,Y_testPredicted, average='micro')))
print(np.unique(Y_testPredicted))
sns.heatmap(confusion_matrix(Y_labelTest,Y_testPredicted.reshape((-1,1))), annot=True,xticklabels=range(1,6), yticklabels=range(1,6))
plt.title('Confusion matrix')
plt.xlabel('Predictions')
plt.ylabel('True values')
plt.show()
