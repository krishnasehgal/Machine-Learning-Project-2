
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[2]:


maxAcc = 0.0     
maxIter = 0
C_Lambda = 0.03          # Initialise Regulariser for Basis Function
TrainingPercent = 80     # Used 80% of dataset for Training
ValidationPercent = 10   # Used 10% of dataset for Validation
TestPercent = 10         # Used 10% of dataset for Test Percent
M = 10                   # Number of Clusters
PHI = []                 # Empty List for Design Matrix
IsSynthetic = False


# In[3]:


def GetTargetVector(filePath):
    t = []    # Target Class
    with open(filePath, 'rU') as f:      
        reader = csv.reader(f)      # Reading Target Set File
        for row in reader:           
            t.append(int(row[0]))   # Appending data into target list
    #print("Raw Training Generated..")
    return t

def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)    # Reading Raw Data file  
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column)) # Appending data into Data Matrix
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)  # Deleting Features with 0 Variance  
    dataMatrix = np.transpose(dataMatrix)     # Transpose of Matrix 
    #print ("Data Matrix Generated..")
    return dataMatrix

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):  # Function for Splitting Target Value data for Training
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))  # Using 80% of Target Data for Training
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t
  
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):  # Function for Splitting Cleaned data for Training
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))  # Using 80% of Cleaned Data for Training    
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): # Function for Splitting Cleaned data for Validation
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))  # Using 10% of Cleaned Data for Validation
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): # Function for Splitting Cleaned data for Validation
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))       # Using 10% of Cleaned Data for Validation
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):  #Generating Covariance Matrix
    BigSigma    = np.zeros((len(Data),len(Data)))      # 41x41 empty matrix
    DataT       = np.transpose(Data)                   # 69kx41 Full data matrix
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))   # Training length     
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])            
        varVect.append(np.var(vct))   # Variance is calculated for each feature
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]   # Diagonal Matrix with sigma^2(Variance)
    
    BigSigma = np.dot(200,BigSigma)   # Big Sigma value is normalised
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):  # To Calculate exponential term in Gaussian Distribution
    R = np.subtract(DataRow,MuRow)  
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))   # Calculating phi matrix value
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80): # Design Matrix
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)   # Inverse of Big Sigma
    for  C in range(0,len(MuMatrix)):     
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv) # Calculating 10x57k phi matrix
    #print ("PHI Generated..")
    return PHI

def GetWeightsClosedForm(PHI, T, Lambda): 
    Lambda_I = np.identity(len(PHI[0])) 
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)   # Transpose of Design Matrix 10 X 57k
    PHI_SQR     = np.dot(PHI_T,PHI)   # PHI(T)xPHI  10x10
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR) # Adding Regularizer
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI) # Inverse of PHI(T)xPHI
    INTER       = np.dot(PHI_SQR_INV, PHI_T) # Inverse(PHI(T)xPHI)xPHI(T) 10x57k
    W           = np.dot(INTER, T)   # (Inverse(PHI(T)xPHI)xPHI(T))x(Target Value) 10x1
    ##print ("Training Weights Generated..")
    return W

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))  # Returns target data value used for validation and testing
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):    
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)  # (Actual value-Predicted value)^2
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):   # Check number of correctly predicted values
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))     # Return Accuracy = correctly predicted/ total predicted
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))  # return accuracy and erms


# ## Fetch and Prepare Dataset

# In[4]:


RawTarget = GetTargetVector('Querylevelnorm_t.csv')   # Fetching Target data File
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)  # Fetching Raw data File


# ## Prepare Training Data

# In[5]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[6]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget)))) # Target data for Validation 
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget))) #raw data for Validation
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[7]:


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct)))) # Target data for Testing 
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))   #raw data for testing
print(ValDataAct.shape)
print(ValData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[8]:


ErmsArr = []   # Empty List to store ERMS value
AccuracyArr = [] #list to store Accuracy

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData)) # K-means Clustering Fitting
Mu = kmeans.cluster_centers_ #Centroid Values

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic) # Calculate Big Sigma
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent) # Design Matrix for training
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda))  #Regularisation
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) # Design Matrix for testing
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100) # Design Matrix for validation


# In[9]:


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[10]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget)) #erms for training
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct)) #erms for validation
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct)) #erms for testing


# In[11]:


print ('UBITname      = Krishna Sehgal')
print ('Person Number = 50291124')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = 10 \nLambda = 0.9")
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))    #erms for training
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))  #erms for validation
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))        #erms for testing
print ("Accuracy Training   = " + str(float(TrainingAccuracy.split(',')[0])))  #Accuracy for training
print ("Accuracy Validation = " + str(float(ValidationAccuracy.split(',')[0])))#Accuracy for Validation
print ("Accuracy Testing    = " + str(float(TestAccuracy.split(',')[0]))) #Accuracy for testing


# ## Gradient Descent solution for Linear Regression

# In[12]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[13]:


W_Now        = np.dot(220, W)  #Initialise weight randomly
La           = 2               
learningRate = 0.05
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

Ermstr = []
Ermsval = []
Ermstest = []

learningRate = [0.01,0.05,0.1,0.15, 0.20]
for j in range(0,5):
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    for i in range(0,400):

        #print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])  # Weights update
        La_Delta_E_W  = np.dot(La,W_Now) # Scaling
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)   
        Delta_W       = -np.dot(learningRate[j],Delta_E)  # Learning Updation
        W_T_Next      = W_Now + Delta_W    # After Update of weight
        W_Now         = W_T_Next           # Next weight

        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) # Predicted Target value for training data
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)  
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))

        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) #Predicted Target value for Validation data
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))

        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next)   #Predicted Target value for testing
        Erms_Test = GetErms(TEST_OUT,TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        
       
    print('loop complete')
    Ermstr.append(np.around(min(L_Erms_TR),5))
    Ermsval.append(np.around(min(L_Erms_Val),5))
    Ermstest.append(np.around(min(L_Erms_Test),5))
    print(Ermstr)
    print(Ermsval)
    print(Ermstest)


# In[14]:


print(Ermstr)
print(Ermsval)
print(Ermstest)

print ('----------Gradient Descent Solution--------------------')

print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))


# In[15]:


learningRate = [0.01,0.05,0.1,0.15, 0.20, 1]
for j in range(0,6):
    print (learningRate[j])

