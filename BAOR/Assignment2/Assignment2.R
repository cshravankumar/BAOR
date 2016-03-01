#############################
# Shravan Kumar Chandrasekaran
# IEOR 4574 
# Homework 2
# Due date: 02/18
#############################
library(boot) #contains useful functions for cross-validation
library(leaps) #contains the function regsubsets

getwd()
dir = "/Users/Shravan/BAOR/Assignment2"

#Setting working directory
setwd(dir)

#Reading College Data
Data<-read.csv("CollegeData.csv")
#Summaziring the Data
summary(Data)

Data_Cleaned<- na.omit(Data)

#Reading College Dictionary Data
Data_Dictionary<-read.csv("CollegeDataDictionary.csv")
#Summaziring the Data
summary(Data_Dictionary)
Data_Dictionary_Cleaned<-na.omit(Data_Dictionary)


#Making new log columns for $ columns
Data_Cleaned$COSTT4_A_Log<-log(as.numeric(Data_Cleaned$COSTT4_A))
Data_Cleaned$TUITIONFEE_OUT_Log<-log(as.numeric(Data_Cleaned$TUITIONFEE_OUT))
Data_Cleaned$TUITFTE_Log<-log(as.numeric(Data_Cleaned$TUITFTE))
Data_Cleaned$AVGFACSAL_Log<-log(as.numeric(Data_Cleaned$AVGFACSAL))

#Making interaction terms for Log columns
Data_Cleaned$INTER_COSTT4_A_Log_TUITIONFEE_OUT_Log <- Data_Cleaned$COSTT4_A_Log*Data_Cleaned$TUITIONFEE_OUT_Log
Data_Cleaned$INTER_COSTT4_A_Log_TUITFTE_Log <- Data_Cleaned$COSTT4_A_Log*Data_Cleaned$TUITFTE_Log
Data_Cleaned$INTER_COSTT4_A_Log_AVGFACSAL_Log <- Data_Cleaned$COSTT4_A_Log*Data_Cleaned$AVGFACSAL_Log
Data_Cleaned$INTER_TUITIONFEE_OUT_Log_TUITFTE_Log <- Data_Cleaned$TUITIONFEE_OUT_Log*Data_Cleaned$TUITFTE_Log
Data_Cleaned$INTER_TUITIONFEE_OUT_Log_AVGFACSAL_Log <- Data_Cleaned$TUITIONFEE_OUT_Log*Data_Cleaned$AVGFACSAL_Log
Data_Cleaned$TUITFTE_Log_AVGFACSAL_Log <- Data_Cleaned$TUITFTE_Log*Data_Cleaned$AVGFACSAL_Log

#Finding means for existing integer columns
MEAN <-rep(0, times=19)
for(i in 2:20)
{
MEAN[i-1]<-mean(Data_Cleaned[,i])
}

#Dividing into Test and Train data sets (25:75 Split)
set.seed(4574)
train <- sample(1:nrow(Data_Cleaned), 0.75*nrow(Data_Cleaned))
test = -train

#SAT_AVG for Train and Test Data Sets

mean(Data_Cleaned$SAT_AVG[train])
mean(Data_Cleaned$SAT_AVG[test])


#Creating our own "predict" function
#Takes as input:
#regfit.full: the outputted model from regsubsets()
#newdata: the new dataaset you want to use to generate predictions
#t: the number of parameters in your model
predict.regsubsets=function(regfit.full,newdata,t){
  #In this problem, form="Salary~.". It represents the modeling argument we inputted when calling regsubsets()
  form=as.formula(regfit.full$call[[2]])
  mat=model.matrix(form,newdata) #mat = model.matrix(Salary~., newdata)
  coefi=coef(regfit.full,id=t) #obtain the coefficients of the model corresponding to t
  xvars=names(coefi)
  pred = mat[,xvars]%*%coefi
  return(pred)
}




#Best Subset Selection using 5-fold Cross Validation
k=5
p=8
Data_Cleaned.train = Data_Cleaned[train,]
#we randomly assign each datapoint in the training set to our k folds. note that in this implementation
#not all of the folds will necessarily have the same number of data points
folds=sample(1:k,nrow(Data_Cleaned.train),replace=TRUE)
#cv.errors[j,t] represents the MSE from the best model using t parameters evaluated on fold j
#note: we initialize all values in this matrix to NA. However, all entries will eventually be filled in.
cv.errors=array(NA,dim=c(k,p)) 
for(j in 1:k){
  #let t = num. of parameters. For t=1,...,p, find the best model using MSE on Data_Cleaned.train[folds!=j,]
  #This corresponds to steps 2a-c in slide Session 7-10
  #Note: "!=" means "not equal to" in R
  best.fit=regsubsets(SAT_AVG~.-INSTNM,data=Data_Cleaned.train[folds!=j,],nvmax=p)
  #For t=1,...,p, evaluate the best model using t predictors on fold j (Data_Cleaned.train[folds==j,])
  #This corresponds to step 2d in the slides
  for(t in 1:p){
    pred=predict.regsubsets(best.fit,Data_Cleaned.train[folds==j,],t)
    actual=Data_Cleaned.train$SAT_AVG[folds==j]
    #cv.errors[j,t] represents the MSE from the best model using t parameters evaluated on fold j
    #(see step 2d of slides)
    cv.errors[j,t]=mean((actual-pred)^2)
  }
}

#average MSEs across the folds j=1,...,k (step 2e)
#"apply(cv.errors,2,mean)": apply the "mean" functions to the columns of matrix cv.errors
#note: the second argument specifies whether we should apply the function to "1" the rows, or "2" the columns
mean.cv.errors=apply(cv.errors,2,mean)
mean.cv.errors

#compute the "best" number of parameters, t*, through minimizing CV MSEs over t=1,...,p (step 3)
best.model = which.min(mean.cv.errors)

#find the best model with t* predictors using entire training dataset (step 4)
regfit.full=regsubsets(SAT_AVG~.-INSTNM,data=Data_Cleaned.train, nvmax=p)

#evaluate MSE of final chosen model on test dataset (step 5)
pred=predict.regsubsets(regfit.full,Data_Cleaned[test,],best.model)
actual = Data_Cleaned$SAT_AVG[test];
mean((actual - pred)^2) #test set MSE

###########################################################
#### Ridge and Lasso Regression
###########################################################
#prepare the arguments for glmnet()
x=model.matrix(SAT_AVG~.-INSTNM,Data_Cleaned)[,-1]
#head(x)
y=Data_Cleaned$SAT_AVG

#install.packages("glmnet")
library(glmnet)
#Ridge and Lasso regression have a tuneable parameter: lambda (See Session 7-19)
#We wish to choose the best model using CV among lambda=10^-2,10^-1,...,10^10
grid=10^(-3:3) #set sequence of lambdas we want to test

#set.seed(65)
#Use 10-fold CV to choose the best value of lambda for ridge regression
#For the command below, alpha=0: ridge regression, alpha=1: lasso regression
cv.out=cv.glmnet(x[train,],y[train],alpha=0,lambda=grid,nfolds=5) 
#plot(cv.out)
bestlam=cv.out$lambda.min

#Train model with best value of lambda on the training set
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=bestlam)

#Evaluate this model on the test set
pred=predict(ridge.mod,x[test,])
actual = y[test]
mean((actual-pred)^2) 

