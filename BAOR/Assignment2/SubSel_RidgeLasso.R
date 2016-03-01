###########################################################
#### Subset Selection
###########################################################

#install.packages("ISLR")
library(ISLR) #contains the dataset "hitters"
#install.packages("leaps")
library(leaps) #contains the function regsubsets
head(Hitters) #baseball player dataset. We want to predict Salary based on other variables

#Remove NA entries in Hitters
#dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters=na.omit(Hitters) #omits rows corresponding to NA entries
#dim(Hitters)

# Make training and test sets
set.seed(1)
train=sample(1:nrow(Hitters),0.75*nrow(Hitters))
test=-train

#nvmax = maximum number of predictors to consider (default = 8)
#number of predictors in dataset (= 19 in Hitters)
p = ncol(Hitters) - 1; #we subtract the response variable

#Best Subset Selection using Traditional Approach (see Session 7-8)
regfit.full=regsubsets(Salary~.,data=Hitters[train,],nvmax=p)

#regfit.full contains p models, where model t is the best model
#obtained using exactly t predictors (t ranges from 1 to p)
#Note: for each t, the best model was obtained through minimizing training set MSE
reg.summary=summary(regfit.full)
reg.summary

#now, we have to select which model we want to use from 
#the p models in regfit.full. To do so, we use adjusted R^2
reg.summary$adjr2
best.model = which.max(reg.summary$adjr2)

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

#evaluate the best model on the test set
pred=predict.regsubsets(regfit.full,Hitters[test,],best.model)
actual = Hitters$Salary[test];
mean((actual-pred)^2) #test set MSE

# Forward Stepwise Selection
regfit.fwd=regsubsets(Salary~.,data=Hitters[train,],nvmax=p,method="forward")
best.model.fwd = which.min(summary(regfit.fwd)$adjr2)

#Do we obtained the same model with both methods?
coef(regfit.full,best.model)
coef(regfit.fwd,best.model.fwd)

#Best Subset Selection using 10-fold Cross Validation
k=10
set.seed(1)
Hitters.train = Hitters[train,]
#we randomly assign each datapoint in the training set to our k folds. note that in this implementation
#not all of the folds will necessarily have the same number of data points
folds=sample(1:k,nrow(Hitters.train),replace=TRUE)
#cv.errors[j,t] represents the MSE from the best model using t parameters evaluated on fold j
#note: we initialize all values in this matrix to NA. However, all entries will eventually be filled in.
cv.errors=array(NA,dim=c(k,p)) 
for(j in 1:k){
  #let t = num. of parameters. For t=1,...,p, find the best model using MSE on Hitters.train[folds!=j,]
  #This corresponds to steps 2a-c in slide Session 7-10
  #Note: "!=" means "not equal to" in R
  best.fit=regsubsets(Salary~.,data=Hitters.train[folds!=j,],nvmax=p)
  #For t=1,...,p, evaluate the best model using t predictors on fold j (Hitters.train[folds==j,])
  #This corresponds to step 2d in the slides
  for(t in 1:p){
    pred=predict.regsubsets(best.fit,Hitters.train[folds==j,],t)
    actual=Hitters.train$Salary[folds==j]
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
regfit.full=regsubsets(Salary~.,data=Hitters.train, nvmax=19)

#evaluate MSE of final chosen model on test dataset (step 5)
pred=predict.regsubsets(regfit.full,Hitters[test,],best.model)
actual = Hitters$Salary[test];
mean((actual - pred)^2) #test set MSE

###########################################################
#### Ridge and Lasso Regression
###########################################################
#prepare the arguments for glmnet()
x=model.matrix(Salary~.,Hitters)[,-1]
#head(x)
y=Hitters$Salary

#install.packages("glmnet")
library(glmnet)
#Ridge and Lasso regression have a tuneable parameter: lambda (See Session 7-19)
#We wish to choose the best model using CV among lambda=10^-2,10^-1,...,10^10
grid=10^(-2:10) #set sequence of lambdas we want to test

set.seed(65)
#Use 10-fold CV to choose the best value of lambda for ridge regression
#For the command below, alpha=0: ridge regression, alpha=1: lasso regression
cv.out=cv.glmnet(x[train,],y[train],alpha=0,lambda=grid,nfolds=10) 
#plot(cv.out)
bestlam=cv.out$lambda.min

#Train model with best value of lambda on the training set
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=bestlam)

#Evaluate this model on the test set
pred=predict(ridge.mod,x[test,])
actual = y[test]
mean((actual-pred)^2) 

###########################################################
#### Multinomial Logistic Regression with Lasso Penalty
###########################################################

#install and load the "caret" package (which contains the "iris" dataset)
#install.packages("caret")
library(caret)
library(glmnet)

#The iris dataset contains 150 observations for three different types of iris flower
#We will predict the type of flower given the other variables (petal length, etc)
head(iris)
unique(iris$Species) #the three different types of iris flowers
dim(iris)

#prepare the arguments for glmnet()
x=model.matrix(Species~.,iris)[,-1]
y=as.factor(iris$Species)

#create our training set
set.seed(1)
train=sample(1:nrow(iris),0.75*nrow(iris))
test=-train

#Ridge and Lasso regression have a tuneable parameter: lambda (See Session 7-19)
#We wish to choose the best model using CV among lambda=10^-2,10^-1,...,10^10
grid=10^(-2:10) #set sequence of lambdas we want to test

#Use 10-fold CV to choose the best value of lambda for lasso regression
#For the command below, alpha=0: ridge regression, alpha=1: lasso regression
#Note: to perform normal logistic regression when the response is binary, change "multinomial" to "binomial"
cv.out=cv.glmnet(x[train,],y[train],alpha=1,lambda=grid,family="multinomial",nfolds=10)
bestlam=cv.out$lambda.min

#Train model with best value of lambda on the training set
lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=bestlam,family="multinomial")

#Evaluate this model on the test set
pred = predict(lasso.mod, x[test,],type="class");
table(pred,y[test])