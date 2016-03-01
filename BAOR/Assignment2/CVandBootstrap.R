#Which function/logical operators
x = c(1,2,5,15,3,10)
which(x >= 10) #indices of x which are >= 10
which(x >= 4 & x < 8) #indices of x which are >= 4 AND < 8
x[x < 2 | x > 10] #values of x which are < 2 OR > 10

#Creating your own functions
#We've gone over built-in R functions like sum(), mean(), plot(). How do we create our own?
#Here is a function for summing two vectors together (with the same dimension)
#vec1,vec2 are the parameters to our model. sum2vec is the function name
sum2vec = function(vec1,vec2) {
  result = vec1+vec2
  return(result)
}

#Applying our function to an example:
x = c(1,2,5,3,6)
y = c(3,1,2,5,4)
sum2vec(x,y)

#For Loops
#Example 1 - output should be the numbers 1 through 10
#The for loop below is saying the following:
#Start with i=1. Run the code in the brackets, and increment i by 1. Repeat this process until i=10
#In other words, run the code for i=1,2,3,...,10 and stop
for (i in 1:10) {
  print(i)
}

#Example 2 - summing the contents of a vector
#Let's create our own "sum" function using for loops
sum2 = function(x) {
  s = 0; #our sum
  for (i in 1:length(x)) {
    s = s + x[i];
  }
  return(s);
}

x = c(1,2,5,3,6)
sum2(x) #should equal sum(x)
sum(x)

#Cross Validation

#Download Auto.csv off of Canvas (make sure to set your directory to the folder which contains Auto.csv)
#Load Auto and remove rows with missing values, denoted by a "?" in the dataset
Auto = read.csv("Auto.csv",na.strings ="?") #read in the dataset, replacing all instances of "?" with "NA"
Auto = na.omit(Auto)
head(Auto)
dim(Auto)
attach(Auto)

#Want to regress mpg (m) on horsepower (h)
#We want to choose the best out of 5 different candidate models: polynomials with degree 1,...,5, i.e.
#(1) m = b0 + b1*h,
#(2) m = b0 + b1*h + b2*h^2,
#...
#(5) m = b0 + b1*h + b2*h^2 + b3*h^3 + b4*h^4 + b5*h^5

#We first divide the data into a training (70%) and test set (30%)
set.seed(1)
train=sample(1:nrow(Auto),0.7*nrow(Auto))
test = -train 

#Validation set approach: we divide the training data into two sets, a training (70%) and validation set (30%)
#number of elements in our training set
num.train.v = floor(0.7*length(train))
#training set
train.v = train[1:num.train.v] #make train.v first num.train.v elements (we can do this because train is a random sample)
#validation set
valid.v = train[(num.train.v+1):length(train)]

#train model on training set and estimate MSE on validation set
mse = rep(0,5)
for (i in 1:5){
  lm.fit=lm(mpg~poly(horsepower,i),data=Auto[train.v,])
  mpg.pred = predict(lm.fit,newdata = Auto[valid.v,])
  mse[i] = mean((mpg[valid.v]-mpg.pred)^2)
}
best.model = which.min(mse)

#train best model on training + validation set
lm.fit=lm(mpg~poly(horsepower,best.model),data=Auto[train,])
#evaluate model on test set
mpg.pred = predict(lm.fit,newdata = Auto[test,])
mse.best.model = mean((mpg[test]-mpg.pred)^2)

#Cross-Validation
#use the library command to load R packages (collections of useful functions or datasets)
library(boot) #contains useful functions for cross-validation

cv.error=rep(0,5) #cross validation MSEs for our 5 models
for (i in 1:5){
  glm.fit=glm(mpg~poly(horsepower,i),data=Auto[train,])
  #delta[1] corresponds to MSE for object "cv.glm(Auto,glm.fit)"
  cv.error[i]=cv.glm(Auto[train,],glm.fit)$delta[1] # Leave-One-Out Cross-Validation
  #cv.error[i]=cv.glm(Auto[train,],glm.fit,K=10)$delta[1] #K-fold cross validation (K=10)
}
best.model = which.min(cv.error)

#train best model on entire training set
lm.fit=lm(mpg~poly(horsepower,best.model),data=Auto[train,])
#evaluate model on test set
mpg.pred = predict(lm.fit,newdata = Auto[test,])
mse.best.model = mean((mpg[test]-mpg.pred)^2)

detach(Auto)

#Bootstrapping
library(ISLR) #contains the dataset "Portfolio"
head(Portfolio) #contains two possibly correlated stocks, X and Y
#The minimum variance portfolio is the value of alpha which minimizes Var(alpha*X + (1-alpha)*Y)
#We will use bootstrapping to estimate the standard error of our estimate of alpha*

#outputs an estimate for alpha using the data in data[indices,], 
#where "indices" corresponds to our bootstrap sample
alpha.fn = function(data,indices){
  X = data$X[indices]
  Y = data$Y[indices]
  alpha = (var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)) #our estimate of alpha* on this sample
  return(alpha)
}
B = 1000 #number of bootstrap samples
boot(Portfolio,alpha.fn,B) 

#Here's how the boot() command works:
res = rep(0,B) #our estimate of alpha* for each bootstrap sample
for (i in 1:B){
  indices = sample(1:nrow(Portfolio),nrow(Portfolio),replace=T)
  res[i] = alpha.fn(Portfolio,indices)
}
hist(res) #approximate "distribution" of our estimate
mean(res) #our estimate 
sd(res) #standard error of our estimate for confidence intervals
