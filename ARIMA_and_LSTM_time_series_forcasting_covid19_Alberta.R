#ARIMA and LSTM time series for covid19 cases in Alberta
# data 612  
# 4/15/2020
#Navid Youseafabdi
#navid.yousefabadi@ucalgary.ca

# loading packages
library(xlsx)
library(forecast)
library(keras)
library(tensorflow)

library(wavScalogram)

library(utils)
library(tidyr)
library(dplyr)

# reading data
covid19_alberta <- read.xlsx("covid19_alberta_by_day.xlsx", sheetName = "Sheet1", header=TRUE)

#removing first three rows
covid19_alberta <- covid19_alberta[-c(1,2,3),] 

#removing last column
#covid19_alberta <- covid19_alberta[-c(nrow(covid19_alberta)),] 

View(covid19_alberta)

# splitting data into train and valid sets
train = covid19_alberta$count[1:23]
train
valid = covid19_alberta$count[24:nrow(covid19_alberta)]
valid

# training ARIMA model
model = auto.arima(train)

# model summary
summary(model)

#plotting the residuals 
tsdisplay(residuals(model), lag.max=45, main='(0,1,0) Model Residuals') 

#plotting the forecast
fcast <- forecast(model, h=30)
plot(fcast)

lines ((length(train)+1):(length(train)+length(valid)), valid, col="red", lwd=3)


# forecasting
forecast = predict(model, n.ahead = 14)

# evaluation
rmse(valid$International.airline.passengers, forecast$pred)


############################################  LSTM for time series 

data <- covid19_alberta$count[1:nrow(covid19_alberta)]

# transform data to stationarity calculating one difference 
diffed = diff(data, differences = 1)
length(diffed)

#k is the number of lags
lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised = lag_transform(diffed, 1)
View(supervised)

#seperating test and train sets
N = nrow(supervised)
n = round(N *0.7, digits = 0)
train = supervised[1:n, ]
test  = supervised[(n+1):N,  ]
test
train

## normalizing the data
scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  
}


Scaled = scale_data(train, test, c(-1, 1))

y_train = Scaled$scaled_train[, 2]
x_train = Scaled$scaled_train[, 1]

y_test = Scaled$scaled_test[, 2]
x_test = Scaled$scaled_test[, 1]


## inverse-transform
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}


# Reshape the input to 3-dim
dim(x_train) <- c(length(x_train), 1, 1)

# specify required arguments
X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]
batch_size = 1                # must be a common factor of both the train and test samples
units = 3                     # can adjust this, in model tuninig phase

#=========================================================================================

model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 2)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics =  c("accuracy")
)

summary(model)


Epochs = 300  
for(i in 1:Epochs ){
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


L = length(x_test)
scaler = Scaled$scaler

predictions = numeric(L)
test

#testing for L times predicting given the first step
tem <- data[n+1]
X <- x_test[1]
for(i in 1:L){

  dim(X) = c(1,1,1)
  yhat <- model %>% predict(X, batch_size=batch_size)
  #store for next prediction
  X <- yhat
  # invert scaling
  yhat <- invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  <- yhat + tem
  # store
  predictions[i] <- yhat
  
  #update temp
  tem <- yhat
  
}

#plotting the predictions
plot(1:length(predictions), predictions,col="red", ylim = c(20,200), pch=19)
points(1:length(predictions), data[n+1:length(predictions)+1],col="green", pch=19)



