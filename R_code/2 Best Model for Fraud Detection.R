
library(ggplot2)
library(randomForest)
library(caret)
library(readr)
library(dplyr)
library(DMwR)
library(nnet) 
library(xgboost)
library(e1071)  
library(gbm)    
data <- read.csv("enron.csv")
sum(data$salary[-131],na.rm = T)
data$salary[131]
#移除第131行观测
data <- data[-131,]

#响应变量转换为因子变量，为的是后面重采样
data$poi <- as.factor(data$poi)

data <- data[,!names(data)  %in%  c("X","email_address")]

#数据重采样
set.seed(123)
mydata <- SMOTE(poi~.,data=data)
poi <- as.factor(mydata$poi)
mydata <- mydata[,!names(mydata)  %in%  c("poi")]
missing_values <- is.na(mydata)
mean_values <- colMeans(mydata, na.rm = TRUE)
for (i in 1:ncol(mydata)) {
  mydata[missing_values[, i], i] <- mean_values[i]
}

set.seed(123)
imp <- filterVarImp(x = mydata, y = poi)
imp <- data.frame(cbind(variable = rownames(imp), score = imp[,1]))
imp$score <- as.double(imp$score)
ggplot(imp, aes(x=reorder(variable, score), y=score)) + 
  geom_point() +
  geom_segment(aes(x=variable,xend=variable,y=0,yend=score)) +
  ylab("Importance") +
  xlab("Variable Name") +
  coord_flip() +
  theme_bw()
sort_imp <- imp[order(imp$score,decreasing = TRUE),]
features <- sort_imp[c(1:10),1]
mydata$poi <- poi
splitIndex <- createDataPartition(mydata$poi, p = 0.7, list = FALSE)


#因为第5个变量几乎是一个常数，对于模型没有任何用处，因此移除
train <- mydata[splitIndex, -5]
test <- mydata[-splitIndex, -5]




#rf_model
#gbm_model

newdata <- data.frame(salary=c(378601,300885),to_messages=c(2858,1500),
                      deferral_payments=c(346576,514329),total_payments=c(2659589,15456290),
                      loan_advances=c(28551667,40962500),bonus=c(1350000,1565959),
                      restricted_stock_deferred=c(-225368.5,15356290),deferred_income=c(-833,-508406.8),
                      total_stock_value=c(252055,4463660),expenses=c(65907,51281.73),
                      from_poi_to_this_person=c(140,10),exercised_stock_options=c(3490288,2604490),
                      from_messages=c(27,554),other=c(1621,1829457),
                      from_this_person_to_poi=c(15,63),long_term_incentive=c(974293,927290),
                      shared_receipt_with_poi=c(1593,1428),restricted_stock=c(252055,8453763),
                      director_fees=c(61306,101444))

normal_situation <- data.frame(
  salary = c(85000, 120000), # Average salary level
  to_messages = c(700, 1200), # The normal number of letters received
  deferral_payments = c(5000, 10000), # Deferred payments in the normal range
  total_payments = c(120000, 150000), # Total payments are in line with salary
  loan_advances = c(0, 0), # Regular employees should not have loan advances
  bonus = c(10000, 20000), # A reasonable bonus amount
  restricted_stock_deferred = c(0, 0), # Normally there are no restricted stock deferrals
  deferred_income = c(-5000, -10000), # Deferred revenue is usually negative,
                                      #  but it should not be too high
  total_stock_value = c(50000, 80000), # The total stock value is reasonable
  expenses = c(5000, 8000), 
  from_poi_to_this_person = c(10, 20), # The number of normal messages from POI
  exercised_stock_options = c(10000, 25000), 
  from_messages = c(50, 60), 
  other = c(1000, 3000),
  from_this_person_to_poi = c(5, 10), 
  long_term_incentive = c(5000, 7000), 
  shared_receipt_with_poi = c(300, 500), 
  restricted_stock = c(20000, 30000), 
  director_fees = c(0, 0) # Employees usually do not receive director fees
)


predict(gbm_model,normal_situation)
predict(rf_model,normal_situation)
predict(gbm_model,newdata)
predict(rf_model,newdata)  


#4.GBM
# GBM模型的详细调参
set.seed(123)
gbm_tune_grid <- expand.grid(interaction.depth = c(1, 3, 5),
                             n.trees = seq(50, 200, by = 50),
                             shrinkage = c(0.01, 0.1),
                             n.minobsinnode = c(10, 20))
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
gbm_model <- train(poi ~ ., data = train, method = "gbm",
                   trControl = control, tuneGrid = gbm_tune_grid, verbose = FALSE)

# 使用GBM模型进行预测
predictions_gbm <- predict(gbm_model, newdata = test)

# 计算混淆矩阵
conf_matrix_gbm <- confusionMatrix(predictions_gbm, test$poi)
conf_matrix_gbm
print("GBM Accuracy:")
print(conf_matrix_gbm$overall["Accuracy"])

# 计算其他性能指标
gbm_precision <- conf_matrix_gbm$byClass['Precision']
gbm_recall <- conf_matrix_gbm$byClass['Recall']
gbm_F1 <- 2 * (gbm_precision * gbm_recall) / (gbm_precision + gbm_recall)
print(paste("GBM Precision:", gbm_precision))
print(paste("GBM Recall:", gbm_recall))
print(paste("GBM F1 Score:", gbm_F1))


#6.Random Forest

# 设置随机数种子
set.seed(123)

# 定义需要在模型训练中使用的特征
features <- colnames(train)[-which(colnames(train) == "poi")]

# 训练随机森林模型
rf_model <- train(
  x = train[, features],
  y = train$poi,
  method = 'rf',
  metric = 'Accuracy',
  tuneGrid = expand.grid(mtry = c(2, 4, 6, 8)),
  trControl = trainControl(method = 'cv', number = 10),
  ntree = 500  # 设置固定的树数量
)

# 使用RF模型进行预测
pred_rf <- predict(rf_model, newdata = test)

# 计算混淆矩阵
conf_rf <- confusionMatrix(pred_rf, test$poi)
conf_rf 
print("RF Accuracy:")
print(conf_rf$overall["Accuracy"])

# 计算其他性能指标
rf_precision <- conf_rf$byClass['Precision']
rf_recall <- conf_rf$byClass['Recall']
rf_F1 <- 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)
print(paste("RF Precision:", rf_precision))
print(paste("RF Recall:", rf_recall))
print(paste("RF F1 Score:", rf_F1))









