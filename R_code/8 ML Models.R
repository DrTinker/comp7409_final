
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



#1.SVM

# 设置随机数种子
set.seed(123)

# 创建更细致的调参网格
svm_tune_grid <- expand.grid(
  sigma = seq(0.01, 0.1, length = 10),  # 细化 sigma 范围
  C = 2^(seq(3, 8, length = 10))        # 细化 C 范围
)

# 训练 SVM 模型
svm_model <- train(
  poi ~ .,
  data = train,
  method = "svmRadial",
  trControl = trainControl(method = "cv", number = 10),  # 增加 CV 折数
  tuneGrid = svm_tune_grid
)

# 进行预测并评估性能
predictions_svm <- predict(svm_model, newdata = test)
conf_matrix_svm <- confusionMatrix(predictions_svm, test$poi)
conf_matrix_svm


#2.Logistic Regression


control <- trainControl(method = "cv", number = 10)
set.seed(123)
logit_tune_grid <- expand.grid(alpha = 0:1, # L1和L2正则化混合
  lambda = seq(0.001, 0.1, length = 10)) # 正则化强度
logit_model <- train(poi ~ ., data = train, method = "glmnet",
  trControl = control, tuneGrid = logit_tune_grid)
predictions_logit <- predict(logit_model, newdata = test)
conf_matrix_logit <- confusionMatrix(predictions_logit, test$poi)
conf_matrix_logit
print("Logistic Regression Accuracy:")
print(conf_matrix_logit$overall["Accuracy"])


#3.KNN

set.seed(123)
knn_tune_grid <- expand.grid(k = seq(1, 21, by = 2)) # 奇数的k值通常更好
knn_model <- train(poi ~ ., data = train, method = "knn",
  trControl = control, tuneGrid = knn_tune_grid)
predictions_knn <- predict(knn_model, newdata = test)
conf_matrix_knn <- confusionMatrix(predictions_knn, test$poi)
conf_matrix_knn
print("KNN Accuracy:")
print(conf_matrix_knn$overall["Accuracy"])



#4.GBM
# GBM模型的详细调参
set.seed(123)
gbm_tune_grid <- expand.grid(interaction.depth = c(1, 3, 5),
  n.trees = seq(50, 200, by = 50),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = c(10, 20))
gbm_model <- train(poi ~ ., data = train, method = "gbm",
  trControl = control, tuneGrid = gbm_tune_grid, verbose = FALSE)

predictions_gbm <- predict(gbm_model, newdata = test)
conf_matrix_gbm <- confusionMatrix(predictions_gbm, test$poi)
conf_matrix_gbm
print("GBM Accuracy:")
print(conf_matrix_gbm$overall["Accuracy"])

#5.Decision Tree
# 设置随机数种子
set.seed(123)

# 创建调参网格
dt_tune_grid <- expand.grid(cp = seq(0.001, 0.1, length = 10))

# 训练决策树模型
dt_model <- train(
  poi ~ .,
  data = train,
  method = "rpart",
  trControl = control,
  tuneGrid = dt_tune_grid
)

# 进行预测并评估性能
predictions_dt <- predict(dt_model, newdata = test)
conf_matrix_dt <- confusionMatrix(predictions_dt, test$poi)
conf_matrix_dt
print("Decision Tree Accuracy:")
print(conf_matrix_dt$overall["Accuracy"])

#6.Random Forest

# 设置随机数种子
set.seed(123)

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


# 进行预测并评估性能
pred_rf <- predict(rf_model, newdata = test)
conf_rf <- confusionMatrix(pred_rf, test$poi)
conf_rf 
print(conf_rf$overall["Accuracy"])


#7.Adaboost


ada_model <- train(
  x = train[, features],
  y = train$poi,
  method = "ada",
  metric = "Accuracy",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(
    iter = c(50, 100, 200), 
    maxdepth = c(2,4,6),
    nu = 1
  )
)

bparams <- ada_model$bestTune
pred_ada <- predict(ada_model, newdata = test)
conf_ada <- confusionMatrix(pred_ada, test$poi)
conf_ada
print(conf_ada$overall["Accuracy"])

#8.Neural Network


library(caret)
library(nnet)

# 数据预处理 - 标准化
standardized_train <- as.data.frame(scale(train[, -ncol(train)]))
standardized_train$poi <- train$poi

standardized_test <- as.data.frame(scale(test[, -ncol(test)]))
standardized_test$poi <- test$poi

# 设置训练控制参数
train_control <- trainControl(method = "cv", number = 10)

# 创建调参网格
tune_grid <- expand.grid(size = c(5, 10, 15, 20), decay = c(0.01, 0.001, 0.0001))

# 训练神经网络模型
nnet_model <- train(
  poi ~ .,
  data = standardized_train,
  method = "nnet",
  trControl = train_control,
  tuneGrid = tune_grid,
  trace = FALSE,
  MaxNWts = 10000,
  maxit = 200
)



# 进行预测并评估性能
predictions <- predict(nnet_model, newdata = standardized_test)
conf_matrix <- confusionMatrix(predictions, standardized_test$poi)
conf_matrix
print(conf_matrix$overall["Accuracy"])









