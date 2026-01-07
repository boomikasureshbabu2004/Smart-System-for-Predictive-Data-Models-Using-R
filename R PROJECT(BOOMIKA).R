
#----------------------------------- R PROJECT ------------------------------------
dataset = read.csv("~/Data Science/Dataset/loan_dataset.csv")
View(dataset)
dataset <- dataset [-c(4)]
dataset <- dataset[2:12]

#EDA:
dim(dataset)
str(dataset)
head(dataset)
summary(dataset)
colSums(is.na(dataset))

#ENCODING:
dataset$Gender = factor(dataset$Gender, 
                        levels = c('Male','Female'), labels = c(1,2))
dataset$Married = factor(dataset$Married, 
                         levels = c('No','Yes'),labels = c(1,2))
dataset$Education = factor(dataset$Education, 
                           levels = c('Graduate','Not Graduate'),labels = c(1,2))
dataset$Self_Employed = factor(dataset$Self_Employed, 
                               levels = c('No','Yes'),labels = c(1,2))
dataset$Property_Area = factor(dataset$Property_Area, 
                               levels = c('Urban','Rural','Semiurban'),labels = c(1,2,3))
dataset$Loan_Status = factor(dataset$Loan_Status, levels = c('Y','N'))

#HANDLING NULL VALUES:
library(DescTools)
dataset$Gender[is.na(dataset$Gender)] <- Mode(na.omit(dataset$Gender))
dataset$Married[is.na(dataset$Married)] <- Mode(na.omit(dataset$Married))
dataset$Self_Employed[is.na(dataset$Self_Employed)] <- Mode(na.omit(dataset$Self_Employed))
dataset$LoanAmount[is.na(dataset$LoanAmount)] <- median(na.omit(dataset$LoanAmount))
dataset$Loan_Amount_Term[is.na(dataset$Loan_Amount_Term)] <- median(na.omit(dataset$Loan_Amount_Term))
dataset$Credit_History[is.na(dataset$Credit_History)] <- median(na.omit(dataset$Credit_History))

colSums(is.na(dataset))

#CONVERT INTO NUMERIC:
dataset$Gender <- as.numeric(dataset$Gender)
dataset$Married <- as.numeric (dataset$Married)
dataset$Education <- as.numeric(dataset$Education)
dataset$Self_Employed <- as.numeric(dataset$Self_Employed)
dataset$Property_Area <- as.numeric(dataset$Property_Area)

colSums(is.na(dataset))

#HISTOGRAM:
par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))

# Numeric features
numeric_features <- c("ApplicantIncome", "CoapplicantIncome", "LoanAmount")
for (feature in numeric_features) {
  hist(dataset[[feature]],
       main = paste("Histogram of", feature),
       xlab = feature,
       col = "pink",
       border = "black")
}
# Categorical features
categorical_features <- c("Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status")
for (feature in categorical_features) {
  counts <- table(dataset[[feature]])
  barplot(counts,
          main = paste("Bar plot of", feature),
          col = "skyblue",
          border = "black")
}

#RESET LAYOUT:
par(mfrow = c(1,1))

#SPLITTING A DATA:
library(caTools)
set.seed(123)
split = sample.split(dataset$Loan_Status, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#FEATURE SCALING:
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

#MODEL FITTING:

#----------------------------------------------------------
#LOGISTIC REGRESSION:
dataset$Loan_Status = factor(dataset$Loan_Status, levels = c('Y','N'), labels = c(0,1))
dataset$Loan_Status <- factor(dataset$Loan_Status, levels = c(0,1))

log_reg = glm(formula = Loan_Status ~ .,#output
                 family = binomial,
                 data = training_set)
summary(log_reg)
prob_pred = predict(log_reg, type = 'response',newdata = test_set)
prob_pred

y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred

cm = table(test_set[, 11], y_pred)
cm

Accuracy=sum(diag(cm)/sum(cm))
Accuracy
#0.8278689
#----------------------------------------------------------
#CROSS VALIDATION:
library(caret)
library(kernlab)
library(e1071)  

set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 20, repeats = 15)
model <- train(Loan_Status ~., data = dataset, method = "glm",
               trControl = train.control)

print(model)
#Accuracy   Kappa    
#0.8105661  0.4787004
#----------------------------------------------------------
#SUPPORT VECTOR MACHINE (SVM):
library(e1071)        
svm_classifier = svm(formula = Loan_Status ~ .,
                      data = training_set,
                      type = 'C-classification',
                      kernel = 'linear')

y_pred = predict(svm_classifier, newdata = test_set)

cm = table(test_set$Loan_Status, y_pred)
cm
accuracy = sum(diag(cm)) / sum(cm)
accuracy 
#0.8360656
#----------------------------------------------------------
#CROSS VALIDATION:
library(caret)
library(kernlab)
library(e1071)  

set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3)
model <- train(Loan_Status ~., data = dataset, method = "svmLinear",
               trControl = train.control)
print(model)
#Accuracy   Kappa    
#0.8094799  0.4768601
#----------------------------------------------------------
#DECISION TREE:
library(tree)#new method
classifier = tree(formula = Loan_Status ~ .,
                  data = training_set)

y_pred = predict(classifier, newdata = test_set, type = 'class')
y_pred

cm = table(test_set[, 11], y_pred)
cm
#y_pred
# Y  N
# Y 83  1
# N 19 19

Accuracy=sum(diag(cm)/sum(cm))
Accuracy
#0.8360656
#----------------------------------------------------------
#CROSS VALIDATION:
library(caret)
library(kernlab)
library(e1071)  

set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3)
model <- train(Loan_Status ~., data = dataset, method = "rpart",
               trControl = train.control)
print(model)
#cp           Accuracy   Kappa    
#0.002604167  0.7611371  0.4083265
#0.009114583  0.7959153  0.4508818
#0.390625000  0.7432010  0.2254268

#----------------------------------------------------------
#PRINCIPAL COMPONENT ANALYSIS(PCA):
library(caTools)
library(caret)
library(e1071)
library(Rfast)
library(factoextra)

pca_components <- 2
pca_model <- preProcess(x = training_set[-11], method = 'pca', pcaComp = pca_components)
training_set_pca <- predict(pca_model, training_set)
test_set_pca <- predict(pca_model, test_set)

#----------------------------------------------------------
#VISUALIZATION:
library(plotly)
pca_plot <- plot_ly(
  data = training_set_pca,
  x = ~PC1,
  y = ~PC2,
  color = ~Loan_Status,
  colors = c("red", "blue"),  
  type = "scatter",
  mode = "markers",
  marker = list(size = 6, opacity = 0.7, symbol = "circle") 
)
pca_plot <- pca_plot %>% layout(
  title = "PCA Components Scatter Plot",
  xaxis = list(title = "PC1"),
  yaxis = list(title = "PC2"),
  legend = list(title = list(text = "Loan Status"))
)
pca_plot

#----------------------------------------------------------
#DATASET REPORT
library(DataExplorer)
create_report(data)

#----------------------------------------------------------
#                       CROSS VALIDATION
#ACCURACY                Before      After 
#LOGISTIC REGRESSION :  0.8278689    0.8105661  
#SVM                 :  0.8360656    0.8094799  
#DECISION TREE       :  0.8360656    0.7432010   
