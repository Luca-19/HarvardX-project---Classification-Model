####################
# Data preparation #
####################

# suppress the warnings visualization for the readability of the report
options(warn = -1) 

# upload of the packages
if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)
if (!require(caret)) install.packages('caret')
library(caret)
if (!require(data.table)) install.packages('data.table')
library(data.table)
if (!require(httr)) install.packages('httr')
library(httr)

# dataset url: https://www.kaggle.com/datasets/harbhajansingh21/persistent-vs-nonpersistent
# file download: https://raw.githubusercontent.com/Luca-19/HarvardX-project---Classification-Model/main/Persistent_vs_NonPersistent.csv

dl <- tempfile()

download.file("https://raw.githubusercontent.com/Luca-19/HarvardX-project---Classification-Model/main/Persistent_vs_NonPersistent.csv", dl)
dataset<-read.csv(dl)

# packages required by caret for the models
if (!require(rpart)) install.packages('rpart') #rpart
library(rpart)
if (!require(randomForest)) install.packages('randomForest') #random forest
library(randomForest)
if (!require(kernlab)) install.packages('kernlab') #swm
library(kernlab)
if (!require(glmnet)) install.packages('glmnet') #glmnet
library(glmnet)
if (!require(pROC)) install.packages('pROC') #pROC package used for ROC curve
library(pROC)
if (!require(caretEnsemble)) install.packages('caretEnsemble') #caretEnsemble package for stacking the models
library(caretEnsemble)

#other specific packages
if (!require(fastDummies)) install.packages('fastDummies') # to create dummy variables
library(fastDummies)
if (!require(corrplot)) install.packages('corrplot') # to plot the correlations among predictors
library(corrplot)

# restore the warnings visualization
options(warn=0)

#############################
# Exploratory data analysis #
#############################

# dataset structure
str(dataset)

# overview of the dataset
summary(dataset)

# verify if missing values are present
any(is.na(dataset))

# split between numerical and categorical predictors

datanum<-dataset %>%
  select_if(is.numeric)

# categorical predictors: elimination of categorical predictors without information content
datacat<-dataset %>%
  select_if(negate(is.numeric))
datacat<-datacat %>% select(-c(Ptid,Ntm_Speciality))

##########################
# Categorical predictors #
##########################

### global view of categorical predictors

tidycat<-datacat %>% gather(key=predictors,value=value,-Persistency_Flag)

# summarize for each predictor and category within predictors the percentage of Persistent cases

table<-tidycat %>% group_by(predictors,value) %>% summarize(Persistent_rate=mean(Persistency_Flag=="Persistent"),count=n()) %>%
  group_by(predictors) %>% summarize(min=min(Persistent_rate),max=max(Persistent_rate))

# the percentage of Persistent cases across different categories and predictors
# in order to put in evidence the features that influence more the outcome

table %>% arrange(desc(max))

table %>% mutate(x1 = fct_reorder(predictors, desc(max))) %>% ggplot(aes(x=x1,y=min))+geom_point(color='red')+geom_point(aes(x=x1,y=max),color='blue')+
  geom_segment(aes(x = x1,y = min, xend = x1,yend = max))

# Categorical Predictor analysis: Dexa_During_Rx

datacat %>% ggplot(aes(x=Dexa_During_Rx,fill=as.factor(Persistency_Flag)))+
  geom_bar(aes( y=..count../tapply(..count.., ..x.. ,sum)[..x..]))+
  ylab('Persistency_Flag vs Dexa_During_Rx, %') +
  scale_y_continuous(labels = scales::percent)

datacat %>% group_by(Dexa_During_Rx) %>% summarize(Persistent_rate=mean(Persistency_Flag=="Persistent"))

# Categorical Predictor analysis: Concom_Viral_Vaccines

datacat %>% ggplot(aes(x=Concom_Viral_Vaccines,fill=as.factor(Persistency_Flag)))+
  geom_bar(aes( y=..count../tapply(..count.., ..x.. ,sum)[..x..]))+
  ylab('Persistency_Flag vs Concom_Viral_Vaccines, %') +
  scale_y_continuous(labels = scales::percent)

datacat %>% group_by(Concom_Viral_Vaccines) %>% summarize(Persistent_rate=mean(Persistency_Flag=="Persistent"))

# Categorical Predictor analysis: Comorb_Long_Term_Current_Drug_Therapy

datacat %>% ggplot(aes(x=Comorb_Long_Term_Current_Drug_Therapy,fill=as.factor(Persistency_Flag)))+
  geom_bar(aes( y=..count../tapply(..count.., ..x.. ,sum)[..x..]))+
  ylab('Persistency_Flag vs Comorb_Long_Term_Current_Drug_Therapy, %') +
  scale_y_continuous(labels = scales::percent)

datacat %>% group_by(Comorb_Long_Term_Current_Drug_Therapy) %>% summarize(Persistent_rate=mean(Persistency_Flag=="Persistent"))

########################
# Numerical predictors #
########################

# global view of numerical predictors
y<-as.factor(ifelse(datacat$Persistency_Flag=="Persistent",1,0))
relevance<-cbind(y,datanum)

#boxplot
relevance %>% ggplot(aes(x=y,y=Dexa_Freq_During_Rx))+geom_boxplot()
relevance %>% ggplot(aes(x=y,y=Count_Of_Risks))+geom_boxplot()

#density plot
relevance %>% ggplot(aes(x=Dexa_Freq_During_Rx,fill=y))+geom_density(alpha=0.3,xlim=c(0,50))
relevance %>% ggplot(aes(x=Count_Of_Risks,fill=y))+geom_density(alpha=0.3)

##########################
# Predictors correlation #
##########################

# dummy vars creation
datacat1<-datacat %>% select(-c(Persistency_Flag))
dataf <- dummy_cols(datacat1,remove_selected_columns = TRUE,remove_first_dummy = TRUE)

# re-build of the dataset
cor_set<-cbind(dataf,datanum)

# correlation plot: evidence of the correlation inside the structure
correlation<-cor(cor_set,method="spearman")
corrplot(correlation,tl.pos='n')

# selection of the correlated predictors
highCorr<-findCorrelation(correlation,cutoff=.6)
highCorr_list<-cor_set[,highCorr]
str(highCorr_list)

##################
# Data partition #
##################

set.seed(100, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(100)
Persistency_Flag<-datacat$Persistency_Flag

# re-build of the dataset with outcome
rev_set<-cbind(Persistency_Flag,dataf,datanum)
rev_set$Persistency_Flag<-factor(rev_set$Persistency_Flag)

# Validation set will be 20% of dataset
split<-createDataPartition(rev_set$Persistency_Flag,p=.8,times=1,list=FALSE)
training<-rev_set[split,]
test<-rev_set[-split,]

# split between outcome and features in the training set
outcome<-ifelse(training$Persistency_Flag=="Non-Persistent","Non.Persistent","Persistent")
features<-training %>% select(-c(Persistency_Flag))

#replace the term Non-Persistent with Non.Persistent
test$Persistency_Flag<-factor(ifelse(test$Persistency_Flag=="Non-Persistent","Non.Persistent","Persistent"))

###################
## Model analysis #
###################

# fit control for the function train
fit.control <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)

# suppress the warnings for better readability of the report
options(warn=-1)

# 1. model GLM ######################

set.seed(7, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(7)
# model GLM training on the train set
model_glm<-train(x=features,y=outcome,method="glm",family="binomial",trControl = fit.control)

# prediction on the test set
y_pred_glm<-factor(predict(model_glm,test))
# accuracy calculation
accuracy_glm<-confusionMatrix(y_pred_glm,test$Persistency_Flag) $overall["Accuracy"]

# select sig. variables
toselect.x <- summary(model_glm)$coeff[-1,4] < 0.05
relevant.x <- names(toselect.x)[toselect.x == TRUE]
# show sig. variables
relevant.x

#2. model RPART #####################

set.seed(15, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(15)
# model RPART training on the train set - cross-validation to choose cp
model_rpart<-train(x=features,y=outcome,method="rpart",
                   tuneGrid=data.frame(cp=seq(0,0.05,len=25)),
                   trControl = fit.control)

# plot cp optimization
plot(model_rpart)
# summary model rpart
model_rpart

# prediction on the test set
y_pred_rpart<-predict(model_rpart,test)
# accuracy calculation
accuracy_rpart<-confusionMatrix(y_pred_rpart,test$Persistency_Flag) $overall["Accuracy"] 

#3. model RANDOM FOREST ##############

set.seed(70, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(70)

# tune grid with mtry=5,10,15
grid<-expand.grid(.mtry=c(5,10,15))

# model RANDOM FOREST training on the train set
# this line takes a long time to run
model_rf<-train(x=features,y=outcome,method="rf",trControl = fit.control,tuneGrid=grid)

# plot the tuning of mtry (number of the variables randomly sampled as candidates at each split)
ggplot(model_rf)

#best tune model
model_rf$bestTune

# prediction on the test set
y_pred_rf<-predict(model_rf,test)
# accuracy calculation
accuracy_rf<-confusionMatrix(y_pred_rf,test$Persistency_Flag) $overall["Accuracy"]

# main important features
important_var<-varImp(model_rf)
main_var<-data.frame(important_var$importance) %>% arrange(desc(Overall)) %>% top_n(30)
relevant.xrf<-row.names(main_var)

#4. model SVM #######################

set.seed(87, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(87)
training_set<-cbind(outcome,features)

# model SVM training on the train set
# this line takes a long time to run
# tuning cost parameter with tuneLength argument

model_svm<-train(outcome~.,data=training_set,method="svmRadial",trControl = fit.control,tuneLength=10)

# plot cost parameter optimization
plot(model_svm)

#best tune model
model_svm$bestTune

# prediction on the test set
y_pred_svm<-predict(model_svm,test,type="raw")
# accuracy calculation
accuracy_svm<-confusionMatrix(y_pred_svm,test$Persistency_Flag) $overall["Accuracy"] 

#5. model GLMNET ####################

set.seed(300, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(300)

# grid for tuning parameters (alpha and lambda)
glmnGrid<-expand.grid(.alpha=c(0,0.1,0.2,0.4,0.6,0.8,1),
                      .lambda=seq(.01,.2,length=20))

# model GLMNET training on the train set
model_glmnet<-train(x=features,y=outcome,method="glmnet",trControl = fit.control,tuneGrid=glmnGrid,family="binomial")

# plot the result of tuning
ggplot(model_glmnet)

model_glmnet$bestTune

# prediction on the test set
y_pred_glmnet<-predict(model_glmnet,test)
# accuracy calculation
accuracy_glmnet<-confusionMatrix(y_pred_glmnet,test$Persistency_Flag) $overall["Accuracy"] 

#################################
### Model evaluation - Accuracy #
#################################

# restore the warnings visualization
options(warn=0)

# table with the accuracy results
accuracy_results<-data.frame(method=c("glm","rpart",
                                  "rf",
                                  "svm","glmnet"),
                         Accuracy=c(accuracy_glm,accuracy_rpart,accuracy_rf,accuracy_svm,accuracy_glmnet))

knitr::kable(accuracy_results)

# join the main factors
relevant.xrf<-data.frame(relevant.xrf)
colnames(relevant.xrf)<-"relevant.x"

# intersect the main factors for glm and random forest
inner_join(data.frame(relevant.x),relevant.xrf,by="relevant.x")

############################
### Model evaluation - ROC #
############################

test_y<-ifelse(test$Persistency_Flag=="Persistent",1,0)

#1. ROC glm
test_pred_glm<-predict(model_glm,test,type="prob")
# plot ROC curve for glm model
roc_mod_glm = roc(test_y, test_pred_glm$Persistent,levels = c(0, 1), direction = "<")
plot(roc_mod_glm, main="ROC curve -- Model comparison ",legacy.axes=TRUE)

#2. ROC rpart
test_pred_rpart<-predict(model_rpart,test,type="prob")
# plot ROC curve for rpart model
roc_mod_rpart = roc(test_y, test_pred_rpart$Persistent,levels = c(0, 1), direction = "<")
lines(roc_mod_rpart,col="blue")

#3. ROC random forest
test_pred_rf<-predict(model_rf,test,type="prob")
# plot ROC curve for random forest
roc_mod_rf<- roc(test_y, test_pred_rf$Persistent,levels = c(0, 1), direction = "<")
lines(roc_mod_rf,col="red")

#4. ROC svm
test_pred_svm<-predict(model_svm,test,type="prob")
# plot ROC curve for svm model
roc_mod_svm<- roc(test_y, test_pred_svm$Persistent,levels = c(0, 1), direction = "<")
lines(roc_mod_svm,col="green")

#5. ROC glmnet
test_pred_glmnet<-predict(model_glmnet,test,type="prob")
# plot ROC curve for glmnet model
roc_mod_glmnet<- roc(test_y, test_pred_glmnet$Persistent,levels = c(0, 1), direction = "<")
lines(roc_mod_glmnet,col="light blue")

# add legend 
legend(0.7,0.45, c('glm','rpart','rf','svm','glmnet'),lty=c(1,1),
       lwd=c(2,2),col=c('black','blue','red','green','light blue'))

# Area under curve ROC
AUC_glm<-auc(roc_mod_glm)
AUC_rpart<-auc(roc_mod_rpart)
AUC_rf<-auc(roc_mod_rf)
AUC_svm<-auc(roc_mod_svm)
AUC_glmnet<-auc(roc_mod_glmnet)
AUC_results<-c(AUC_glm,AUC_rpart,AUC_rf,AUC_svm,AUC_glmnet)

############################################
# Recap of AUC, accuracy and other metrics #
############################################

results<-cbind(accuracy_results,AUC_results)
knitr::kable(results)

# Calculation of Sensitivity and Specificity

glm<-confusionMatrix(y_pred_glm,test$Persistency_Flag,
                     positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]
rpart<-confusionMatrix(y_pred_rpart,test$Persistency_Flag,
                       positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]
rf<-confusionMatrix(y_pred_rf,test$Persistency_Flag,
                    positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]
svm<-confusionMatrix(y_pred_svm,test$Persistency_Flag,
                     positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]
glmnet<-confusionMatrix(y_pred_glmnet,test$Persistency_Flag,
                        positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]

cm_results<-data.frame(rbind(glm,rpart,rf,svm,glmnet))

# final table
results<-cbind(results[,2:3],cm_results)
knitr::kable(results)

############################################
### Ensemble the results and further steps #
############################################

### Ensemble the results using the average of the class probabilities
y_ensemble<-ifelse((test_pred_glm$Persistent+test_pred_rpart$Persistent+test_pred_rf$Persistent+
                      test_pred_svm$Persistent+test_pred_glmnet$Persistent)/5>0.5,1,0)
y_pred_agg<-factor(ifelse(y_ensemble==1,"Persistent","Non.Persistent"))

# Accuracy calculation
accuracy_agg<-confusionMatrix(y_pred_agg,test$Persistency_Flag) $overall["Accuracy"]
accuracy_agg

# Area under curve ROC
test_pred_agg<-(test_pred_glm$Persistent+test_pred_rpart$Persistent+test_pred_rf$Persistent+
  test_pred_svm$Persistent+test_pred_glmnet$Persistent)/5
AUC_mod_agg<- auc(roc(test_y, test_pred_agg,levels = c(0, 1), direction = "<"))

# Calculation of Sensitivity and Specificity
sens_agg<-confusionMatrix(y_pred_agg,test$Persistency_Flag,
                        positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]

# performance for simple ensemble using the average of the results of the models
ensemble_avg_mod<-c(accuracy_agg,AUC_mod_agg,sens_agg)


##############################
### caret ensemble: stacking #
##############################

set.seed(77, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(77)

# suppress the warnings for better readability of the report
options(warn=-1)

# train parameters:
# resampling method = cv
# classProbs = TRUE: class probabilities are computed for classification models
# twoClassSummary computes sensitivity, specificity and the area under the ROC curve

my_control <- trainControl(method = 'cv', # for "cross-validation"
                           number = 20, # number of k-folds
                           savePredictions = 'final',
                           summaryFunction = twoClassSummary, classProbs = TRUE,
                           allowParallel = TRUE)

set.seed(123, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(123)

# list of the base algoritm - training on the training set
# we use the models with the parameters already tuned before

model_list <- caretList(outcome~.,data=training_set,
                        trControl = my_control,
                        methodList = NULL,
                        tuneList = list(
                          glmnet1=caretModelSpec(method='glmnet', tuneGrid=data.frame(alpha=0,lambda=0.08)),
                          svm1=caretModelSpec(method='svmRadial', tuneGrid=data.frame(sigma=0.008001509,C=0.25)),
                          rf1=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=5))))

set.seed(85, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(85)

# model stacking with GLM ###############

glm_ensemble <- caretStack(model_list, 
                         method = 'glm', 
                         metric = 'ROC', 
                         trControl = my_control)
# model summary
summary(glm_ensemble)

# prediction on the test set and calculation of the accuracy of the model
predict_ens_glm <- predict(glm_ensemble, newdata = test)

accuracy_ens<-confusionMatrix(predict_ens_glm,test$Persistency_Flag) $overall["Accuracy"]
accuracy_ens

# coefficient for the base models 
CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
CF/sum(CF)

# Area under curve ROC
model_preds <- predict(glm_ensemble, newdata=test, type="prob")
auc_mod_ens<- auc(roc(test_y, model_preds))

# Calculation of Sensitivity and Specificity on the test set
sens_ens<-confusionMatrix(predict_ens_glm,test$Persistency_Flag,
                        positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]

# performance for ensemble via stacking glm
stacking_glm<-c(accuracy_ens,auc_mod_ens,sens_ens)

### model stacking with random forest #####

set.seed(72, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(72)

# model stacking with random forest - caretStack
rf_ensemble <- caretStack(model_list, 
                           method = 'rf', 
                           metric = 'ROC', 
                           trControl = my_control)

# prediction on the test set and calculation of the accuracy of the model
predict_ens_rf <- predict(rf_ensemble, newdata = test)
accuracy_ens_rf<-confusionMatrix(predict_ens_rf,test$Persistency_Flag) $overall["Accuracy"]
accuracy_ens_rf

# Area under curve ROC
model_preds_rf <- predict(rf_ensemble, newdata=test, type="prob")
auc_mod_ens_rf<- auc(roc(test_y, model_preds_rf))

# Calculation of Sensitivity and Specificity on the test set
sens_ens_rf<-confusionMatrix(predict_ens_rf,test$Persistency_Flag,
                          positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]

# performance for ensemble via stacking rf
stacking_rf<-c(accuracy_ens_rf,auc_mod_ens_rf,sens_ens_rf)

# table of results updated ###############

# results summary
results<-rbind(results,ensemble_avg_mod,stacking_glm,stacking_rf)
rownames(results)<-NULL
final_results<-data.frame(model=c("glm","rpart","rf","svm","glmnet",
                                   "ensemble (average)","stacking with glm","stacking with rf"),
                             results)
knitr::kable(final_results)

###################################
### trial to improve sensitivity  #
###################################

# function to define a different set of performance measures 
# Persistent is the level 2, while Non.Persistent is the level 1

fourStats<-function(data,lev=levels(data$obs),model=NULL)
  {
  accKapp<-postResample(data[,"pred"],data[,"obs"])
  out<-c(accKapp,
         sensitivity(data[,"pred"],data[,"obs"],lev[2]),
         specificity(data[,"pred"],data[,"obs"],lev[1]))
  names(out)[3:4]<-c("Sens","Spec")
  out
}

# different trainControl setting coherent with
# the new set of performance measures

ctrl_sens<-trainControl(method="cv",
                   classProbs=TRUE,
                   summaryFunction=fourStats,
                   verboseIter=FALSE)

# cost sensitive training matrix (FN=2.2,FP=1)
# reference value on the rows and predictions on the columns

costMatrix<-matrix(c(0,2.2,1,0),ncol=2)
rownames(costMatrix)<-levels(as.factor(outcome))
colnames(costMatrix)<-levels(as.factor(outcome))

set.seed(1103, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1103)

# model rpart with costMatrix training on the train set
model_rpart_sens<-train(x=features,y=outcome,method="rpart",
                   metric="Kappa",
                   trControl = ctrl_sens,
                   parms=list(loss=costMatrix))

model_rpart_sens

# prediction on the test set
y_pred_rpart_sens<-predict(model_rpart_sens,test)
# accuracy calculation
accuracy_rpart_sens<-confusionMatrix(y_pred_rpart_sens,test$Persistency_Flag) $overall["Accuracy"]
accuracy_rpart_sens

# Area under curve ROC
model_preds_rpart_sens <- predict(model_rpart_sens, newdata=test, type="prob")
auc_mod_rpart_sens<- auc(roc(test_y, model_preds_rpart_sens$Persistent))

# sensitivity-specificity calculation
rpart_sens<-confusionMatrix(y_pred_rpart_sens,test$Persistency_Flag,
                            positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]
rpart_with_cost_matrix<-c(accuracy_rpart_sens,auc_mod_rpart_sens,rpart_sens)

# trial with a different cost matrix ############
# change the cost matrix with FN=3 ##############

costMatrix2<-matrix(c(0,3,1,0),ncol=2)
rownames(costMatrix2)<-levels(as.factor(outcome))
colnames(costMatrix2)<-levels(as.factor(outcome))

set.seed(1109, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1109)

# model rpart with costMatrix training on the train set
model_rpart_sens_2<-train(x=features,y=outcome,method="rpart",
                        metric="Kappa",
                        trControl = ctrl_sens,
                        parms=list(loss=costMatrix2))

# prediction on the test set
y_pred_rpart_sens_2<-predict(model_rpart_sens_2,test)

# sensitivity-specificity calculation
rpart_sens2<-confusionMatrix(y_pred_rpart_sens_2,test$Persistency_Flag,
                            positive="Persistent")$byClass[(c("Sensitivity","Specificity"))]
rpart_sens2

######################
### FINAL RESULTS  ###
######################

# final recap
last<-final_results %>% add_row(model = "rpart -cost matr", Accuracy = accuracy_rpart_sens,
                                AUC_results=auc_mod_rpart_sens[1],Sensitivity=rpart_sens[1],
                                Specificity=rpart_sens[2])

knitr::kable(last)

# substitute the results with ranking of the relative column/metric

rank_last<-last %>% mutate(Accuracy_r = rank(-Accuracy, ties.method = 'first')) %>%
  mutate(AUC_r = rank(-AUC_results, ties.method = 'first')) %>%
  mutate(Sensitivity_r = rank(-Sensitivity, ties.method = 'first')) %>%
  mutate(Specificity_r = rank(-Specificity, ties.method = 'first')) %>%
  select(model,Accuracy_r,AUC_r,Sensitivity_r,Specificity_r)

# transformation of the table in a tidy format in order to make a heatmap 
# with the ranking of the results for each metric

tidylast<-rank_last %>% gather(key=metric,value=value,-c(model))

# plot the heatmap of the ranking for each metric
ggplot(tidylast, aes(x = metric, y = model, fill = value)) + geom_tile()+
  geom_text(aes(label = value))+
  scale_fill_gradient(high = "white", low = "green")+
  ggtitle("Evaluation Grid: model ranking for each metric")

######################
######################
