rm(list=ls())

pacman::p_load(MatchIt, postDoubleR, data.table, glmnet, tidyverse, gbm, randomForest, party, mlr3, mlr3learners, DoubleML,
               paradox, mlr3tuning, e1071, nnet, Metrics, grf, dplyr)

#loading and a bit of feature engineering

setwd("C:/Users/syedm/Desktop/Causal ML")
data_init <- load("36174-0001-Data.rda")
data_init <- da36174.0001
data_init <- data_init[!is.na(data_init$CHANGE),]
data_init$treat <- ifelse(data_init$ASSIGNUB == "y", 1, 0) #z for skip, y for breakfast

data_init %>% select(W_KG, H_CM, AGE, SEX, ETHFAC, SITE, INITUB, treat, CHANGE) -> data

data$SEX <- ifelse(data$SEX=="(0) Female", 0, 1)

mapply(function(x, y) data[x, y] <<- 1, seq(nrow(data)), 
       paste0("ETHFAC_", data$ETHFAC))

mapply(function(x, y) data[x, y] <<- 1, seq(nrow(data)), 
       paste0("SITE_", data$SITE))

mapply(function(x, y) data[x, y] <<- 1, seq(nrow(data)), 
       paste0("INITUB_", data$INITUB))

names(data)[names(data) == "INITUB_aSkipper      "] <- "INITUB_aSkipper"

data[, 10:20][is.na(data[, 10:20])] <- 0

#baseline results

X <- as.matrix(data[c("W_KG", "H_CM", "AGE", "SEX", "ETHFAC_a", "ETHFAC_b", "ETHFAC_c", "ETHFAC_d", 
                      "ETHFAC_e", "SITE_a", "SITE_b", "SITE_c", "SITE_e",
                      "INITUB_BreakfastEater", "INITUB_aSkipper")])
Y <- data$CHANGE
W <- data$treat

reg1 <- lm(Y ~ W + X, data=data)
summary(reg1)

TAU <- summary(reg1)$coefficients[2, 1]

###################### Double ML ##########################

seed <- c(1,5,10,25,50)

data_dt <- as.data.table(data)

list_estimate_dml<-list()
list_error_dml<-list()

no_methods=10
estimate=matrix(NA,2,no_methods)
error=matrix(NA,2,no_methods)

for(i in 1:length(seed)){
  
  n_vars <- ncol(data_dt)-2
  data_dt <- as.data.table(data)
  n_obs <- nrow(data_dt)
  
  #double ml - lasso/lasso #Uses postDoubleR package
  
  set.seed(seed[i])
  
  doubleml_data = DoubleMLData$new(data_dt,
                                   y_col = "CHANGE",
                                   d_cols = "treat",
                                   x_cols = c("W_KG", "H_CM", "AGE", "SEX", "ETHFAC_a", "ETHFAC_b", "ETHFAC_c", "ETHFAC_d", 
                                              "ETHFAC_e", "SITE_a", "SITE_b", "SITE_c", "SITE_e",
                                              "INITUB_BreakfastEater", "INITUB_aSkipper"))
  
  lgr::get_logger("mlr3")$set_threshold("warn")
  lgr::get_logger("bbotk")$set_threshold("warn")
  
  learner_classif_glm = lrn("classif.glmnet")
  learner_regr_glm = lrn("regr.glmnet")
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_glm, ml_m=learner_classif_glm)
  
  doubleml_plr$fit()
  
  estimate[1,1]=doubleml_plr$coef
  estimate[2,1]=doubleml_plr$se
  
  error[1,1] = mae(TAU, doubleml_plr$coef)
  error[2,1] = rmse(TAU, doubleml_plr$coef)/mean(TAU)
  
  #double ml - random forest/logit
  
  set.seed(seed[i])
  
  learner_classif_logreg = lrn("classif.log_reg")
  learner_regr_ranger = lrn("regr.ranger", num.trees = 100, mtry = floor(sqrt(n_vars)), min.node.size = 2,
                            max.depth = 6)
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_ranger, ml_m=learner_classif_logreg)
  doubleml_plr$fit()
  
  estimate[1,2]=doubleml_plr$coef
  estimate[2,2]=doubleml_plr$se
  
  error[1,2] = mae(TAU, doubleml_plr$coef)
  error[2,2] = rmse(TAU, doubleml_plr$coef)/mean(TAU)
  
  #double ml - random forest/neural network
  
  set.seed(seed[i])
  
  lgr::get_logger("mlr3")$set_threshold("warn")
  lgr::get_logger("bbotk")$set_threshold("warn")
  
  learner_classif_nn = lrn("classif.nnet", maxit=100, size=c(5))
  learner_regr_ranger = lrn("regr.ranger", num.trees = 100, mtry = floor(sqrt(n_vars)), min.node.size = 2,
                            max.depth = 10)
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_ranger, ml_m=learner_classif_nn)
  
  doubleml_plr$fit()
  
  estimate[1,3]=doubleml_plr$coef
  estimate[2,3]=doubleml_plr$se
  
  error[1,3] = mae(TAU, doubleml_plr$coef)
  error[2,3] = rmse(TAU, doubleml_plr$coef)/mean(TAU)
  
  #double ml - random forest/lasso
  
  set.seed(seed[i])
  
  learner_regr_ranger = lrn("regr.ranger", num.trees=100, mtry=floor(sqrt(n_vars)), max.depth=5, min.node.size=2)
  learner_classif_glm = lrn("classif.glmnet")
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_ranger, ml_m=learner_classif_glm)
  doubleml_plr$fit()
  
  estimate[1,4]=doubleml_plr$coef
  estimate[2,4]=doubleml_plr$se
  
  error[1,4] = mae(TAU, doubleml_plr$coef)
  error[2,4] = rmse(TAU, doubleml_plr$coef)/mean(TAU)
  
  #double ml - random forest/random forest
  
  set.seed(seed[i])
  
  n_rep = 10
  n_folds = 10
  
  learner_regr_ranger = lrn("regr.ranger", num.trees=100, mtry=floor(sqrt(n_vars)), max.depth=5, min.node.size=2)
  learner_classif_ranger = lrn("classif.ranger", num.trees=100, mtry=floor(sqrt(n_vars)), max.depth=5, min.node.size=2)
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_ranger, ml_m=learner_classif_ranger, n_rep=n_rep, n_folds=n_folds)
  doubleml_plr$fit()
  
  estimate[1,5]=doubleml_plr$coef
  estimate[2,5]=doubleml_plr$se
  
  error[1,5] = mae(TAU, doubleml_plr$coef)
  error[2,5] = rmse(TAU, doubleml_plr$coef)/mean(TAU)
  
  #double ml - lasso/neural network
  
  set.seed(seed[i])
  
  learner_classif_nn = lrn("classif.nnet", maxit=100, size=c(5))
  learner_regr_glm = lrn("regr.glmnet")
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_glm, ml_m=learner_classif_nn)
  doubleml_plr$fit()
  
  estimate[1,6]=doubleml_plr$coef
  estimate[2,6]=doubleml_plr$se
  
  error[1,6] = mae(TAU, doubleml_plr$coef)
  error[2,6] = rmse(TAU, doubleml_plr$coef)/mean(TAU)
  
  #double ml - xgboost/xgboost
  
  set.seed(seed[i])
  
  learner_classif_xgb = lrn("classif.xgboost", nrounds = c(100), max_depth = c(10), min_child_weight =1,
                            subsample = 1, gamma = 0.05,  colsample_bytree = 0.8, eta = c(0.1))
  
  learner_regr_xgb = lrn("regr.xgboost", nrounds = c(100), max_depth = c(10), min_child_weight =1,
                         subsample = 1, gamma = 0.05,  colsample_bytree = 0.8, eta = c(0.1))
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_xgb, ml_m=learner_classif_xgb)
  doubleml_plr$fit()
  
  estimate[1,7]=doubleml_plr$coef
  estimate[2,7]=doubleml_plr$se
  
  error[1,7] = mae(TAU, doubleml_plr$coef)
  error[2,7] = rmse(TAU, doubleml_plr$coef)/mean(TAU)
  
  #double ml - ols/logit
  
  set.seed(seed[i])
  
  learner_regr_lm = lrn("regr.lm") 
  learner_classif_logreg = lrn("classif.log_reg")
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_lm, ml_m=learner_classif_logreg)
  doubleml_plr$fit()
  #print(obj_dml_plr_bonus)
  
  estimate[1,8]=doubleml_plr$coef
  estimate[2,8]=doubleml_plr$se
  
  error[1,8] = mae(TAU, doubleml_plr$coef)
  error[2,8] = rmse(TAU, doubleml_plr$coef)/mean(TAU)
  
  #double ml - lasso/lasso #Uses DoubleML package
  
  X <- as.matrix(data[c("W_KG", "H_CM", "AGE", "SEX", "ETHFAC_a", "ETHFAC_b", "ETHFAC_c", "ETHFAC_d", 
                        "ETHFAC_e", "SITE_a", "SITE_b", "SITE_c", "SITE_e",
                        "INITUB_BreakfastEater", "INITUB_aSkipper")])
  Y <- data$CHANGE
  W <- data$treat
  
  result<-double_ML(X,Y,W,method="glmnet", #was ols before
                    k.fld=2, simulations=5,
                    seed.use=seed[i],
                    validate.inputs = TRUE)
  
  estimate[1,9]=result$`ATE/APE`
  estimate[2,9]=result$Std.Err
  
  error[1,9] = mae(TAU, result$`ATE/APE`)
  error[2,9] = rmse(TAU, result$`ATE/APE`)/mean(TAU)
  
  #causal forest
  
  set.seed(seed[i])
  
  c.forest <- causal_forest(X, as.vector(Y), as.vector(W))
  
  #ATE
  ate <- average_treatment_effect(c.forest, target.sample = "all")
  
  estimate[1,10]=ate[[1]]
  estimate[2,10]=ate[[2]]
  
  error[1,10] = mae(TAU, ate[[1]])
  error[2,10] = rmse(TAU, ate[[1]])/mean(TAU)
  
  list_estimate_dml[[i]] <- estimate
  list_error_dml[[i]] <- error
  
}


df_dmltable <- as.data.frame(apply(simplify2array(list_estimate_dml), 1:2, mean))
df_dmltable


########################### PSM ML ############################

pacman::p_load(haven, dplyr, sandwich, lmtest, MatchIt, Metrics, survey, grf, twang, dplyr)

X <- as.matrix(data[c("W_KG", "H_CM", "AGE", "SEX", "ETHFAC_a", "ETHFAC_b", "ETHFAC_c", "ETHFAC_d", 
                      "ETHFAC_e", "SITE_a", "SITE_b", "SITE_c", "SITE_e",
                      "INITUB_BreakfastEater", "INITUB_aSkipper")])
Y <- data$CHANGE
W <- data$treat

list_estimate_psm<-list()
list_error_psm<-list()

no_methods=4
estimate=matrix(NA,2,no_methods)
error=matrix(NA,2,no_methods)

for(i in 1:length(seed)){
  
  #probit
  
  set.seed(seed[i])
  
  m.out1 <- matchit(W ~ X, data = data,
                    method = "full", distance = "glm", link = "probit")
  
  m.data1 <- match.data(m.out1)
  
  fit1 <- lm(Y ~ W + X, data = m.data1, weights = weights)
  
  c <- coeftest(fit1, vcov. = vcovCL, cluster = ~subclass)
  
  estimate[1,1]=c[[2]]
  estimate[2,1]=c[[2,2]]
  
  error[1,1] = mae(TAU, c[[2]])
  error[2,1] = rmse(TAU, c[[2]])/mean(TAU)
  
  #nnet
  
  set.seed(seed[i])
  
  m.out1 <- matchit(W ~ X, data = data,
                    method = "full", distance = "nnet", distance.options=list(size=5))
  
  m.data1 <- match.data(m.out1)
  
  fit1 <- lm(Y ~ W + X, data = m.data1, weights = weights)
  
  c <- coeftest(fit1, vcov. = vcovCL, cluster = ~subclass)
  
  estimate[1,2]=c[[2]]
  estimate[2,2]=c[[2,2]]
  
  error[1,2] = mae(TAU, c[[2]])
  error[2,2] = rmse(TAU, c[[2]])/mean(TAU)
  
  #rf
  
  n_vars <- ncol(X)
  n_obs <- nrow(X)
  
  set.seed(seed[i])
  
  m.out1 <- matchit(treat ~ W_KG + H_CM + AGE + SEX + ETHFAC_a + ETHFAC_b + ETHFAC_c + ETHFAC_d + ETHFAC_e + SITE_a + 
                      SITE_b + SITE_c + SITE_e + INITUB_BreakfastEater + INITUB_aSkipper, data = data,
                    method = "full", distance = "randomforest", distance.options=list(num.trees = 100, mtry = floor(sqrt(n_vars)), 
                                                                                      min.node.size = 2,
                                                                                      max.depth = 10))
  
  m.data1 <- match.data(m.out1)
  
  fit1 <- lm(Y ~ W + X, data = m.data1, weights = weights)
  
  c <- coeftest(fit1, vcov. = vcovCL, cluster = ~subclass)
  
  estimate[1,3]=c[[2]]
  estimate[2,3]=c[[2,2]]
  
  error[1,3] = mae(TAU, c[[2]])
  error[2,3] = rmse(TAU, c[[2]])/mean(TAU)
  
  #xgb
  
  set.seed(seed[i])
  
  ps.xgb = ps(treat ~ W_KG + H_CM + AGE + SEX + ETHFAC_a + ETHFAC_b + ETHFAC_c + ETHFAC_d + ETHFAC_e + SITE_a + 
                SITE_b + SITE_c + SITE_e + INITUB_BreakfastEater + INITUB_aSkipper, 
              data = data, n.trees=100, interaction.depth=10,
              shrinkage=0.1, estimand = "ATE", stop.method=c("es.mean","ks.max"),
              n.minobsinnode = 1, bag.fraction=0.5, n.keep = 1, n.grid = 25, ks.exact = NULL,
              verbose=FALSE, version="xgboost")
  
  data$weights_xgb<-get.weights(ps.xgb, stop.method="es.mean")
  design.ps<-svydesign(ids=~1,weights=~weights_xgb,data=data) #ids is for clustering variable
  
  output1<-svyglm(CHANGE ~ treat + W_KG + H_CM + AGE + SEX + ETHFAC_a + ETHFAC_b + ETHFAC_c + ETHFAC_d + ETHFAC_e + SITE_a + 
                    SITE_b + SITE_c + SITE_e + INITUB_BreakfastEater + INITUB_aSkipper, design=design.ps)
  
  estimate[1,4]=summary(output1)$coefficients[2, 1]
  estimate[2,4]=summary(output1)$coefficients[2, 2]
  
  error[1,4] = mae(TAU, summary(output1)$coefficients[2, 1])
  error[2,4] = rmse(TAU, summary(output1)$coefficients[2, 1])/mean(TAU)
  
  list_estimate_psm[[i]] <- estimate
  list_error_psm[[i]] <- error
  
}

df_psmtable <- as.data.frame(apply(simplify2array(list_estimate_psm), 1:2, mean))
df_psmtable




