

rm(list=ls())
library(beepr)

pacman::p_load(postDoubleR, data.table, glmnet, tidyverse, gbm, randomForest, party, mlr3, mlr3learners, DoubleML,
               paradox, mlr3tuning, e1071, nnet, Metrics, grf)

pacman::p_load(haven, dplyr, sandwich, lmtest, MatchIt, Metrics)
pacman::p_load(survey, grf, twang)

########################## PSM-ML W/ SIMULATED DATA #############################

rm(list=ls())

#simulate data

set.seed(12345)

#change
n_obs <- 150 #baseline: 150; large: 15000
n_var <- 10 #baseline: 10; large: 100
TAU <- rnorm(n=n_obs, mean=1, sd=1) #baseline: 1; heterogenous: rnorm(n=n_obs, mean=1, sd=1)

set.seed(12345)
n <- n_obs
p <- n_var
X <- matrix(rnorm(n * p), n, p)

#XB <- sin(X %*% matrix(rnorm(p))) #nonlinear
XB <- X %*% matrix(rnorm(p)) #linear

W <- rbinom(n, 1, 1 / (1 + exp(-runif(n)))) #Exogenous treatment

Y <- 1+XB+(TAU*W)+rnorm(n, 0, 1)

dt<-data.frame(Y,X,W)
mean(TAU)
sd(TAU)
mean(dt$Y)

seed <- c(1,5,10,25,50)

xnam <- paste("X", 1:n_var, sep="")
fm_w <- as.formula(paste("W ~ ", paste(xnam, collapse= "+")))
fm_y <- as.formula(paste("Y ~ W + ", paste(xnam, collapse= "+")))

reg1 <- lm(fm_y, data=dt)
summary(reg1)

x = X
w = W
y = Y

ps_cate_xl <- function(x, w, y, weight){
  
  x_1 = x[which(w==1),]
  x_0 = x[which(w==0),]
  
  y_1 = y[which(w==1)]
  y_0 = y[which(w==0)]
  
  nobs_1 = nrow(x_1)
  nobs_0 = nrow(x_0)
  
  nobs = nrow(x)
  pobs = ncol(x)
  
  nobs = nrow(X)
  pobs = ncol(X)
  
  k_folds_mu1 = floor(max(3, min(10,nobs_1/4)))
  k_folds_mu0 = floor(max(3, min(10,nobs_0/4)))
  k_folds_p = floor(max(3, min(10,nobs/4)))
  
  foldid_1 = sample(rep(seq(k_folds_mu1), length = nobs_1))
  foldid_0 = sample(rep(seq(k_folds_mu0), length = nobs_0))
  foldid_w = sample(rep(seq(k_folds_p), length = nobs))
  
  t_1_fit = glmnet::cv.glmnet(x_1, y_1, foldid = foldid_1, lambda = c(0.5,1), alpha = 1) 
  mu1_hat = predict(t_1_fit, newx = x, s = "lambda.1se") #lambda_choice = c("lambda.min", "lambda.1se")
  
  t_0_fit = glmnet::cv.glmnet(x_0, y_0, foldid = foldid_0, lambda = c(0.5,1), alpha = 1)
  mu0_hat = predict(t_0_fit, newx = x, s = "lambda.1se") #lambda_choice = c("lambda.min", "lambda.1se")
  
  d_1 = y_1 - mu0_hat[w==1]
  d_0 = mu1_hat[w==0] - y_0
  
  x_1_fit = glmnet::cv.glmnet(x_1, d_1, foldid = foldid_1, lambda = c(0.5,1), alpha = 1)
  x_0_fit = glmnet::cv.glmnet(x_0, d_0, foldid = foldid_0, lambda = c(0.5,1), alpha = 1)
  
  tau_1_pred = predict(x_1_fit, newx = x, s = "lambda.1se")
  tau_0_pred = predict(x_0_fit, newx = x, s ="lambda.1se")
  
  tau_1_pred * (1 - weight) + tau_0_pred * weight
  
}


dml_cate_rl <- function(data, env){
  
  g_hat <- as.matrix(data$predictions$ml_l) # predictions of g_o or y
  m_hat <- as.matrix(data$predictions$ml_m) # predictions of m_o or w or treatment
  
  env$cate_type3 <- ps_cate_xl(x=X, w=W, y=Y, weight=m_hat)
  
}



list_estimate<-list()
list_error<-list()

no_methods=13
estimate=matrix(NA,4,no_methods)
error=matrix(NA,6,no_methods)

for(i in 1:length(seed)){
  
  #probit
  
  set.seed(seed[i])
  
  m.out1 <- matchit(fm_w, data = dt,
                    method = "full", distance = "glm", link = "probit")
  
  m.data1 <- match.data(m.out1)
  
  fit1 <- lm(fm_y, data = m.data1, weights = weights)
  
  c <- coeftest(fit1, vcov. = vcovCL, cluster = ~subclass)
  
  tau_cate_ps <- ps_cate_xl(x=X, w=W, y=Y, weight=m.data1$weights)
  
  estimate[1,1]=c[[2]] #ATE
  estimate[2,1]=c[[2,2]] #sd
  
  estimate[3,1]=mean(tau_cate_ps) #HTE
  estimate[4,1]=sd(tau_cate_ps) #sd
  
  error[1,1] = mae(TAU, c[[2]])
  error[2,1] = rmse(TAU, c[[2]])/mean(TAU)
  error[3,1] = rmse(TAU, c[[2]])/sd(TAU)
  
  error[4,1] = mae(TAU, tau_cate_ps)
  error[5,1] = rmse(TAU, tau_cate_ps)/mean(TAU)
  error[6,1] = rmse(TAU, tau_cate_ps)/sd(TAU)
  
  #nnet
  
  set.seed(seed[i])
  
  m.out1 <- matchit(fm_w, data = dt,
                    method = "full", distance = "nnet", distance.options=list(size=5))
  
  m.data1 <- match.data(m.out1)
  
  fit1 <- lm(fm_y, data = m.data1, weights = weights)
  
  c <- coeftest(fit1, vcov. = vcovCL, cluster = ~subclass)
  
  tau_cate_ps <- ps_cate_xl(x=X, w=W, y=Y, weight=m.data1$weights)
  
  estimate[1,2]=c[[2]]
  estimate[2,2]=c[[2,2]]
  
  estimate[3,2]=mean(tau_cate_ps)
  estimate[4,2]=sd(tau_cate_ps)
  
  error[1,2] = mae(TAU, c[[2]])
  error[2,2] = rmse(TAU, c[[2]])/mean(TAU)
  error[3,2] = rmse(TAU, c[[2]])/sd(TAU)
  
  error[4,2] = mae(TAU, tau_cate_ps)
  error[5,2] = rmse(TAU, tau_cate_ps)/mean(TAU)
  error[6,2] = rmse(TAU, tau_cate_ps)/sd(TAU)
  
  #rf
  
  n_vars <- ncol(X)
  n_obs <- nrow(X)
  
  set.seed(seed[i])
  
  m.out1 <- matchit(fm_w, data = dt,
                    method = "full", distance = "randomforest", distance.options=list(num.trees = 100, mtry = floor(sqrt(n_vars)), 
                                                                                      min.node.size = 2,
                                                                                      max.depth = 10))
  
  m.data1 <- match.data(m.out1)
  
  fit1 <- lm(fm_y, data = m.data1, weights = weights)
  
  c <- coeftest(fit1, vcov. = vcovCL, cluster = ~subclass)
  
  tau_cate_ps <- ps_cate_xl(x=X, w=W, y=Y, weight=m.data1$weights)
  
  estimate[1,3]=c[[2]]
  estimate[2,3]=c[[2,2]]
  
  estimate[3,3]=mean(tau_cate_ps)
  estimate[4,3]=sd(tau_cate_ps)
  
  error[1,3] = mae(TAU, c[[2]])
  error[2,3] = rmse(TAU, c[[2]])/mean(TAU)
  error[3,3] = rmse(TAU, c[[2]])/sd(TAU)
  
  error[4,3] = mae(TAU, tau_cate_ps)
  error[5,3] = rmse(TAU, tau_cate_ps)/mean(TAU)
  error[6,3] = rmse(TAU, tau_cate_ps)/sd(TAU)
  
  #xgb
  
  set.seed(seed[i])
  
  ps.xgb = ps(fm_w, 
              data = dt, n.trees=100, interaction.depth=10,
              shrinkage=0.1, estimand = "ATE", stop.method=c("es.mean","ks.max"),
              n.minobsinnode = 1, bag.fraction=0.5, n.keep = 1, n.grid = 25, ks.exact = NULL,
              verbose=FALSE, version="xgboost")
  
  dt$weights_xgb<-get.weights(ps.xgb, stop.method="es.mean")
  design.ps<-svydesign(ids=~1,weights=~weights_xgb,data=dt) #ids is for clustering variable
  
  output1<-svyglm(fm_y, design=design.ps)
  
  tau_cate_ps <- ps_cate_xl(x=X, w=W, y=Y, weight=dt$weights_xgb)
  
  estimate[1,4]=summary(output1)$coefficients[2, 1]
  estimate[2,4]=summary(output1)$coefficients[2, 2]
  
  estimate[3,4]=mean(tau_cate_ps)
  estimate[4,4]=sd(tau_cate_ps)
  
  error[1,4] = mae(TAU, summary(output1)$coefficients[2, 1])
  error[2,4] = rmse(TAU, summary(output1)$coefficients[2, 1])/mean(TAU)
  error[3,4] = rmse(TAU, summary(output1)$coefficients[2, 1])/sd(TAU)
  
  error[4,4] = mae(TAU, tau_cate_ps)
  error[5,4] = rmse(TAU, tau_cate_ps)/mean(TAU)
  error[6,4] = rmse(TAU, tau_cate_ps)/sd(TAU)
  
  #double machine learning starts
  
  n_vars <- ncol(X)
  data_dt <- as.data.table(dt)
  n_obs <- nrow(data_dt)
  
  #double ml - lasso/lasso #Uses postDoubleR package
  
  set.seed(seed[i])
  
  doubleml_data = DoubleMLData$new(data_dt,
                                   y_col = "Y",
                                   d_cols = "W",
                                   x_cols = xnam)
  
  lgr::get_logger("mlr3")$set_threshold("warn")
  lgr::get_logger("bbotk")$set_threshold("warn")
  
  learner_classif_m = lrn("classif.glmnet")
  learner_regr_g = lrn("regr.glmnet")
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_g, ml_m=learner_classif_m)
  
  doubleml_plr$fit()
  doubleml_plr$fit(store_predictions=TRUE)
  
  y <- as.matrix(dt[,1])
  d <- as.matrix(dt[,12])
  
  
  myEnv <- new.env()
  tau_cate_dml <- dml_cate_rl(doubleml_plr, myEnv)
  
  estimate[1,5]=as.numeric(doubleml_plr$coef)
  estimate[2,5]=as.numeric(doubleml_plr$se)
  
  estimate[3,5]=mean(myEnv$cate_type3)
  estimate[4,5]=sd(myEnv$cate_type3)
  
  error[1,5] = mae(TAU, as.numeric(doubleml_plr$coef))
  error[2,5] = rmse(TAU, as.numeric(doubleml_plr$coef))/mean(TAU)
  error[3,5] = rmse(TAU, as.numeric(doubleml_plr$coef))/sd(TAU)
  
  error[4,5] = mae(TAU, myEnv$cate_type3)
  error[5,5] = rmse(TAU, myEnv$cate_type3)/mean(TAU)
  error[6,5] = rmse(TAU, myEnv$cate_type3)/sd(TAU)
  
  #double ml - random forest/logit
  
  set.seed(seed[i])
  
  learner = lrn("classif.log_reg")
  learner_regr_g = lrn("regr.ranger", num.trees = 100, mtry = floor(sqrt(n_vars)), min.node.size = 2,
                       max.depth = 6)
  learner_classif_m = learner$clone()
  
  obj_dml_plr_bonus = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_g, ml_m=learner_classif_m)
  obj_dml_plr_bonus$fit()
  obj_dml_plr_bonus$fit(store_predictions=TRUE)
  
  y <- as.matrix(dt[,1])
  d <- as.matrix(dt[,12])
  
  myEnv <- new.env()
  tau_cate_dml <- dml_cate_rl(obj_dml_plr_bonus, myEnv)
  
  estimate[1,6]=as.numeric(obj_dml_plr_bonus$coef)
  estimate[2,6]=as.numeric(obj_dml_plr_bonus$se)
  
  estimate[3,6]=mean(myEnv$cate_type3)
  estimate[4,6]=sd(myEnv$cate_type3)
  
  error[1,6] = mae(TAU, as.numeric(obj_dml_plr_bonus$coef))
  error[2,6] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/mean(TAU)
  error[3,6] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/sd(TAU)
  
  error[4,6] = mae(TAU, myEnv$cate_type3)
  error[5,6] = rmse(TAU, myEnv$cate_type3)/mean(TAU)
  error[6,6] = rmse(TAU, myEnv$cate_type3)/sd(TAU)
  
  #double ml - random forest/neural network
  
  set.seed(seed[i])
  
  learner_classif_m = lrn("classif.nnet", maxit=100, size=c(5))
  learner_regr_g = lrn("regr.ranger", num.trees = 100, mtry = floor(sqrt(n_vars)), min.node.size = 2, max.depth = 10)
  
  doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_g, ml_m=learner_classif_m)
  
  doubleml_plr$fit()
  doubleml_plr$fit(store_predictions=TRUE)
  
  y <- as.matrix(dt[,1])
  d <- as.matrix(dt[,12])
  
  
  
  
  g_hat <- as.matrix(doubleml_plr$predictions$ml_l) # predictions of g_o or y
  m_hat <- as.matrix(doubleml_plr$predictions$ml_m) # predictions of m_o or w or treatment
  
  cate_type3 <- ps_cate_xl(x=X, w=W, y=Y, weight=m_hat)
  
  
  estimate[1,7]=as.numeric(doubleml_plr$coef)
  estimate[2,7]=as.numeric(doubleml_plr$se)
  
  estimate[3,7]=mean(cate_type3)
  estimate[4,7]=sd(cate_type3)
  
  error[1,7] = mae(TAU, as.numeric(doubleml_plr$coef))
  error[2,7] = rmse(TAU, as.numeric(doubleml_plr$coef))/mean(TAU)
  error[3,7] = rmse(TAU, as.numeric(doubleml_plr$coef))/sd(TAU)
  
  error[4,7] = mae(TAU, cate_type3)
  error[5,7] = rmse(TAU, cate_type3)/mean(TAU)
  error[6,7] = rmse(TAU, cate_type3)/sd(TAU)
  
  #double ml - random forest/lasso
  
  set.seed(seed[i])
  
  learner = lrn("regr.ranger", num.trees=100, mtry=floor(sqrt(n_vars)), max.depth=5, min.node.size=2)
  learner_regr_g = learner$clone()
  
  learner = lrn("classif.glmnet")
  learner_classif_m = learner$clone()
  
  obj_dml_plr_bonus = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_g, ml_m=learner_classif_m)
  obj_dml_plr_bonus$fit()
  obj_dml_plr_bonus$fit(store_predictions=TRUE)
  
  y <- as.matrix(dt[,1])
  d <- as.matrix(dt[,12])
  
  myEnv <- new.env()
  tau_cate_dml <- dml_cate_rl(obj_dml_plr_bonus, myEnv)
  
  estimate[1,8]=as.numeric(obj_dml_plr_bonus$coef)
  estimate[2,8]=as.numeric(obj_dml_plr_bonus$se)
  
  estimate[3,8]=mean(myEnv$cate_type3)
  estimate[4,8]=sd(myEnv$cate_type3)
  
  error[1,8] = mae(TAU, as.numeric(obj_dml_plr_bonus$coef))
  error[2,8] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/mean(TAU)
  error[3,8] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/sd(TAU)
  
  error[4,8] = mae(TAU, myEnv$cate_type3)
  error[5,8] = rmse(TAU, myEnv$cate_type3)/mean(TAU)
  error[6,8] = rmse(TAU, myEnv$cate_type3)/sd(TAU)

  
  #double ml - random forest/random forest
  
  set.seed(seed[i])
  
  n_rep = 10
  n_folds = 10
  
  learner_g = lrn("regr.ranger", num.trees=100, mtry=floor(sqrt(n_vars)), max.depth=5, min.node.size=2)
  learner_regr_g = learner_g$clone()
  
  learner_m = lrn("classif.ranger", num.trees=100, mtry=floor(sqrt(n_vars)), max.depth=5, min.node.size=2)
  learner_classif_m = learner_m$clone()
  
  obj_dml_plr_bonus = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_g, ml_m=learner_classif_m, n_rep=n_rep, n_folds=n_folds)
  obj_dml_plr_bonus$fit()
  obj_dml_plr_bonus$fit(store_predictions=TRUE)
  obj_dml_plr_bonus$params_names()
  
  
  
  g_hat <- as.matrix(rowMeans(obj_dml_plr_bonus$predictions$ml_l)) # predictions of g_o or y
  m_hat <- as.matrix(rowMeans(obj_dml_plr_bonus$predictions$ml_m)) # predictions of m_o or w or treatment
  
  cate_type3 <- ps_cate_xl(x=X, w=W, y=Y, weight=m_hat)
  
  
  
  estimate[1,9]=as.numeric(obj_dml_plr_bonus$coef)
  estimate[2,9]=as.numeric(obj_dml_plr_bonus$se)
  
  estimate[3,9]=mean(cate_type3)
  estimate[4,9]=sd(cate_type3)
  
  error[1,9] = mae(TAU, as.numeric(obj_dml_plr_bonus$coef))
  error[2,9] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/mean(TAU)
  error[3,9] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/sd(TAU)
  
  error[4,9] = mae(TAU, cate_type3)
  error[5,9] = rmse(TAU, cate_type3)/mean(TAU)
  error[6,9] = rmse(TAU, cate_type3)/sd(TAU)
  
  #double ml - lasso/neural network
  
  set.seed(seed[i])
  
  learner = lrn("classif.nnet", maxit=100, size=c(5))
  learner_classif_m = learner$clone()
  
  learner = lrn("regr.glmnet")
  learner_regr_g = learner$clone()
  
  obj_dml_plr_bonus = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_g, ml_m=learner_classif_m)
  obj_dml_plr_bonus$fit()
  obj_dml_plr_bonus$fit(store_predictions=TRUE)
  
  y <- as.matrix(dt[,1])
  d <- as.matrix(dt[,12])
  
  
  
  g_hat <- as.matrix(obj_dml_plr_bonus$predictions$ml_l) # predictions of g_o or y
  m_hat <- as.matrix(obj_dml_plr_bonus$predictions$ml_m) # predictions of m_o or w or treatment
  
  cate_type3 <- ps_cate_xl(x=X, w=W, y=Y, weight=m_hat)
  
  estimate[1,10]=as.numeric(obj_dml_plr_bonus$coef)
  estimate[2,10]=as.numeric(obj_dml_plr_bonus$se)
  
  estimate[3,10]=mean(cate_type3)
  estimate[4,10]=sd(cate_type3)
  
  error[1,10] = mae(TAU, as.numeric(obj_dml_plr_bonus$coef))
  error[2,10] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/mean(TAU)
  error[3,10] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/sd(TAU)
  
  error[4,10] = mae(TAU, cate_type3)
  error[5,10] = rmse(TAU, cate_type3)/mean(TAU)
  error[6,10] = rmse(TAU, cate_type3)/sd(TAU)
  
  
  #double ml - xgboost/xgboost
  
  set.seed(seed[i])
  
  learner = lrn("classif.xgboost", nrounds = c(100), max_depth = c(10), min_child_weight =1,
                subsample = 1, gamma = 0.05,  colsample_bytree = 0.8, eta = c(0.1))
  learner_classif_m = learner$clone()
  
  learner = lrn("regr.xgboost", nrounds = c(100), max_depth = c(10), min_child_weight =1,
                subsample = 1, gamma = 0.05,  colsample_bytree = 0.8, eta = c(0.1))
  learner_regr_g = learner$clone()
  
  obj_dml_plr_bonus = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_g, ml_m=learner_classif_m)
  obj_dml_plr_bonus$fit()
  obj_dml_plr_bonus$fit(store_predictions=TRUE)
  
  y <- as.matrix(dt[,1])
  d <- as.matrix(dt[,12])
  
  myEnv <- new.env()
  tau_cate_dml <- dml_cate_rl(obj_dml_plr_bonus, myEnv)
  
  estimate[1,11]=as.numeric(obj_dml_plr_bonus$coef)
  estimate[2,11]=as.numeric(obj_dml_plr_bonus$se)
  
  estimate[3,11]=mean(myEnv$cate_type3)
  estimate[4,11]=sd(myEnv$cate_type3)
  
  error[1,11] = mae(TAU, as.numeric(obj_dml_plr_bonus$coef))
  error[2,11] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/mean(TAU)
  error[3,11] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/sd(TAU)
  
  error[4,11] = mae(TAU, myEnv$cate_type3)
  error[5,11] = rmse(TAU, myEnv$cate_type3)/mean(TAU)
  error[6,11] = rmse(TAU, myEnv$cate_type3)/sd(TAU)
  
  #double ml - ols/logit
  
  set.seed(seed[i])
  
  learner = lrn("regr.lm") 
  learner_regr_g = learner$clone()
  
  learner = lrn("classif.log_reg")
  learner_classif_m = learner$clone()
  
  obj_dml_plr_bonus = DoubleMLPLR$new(doubleml_data, ml_g=learner_regr_g, ml_m=learner_classif_m)
  obj_dml_plr_bonus$fit()
  obj_dml_plr_bonus$fit(store_predictions=TRUE)
  
  y <- as.matrix(dt[,1])
  d <- as.matrix(dt[,12])
  
  myEnv <- new.env()
  tau_cate_dml <- dml_cate_rl(obj_dml_plr_bonus, myEnv)
  
  estimate[1,12]=as.numeric(obj_dml_plr_bonus$coef)
  estimate[2,12]=as.numeric(obj_dml_plr_bonus$se)
  
  estimate[3,12]=mean(myEnv$cate_type3)
  estimate[4,12]=sd(myEnv$cate_type3)
  
  error[1,12] = mae(TAU, as.numeric(obj_dml_plr_bonus$coef))
  error[2,12] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/mean(TAU)
  error[3,12] = rmse(TAU, as.numeric(obj_dml_plr_bonus$coef))/sd(TAU)
  
  error[4,12] = mae(TAU, myEnv$cate_type3)
  error[5,12] = rmse(TAU, myEnv$cate_type3)/mean(TAU)
  error[6,12] = rmse(TAU, myEnv$cate_type3)/sd(TAU)
  
  #causal forest
  
  set.seed(seed[i])
  
  c.forest <- causal_forest(X, as.vector(Y), as.vector(W))
  
  ate <- average_treatment_effect(c.forest, target.sample = "all")
  tau_cate_cf = predict (c.forest)$predictions
  
  estimate[1,13]=ate[[1]]
  estimate[2,13]=ate[[2]]
  
  estimate[3,13]=mean(tau_cate_cf)
  estimate[4,13]=sd(tau_cate_cf)
  
  error[1,13] = mae(TAU, ate[[1]])
  error[2,13] = rmse(TAU, ate[[1]])/mean(TAU)
  error[3,13] = rmse(TAU, ate[[1]])/sd(TAU)
  
  error[4,13] = mae(TAU, tau_cate_cf)
  error[5,13] = rmse(TAU, tau_cate_cf)/mean(TAU)
  error[6,13] = rmse(TAU, tau_cate_cf)/sd(TAU)
  
  
  
  list_estimate[[i]] <- estimate
  list_error[[i]] <- error
  
}

beep()


df_dmltable <- as.data.frame(apply(simplify2array(list_estimate), 1:2, mean))
df_dmltable


df_dml <- do.call(rbind.data.frame, list_estimate)
df_dml 


df_dmltable <- as.data.frame(apply(simplify2array(list_error), 1:2, mean))
df_dmltable


df_dml <- do.call(rbind.data.frame, list_error)
df_dml 























