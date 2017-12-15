# First part: introduction
# 1.1 Preliminaries
rm(list=ls())
graphics.off()
set.seed(1234)
library(MASS)
data(Boston)
n <- nrow(Boston) ; 
train <- sample(1:n, 3*round(n / 4))
Boston.train <- Boston[ train, ]
Boston.test <- Boston[-train, ]
#Descriptive analysis
par(mfrow=c(1,2))
hist(Boston.train$medv, freq=FALSE, col="cyan", main="histogram",xlab="")
lines(density(Boston.train$medv), col="darkred")
plot(ecdf(Boston.train$medv), main="distribution function",xlab="")
title(outer=TRUE, main="\n Median rent distribution")

#Some of variables appear to be redundant:
heatmap(abs(cor(Boston.train[, -14])), symm=TRUE)

# Using dispersion diagrams we can present the similarity of the response, the body mass index and the first
# serological variable.
library(GGally)
ggscatmat(Boston.train)

# Second part: multilinear regression and variable selection
# Multilinear regression
null <- lm(medv~1, Boston.train)
full <- lm(medv~., Boston.train)
#As expected, the we reject the nul hypothesis.
anova(null,full)
#The R2 parameter is rather good:
summary(full)
#Now for the diagnostic plots:
par(mfrow=c(2,2))
plot(full, which=c(1,2,4,5))
# The diagnostics appear to be good, with a weaker variance to the small and large adjusted values, but that is
# mostly due to a smaller presence of observations in these ranges. A root-square or logarithmic transformation
# slightly enhance the symmetry of the residuals. There is no presence of strongly aberrant values (the distance
# of Cook remains small for all members), still, 3 points might be removed from the study.
null <- lm(log(medv)~1, Boston.train)
full <- lm(log(medv)~., Boston.train)
summary(full)
par(mfrow=c(2,2))
plot(full, which=c(1,2,4,5))

#Exhaustive search
#There are a total of 2^10 = 1024 possible models. We focus on the best 500 of them.
library(leaps)
bss <- summary(regsubsets(log(medv) ~ . , data=Boston.train, nvmax=10, nbest=500, really.big=TRUE))

# Using this object (by examining it structure - you can use the str command), we can recover all the values we
# need. We recover the best results for each parameter in order to represent them graphicly in separate plots.
bss.size <- as.numeric(rownames(bss$which))
bss.best.rss <- tapply(bss$rss, bss.size, min)
bss.best.adjr2 <- tapply(bss$adjr2, bss.size, max)
bss.best.bic <- tapply(bss$bic , bss.size, min)
bss.best.Cp <- tapply(bss$cp , bss.size, min)
# We complete the statistics with the performance of the null model which is not ouputed by the regsubset
# function.
n <- nrow(Boston.train)
RSS0 <- sum(resid(null)^2)
bss.best.rss <- c(RSS0, bss.best.rss)
bss.best.adjr2 <- c(summary(null )$adj.r.squared, bss.best.adjr2)
bss.best.bic <- c(log(RSS0/n), bss.best.bic)
bss.best.Cp <- c(n+RSS0/(n*summary(full)$sigma^2), bss.best.Cp)
#Plot time!
model.size <- 0:10
par(mfrow=c(2,2))
plot(model.size, log(bss.best.rss), type="b",
     xlab="subset size", ylab="RSS", col="red2" )
points(bss.size, log(bss$rss), pch=20, col="gray", cex=0.7)
plot(model.size, bss.best.adjr2, type="b",
     xlab="subset size", ylab="Adjusted R2", col="red2" )
points(bss.size, bss$adjr2, pch=20, col="gray", cex=0.7)
plot(model.size, bss.best.Cp, type="b",
     xlab="subset size", ylab="Mallows' Cp", col="red2" )
points(bss.size, bss$cp, pch=20, col="gray", cex=0.7)
plot(model.size, bss.best.bic, type="b",
     xlab="subset size", ylab="BIC", col="red2" )
points(bss.size, bss$bic, pch=20, col="gray", cex=0.7)

#Stepwise Selection
scope <- list(lower = terms(log(medv) ~ 1, data=Boston.train),
              upper = terms(log(medv) ~ ., data=Boston.train))
step.AIC <- step(null, scope, direction='both', trace=FALSE)
step.BIC <- step(null, scope, direction='both', k=log(n), trace=FALSE)
# Both the AIC and BIC models include common variables. The variables selected, however, differ between the
# two models, due to the existing correlations between these. While we have a slightly worse R2 with the BIC,
# its interpretation is easier.
step.AIC
step.BIC
par(mfrow=c(2,2))
plot(step.BIC, which=c(1,2,4,5))
summary(step.BIC)

##Third part: penalisation methods
#Ridge Regression
library(glmnet)
x <- as.matrix(Boston.train[, -14])
y <- log(Boston.train$medv)
ridge <- glmnet(x,y,alpha=0)
#We can represent the regularisation path in function of different mesurements:
par(mfrow=c(1,3))
plot(ridge, xvar="lambda")
plot(ridge, xvar="norm")
plot(ridge, xvar="dev")

#The results are interchangable for both methods of cross-validation.
ridge.10cv <- cv.glmnet(x,y,nfolds=10, alpha=0, grouped=FALSE)
ridge.loo <- cv.glmnet(x,y,nfolds=n , alpha=0, grouped=FALSE)
par(mfrow=c(1,2))
plot(ridge.10cv)
plot(ridge.loo)
# We can access the generated models through the predict function. Take care as the new value of the
# predictors used for the prediction must be formated as a matrix.
x0 <- as.matrix(Boston.train[1:5, -14]) # A new observation as an example
predict(ridge, newx=x0, s=ridge.10cv$lambda.min)
predict(ridge, newx=x0, s=ridge.10cv$lambda.1se)
#Lasso Regression
# We just need to use the previous code and modulate the alpha parameter. Note the consecutivelly selected
# variables on the path towards the solution.
lasso <- glmnet(x,y)
par(mfrow=c(1,3))
plot(lasso, xvar="lambda")
plot(lasso, xvar="norm")
plot(lasso, xvar="dev")
#Again the results remain close with both methods of cross-validation.
lasso.10cv <- cv.glmnet(x,y,nfolds=10, grouped=FALSE)
lasso.loo <- cv.glmnet(x,y,nfolds=n , grouped=FALSE)
par(mfrow=c(1,2))
plot(lasso.10cv)
plot(lasso.loo)
# We will therefore study the models that minimise the cross validation error and the most parcimonious model
# within 1 std of the best model.
predict(lasso, x0, s=lasso.10cv$lambda.min)
predict(lasso, x0, s=lasso.loo$lambda.1se)
#Here are the standard parameters for a lasso regression.
n <- nrow(Boston.train)
p <- ncol(Boston.train) - 1 + 1
AIC <- n*log(colMeans((y - predict(lasso, x))^2)) + 2 * lasso$df
BIC <- n*log(colMeans((y - predict(lasso, x))^2)) + log(n) * lasso$df
eBIC <- n*log(colMeans((y - predict(lasso, x))^2)) + log(p) * lasso$df
mBIC <- n*log(colMeans((y - predict(lasso, x))^2)) + (log(n) + 2 *log(p)) * lasso$df
library(ggplot2)
d <- data.frame(lambda = rep(lasso$lambda, 4),
                value = c(AIC, BIC, eBIC, mBIC),
                critere = factor(rep(c("AIC","BIC","eBIC","mBIC"), each=length(lasso$lambda))))
ggplot(d, aes(x=lambda,y=value,colour=critere,group=critere)) + geom_line() + scale_x_log10()
#We name lasso.min, lasso.1se, lasso.BIC and lasso.mBIC the corresponding models.
lambda.min.BIC <- lasso$lambda[which.min(BIC)]
lambda.min.mBIC <- lasso$lambda[which.min(mBIC)]
predict(lasso, x0, s=lambda.min.BIC)
predict(lasso, x0, s=lambda.min.mBIC)

##Forth part: Evaluating the quality of the models obtained
y.test <- log(Boston.test$medv)
x.test <- as.matrix(Boston.test[, -14])
err.null <- mean((y.test - predict(null, Boston.test))^2)
err.full <- mean((y.test - predict(full, Boston.test))^2)
err.sAIC <- mean((y.test - predict(step.AIC, Boston.test))^2)
err.sBIC <- mean((y.test - predict(step.BIC, Boston.test))^2)
err.ridge.min <- mean((y.test - predict(ridge, newx=x.test, s=ridge.10cv$lambda.min))^2)
err.ridge.1se <- mean((y.test - predict(ridge, newx=x.test, s=ridge.10cv$lambda.min))^2)
err.lasso.min <- mean((y.test - predict(lasso, newx=x.test, s=lasso.10cv$lambda.min))^2)
err.lasso.1se <- mean((y.test - predict(lasso, newx=x.test, s=lasso.10cv$lambda.min))^2)
err.lasso.BIC <- mean((y.test - predict(lasso, newx=x.test, s=lambda.min.BIC))^2)
err.lasso.mBIC <- mean((y.test - predict(lasso, newx=x.test, s=lambda.min.mBIC))^2)
res <- data.frame(modele = c("null", "full", "step.AIC", "step.BIC", "ridge.CVmin",
                             "ridge.CV1se", "lasso.CVmin", "lasso.CV1se", "lasso.BIC", "lasso.mBIC"),
                  erreur = c(err.null, err.full, err.sAIC, err.sBIC, err.ridge.min, err.ridge.1se,
                             err.lasso.min, err.lasso.1se, err.lasso.BIC, err.lasso.mBIC))
print(res)

err.lasso <- colMeans((y.test - predict(lasso, newx=x.test))^2)
err.ridge <- colMeans((y.test - predict(ridge, newx=x.test))^2)
par(mfrow=c(1,1))
plot(log10(lasso$lambda), err.lasso, type="l")
axis(3, at=c(log10(lasso.10cv$lambda.min), log10(lasso.10cv$lambda.1se),
             log10(lambda.min.BIC), log10(lambda.min.mBIC)),
     labels = c("CV min", "CV 1se", "BIC", "mBIC"))
abline(v=c(log10(lasso.10cv$lambda.min), log10(lasso.10cv$lambda.1se),
           log10(lambda.min.BIC), log10(lambda.min.mBIC)), lty=3)
plot(log10(ridge$lambda), err.ridge, type="l")
axis(3, at=c(log10(ridge.10cv$lambda.min), log10(ridge.10cv$lambda.1se)),
     labels = c("CV min", "CV 1se"))
abline(v=c(log10(ridge.10cv$lambda.min), log10(ridge.10cv$lambda.1se)), lty=3)
