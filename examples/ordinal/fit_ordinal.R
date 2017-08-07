# add code to fit model in R
library(ordinal)
data(wine)

m <- clmm(rating ~ temp + contact + (1 | judge ),
          data=wine)

write.table(coef(m)[1:4], "intercept_R.txt", row.names=FALSE, col.names=FALSE)
write.table(coef(m)[5:6], "fixef_R.txt", row.names=FALSE, col.names=FALSE)
write.table(ranef(m)[[1]], "ranef.R.txt", row.names=FALSE, col.names=FALSE)
write.table(data.frame(grad=m$gradient), "gradient_R.csv", row.names=FALSE, col.names=FALSE)
