load("/usr/bin/app/test/cpat/Fly_logitModel.RData")
test <- read.table(file="/usr/bin/app/test/cpat/cpat.ORF_info.tsv",sep="\t",header=T)
test$Coding_prob <- predict(mylogit,newdata=test,type="response")
write.table(test, file="/usr/bin/app/test/cpat/cpat.ORF_prob.tsv", quote=F, sep="\t",row.names=FALSE, col.names=TRUE)
