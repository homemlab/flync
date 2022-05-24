# libraries
library(ballgown)
library(genefilter)

# get arguments
args = commandArgs(trailingOnly=TRUE)

workdir <- args[1]
mdfile <- args[2]

setwd(workdir)

# get metadata from file (sample,condition)
pheno_data <- read.csv(mdfile, header = FALSE)

colnames(pheno_data) = c("sample", "condition")

# get paths to the ballgown files per sample
sample_vector <- list.dirs(path = "cov")
sample_vector <- sample_vector[-1]

# initiate the ballgown object
bg_obj <- ballgown(pData = pheno_data, samples = sample_vector)

# filter out transcripts with low variance
bg_obj_filt <- subset(bg_obj, "rowVars(texpr(bg_obj)) >1", genomesubset=TRUE)

# get diff expr for transcript-level features
results_transcripts <- stattest(bg_obj_filt,
                                feature="transcript",
                                covariate="condition",
                                getFC=TRUE, 
                                meas="FPKM")

# add the transcript identification
results_transcripts <- data.frame(geneNames = geneNames(bg_obj_filt),
                                  geneIDs = geneIDs(bg_obj_filt),
                                  results_transcripts)

# write final table
write.csv(results_transcripts, "results/dge.csv", row.names = FALSE)

# get transcript-level FPKM values and coordenates
texpr <- texpr(bg_obj_filt, meas = "all")

# add mean and sd columns
texpr$mean.FPKM <- rowMeans(texpr[ , grepl("FPKM", names(texpr))])
texpr$sd.FPKM <- apply(texpr[ , grepl("FPKM", names(texpr))], 1, sd)
texpr$mean.cov <- rowMeans(texpr[ , grepl("cov", names(texpr))])
texpr$sd.cov <- apply(texpr[ , grepl("cov", names(texpr))], 1, sd)


write.csv(texpr, "results/trans_expr.csv", row.names = FALSE)
