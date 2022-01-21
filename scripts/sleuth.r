#!/usr/bin/Rscript

args = commandArgs(trailingOnly=TRUE)

workdir <- args[1]
mdfile <- args[2]

md <- read.csv(mdfile, header = TRUE, stringsAsFactors = FALSE)

### Following the sleuth Getting Started tutorial ###
library("sleuth")

so <- sleuth_prep(md,
             full_model = ~condition,
             read_bootstrap_tpm=TRUE,
             extra_bootstrap_summary = TRUE,
             transformation_function = function(x) log2(x + 0.5))

so <- sleuth_fit(so, ~condition, 'full')
so <- sleuth_fit(so, ~1, 'reduced')
so <- sleuth_lrt(so, 'reduced', 'full')
wald_test <- colnames(design_matrix(so))[2]
so <- sleuth_wt(so, wald_test)

sleuth_table <- sleuth_results(so, 'reduced:full', 'lrt', show_all = FALSE)
sleuth_significant <- dplyr::filter(sleuth_table, qval <= 0.05)

write.csv(sleuth_table,file.path(workdir, "results_dea/dea_all.csv"), row.names = FALSE)
write.csv(sleuth_significant,file.path(workdir, "results_dea/dea_sig_0.05.csv"), row.names = FALSE)