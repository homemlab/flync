#!/usr/bin/env Rscript
#
# Ballgown Differential Gene Expression Analysis
#
# This script performs differential expression analysis using the Ballgown R package.
# It requires:
#   - StringTie output directories for each sample (containing *.ctab files)
#   - A metadata CSV file with sample information and condition labels
#
# Usage:
#   Rscript ballgown_dge.R <cov_dir> <metadata_csv> <output_dir>
#
# Arguments:
#   cov_dir: Directory containing sample subdirectories with StringTie output
#   metadata_csv: CSV file with columns: sample_id, condition, [optional: replicate, etc.]
#   output_dir: Directory to save DGE results
#

# Load required libraries
suppressPackageStartupMessages({
    library(ballgown)
    library(matrixStats)  # For rowVars function
})

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
    cat("Usage: Rscript ballgown_dge.R <cov_dir> <metadata_csv> <output_dir>\n")
    cat("\nArguments:\n")
    cat("  cov_dir:       Directory containing sample subdirectories with StringTie output\n")
    cat("  metadata_csv:  CSV file with sample_id and condition columns\n")
    cat("  output_dir:    Directory to save DGE results\n")
    quit(status = 1)
}

cov_dir <- args[1]
metadata_csv <- args[2]
output_dir <- args[3]

# Create output directory
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Log file setup
log_file <- file.path(output_dir, "ballgown_dge.log")
log_conn <- file(log_file, open = "wt")
sink(log_conn, type = "output")
sink(log_conn, type = "message")

cat(rep("=", 80), "\n", sep = "")
cat("Ballgown Differential Gene Expression Analysis\n")
cat(rep("=", 80), "\n", sep = "")
cat("\n")

cat("Configuration:\n")
cat("  Coverage directory: ", cov_dir, "\n")
cat("  Metadata file:      ", metadata_csv, "\n")
cat("  Output directory:   ", output_dir, "\n\n")

# Read metadata
cat("Reading metadata...\n")
metadata <- read.csv(metadata_csv, stringsAsFactors = FALSE)

# Validate required columns
if (!("sample_id" %in% colnames(metadata))) {
    stop("Metadata CSV must contain 'sample_id' column")
}

if (!("condition" %in% colnames(metadata))) {
    stop("Metadata CSV must contain 'condition' column for differential expression")
}

cat("  Samples: ", nrow(metadata), "\n")
cat("  Conditions: ", paste(unique(metadata$condition), collapse = ", "), "\n\n")

# Verify sample directories exist
cat("Verifying sample directories...\n")
sample_dirs <- file.path(cov_dir, metadata$sample_id)
missing_dirs <- sample_dirs[!file.exists(sample_dirs)]

if (length(missing_dirs) > 0) {
    cat("ERROR: Missing sample directories:\n")
    cat(paste("  -", missing_dirs, collapse = "\n"), "\n")
    stop("Not all sample directories exist in coverage directory")
}

cat("  All ", nrow(metadata), " sample directories found\n\n")

# Load Ballgown object
cat("Loading Ballgown object from StringTie output...\n")
tryCatch({
    bg <- ballgown(
        dataDir = cov_dir,
        samplePattern = paste(metadata$sample_id, collapse = "|"),
        pData = metadata
    )
    cat("  Successfully loaded Ballgown object\n")
    cat("  Transcripts: ", nrow(texpr(bg)), "\n")
    cat("  Genes: ", length(unique(geneIDs(bg))), "\n\n")
}, error = function(e) {
    cat("ERROR loading Ballgown object:\n")
    cat(as.character(e), "\n")
    stop("Failed to load Ballgown object from StringTie output")
})

# Filter low-abundance transcripts
cat("Filtering low-abundance transcripts...\n")
cat("  Criteria: rowVars > 1\n")
bg_filt <- subset(bg, "rowVars(texpr(bg)) > 1", genomesubset = TRUE)
cat("  Transcripts after filtering: ", nrow(texpr(bg_filt)), "\n\n")

# Perform differential expression analysis
cat("Performing differential expression analysis...\n")

# Get unique conditions
conditions <- unique(metadata$condition)

if (length(conditions) < 2) {
    cat("WARNING: Only one condition found (", conditions[1], "). Skipping DGE analysis.\n")
    cat("For differential expression, provide at least two different conditions in metadata.\n\n")
    
    # Save summary
    summary_df <- data.frame(
        analysis = "ballgown_dge",
        status = "skipped",
        reason = "single_condition",
        samples = nrow(metadata),
        conditions = length(conditions),
        stringsAsFactors = FALSE
    )
    write.csv(summary_df, file.path(output_dir, "dge_summary.csv"), row.names = FALSE)
    
    cat("Analysis complete (no DGE performed)\n")
    quit(status = 0)
}

# Transcript-level differential expression
cat("  Analyzing transcript-level differences...\n")
tryCatch({
    results_transcripts <- stattest(
        bg_filt,
        feature = "transcript",
        covariate = "condition",
        adjustvars = NULL,
        getFC = TRUE,
        meas = "FPKM"
    )
    
    # Add gene names
    results_transcripts <- data.frame(
        results_transcripts,
        gene_name = geneNames(bg_filt),
        gene_id = geneIDs(bg_filt)
    )
    
    # Sort by p-value
    results_transcripts <- results_transcripts[order(results_transcripts$pval), ]
    
    # Save results
    output_file <- file.path(output_dir, "transcript_dge_results.csv")
    write.csv(results_transcripts, output_file, row.names = FALSE)
    cat("    Saved transcript results to: ", output_file, "\n")
    
    # Report significant transcripts
    sig_transcripts <- sum(results_transcripts$qval < 0.05, na.rm = TRUE)
    cat("    Significant transcripts (q < 0.05): ", sig_transcripts, "\n")
    
}, error = function(e) {
    cat("    ERROR in transcript-level analysis:\n")
    cat("    ", as.character(e), "\n")
})

# Gene-level differential expression
cat("  Analyzing gene-level differences...\n")
tryCatch({
    results_genes <- stattest(
        bg_filt,
        feature = "gene",
        covariate = "condition",
        adjustvars = NULL,
        getFC = TRUE,
        meas = "FPKM"
    )
    
    # Sort by p-value
    results_genes <- results_genes[order(results_genes$pval), ]
    
    # Save results
    output_file <- file.path(output_dir, "gene_dge_results.csv")
    write.csv(results_genes, output_file, row.names = FALSE)
    cat("    Saved gene results to: ", output_file, "\n")
    
    # Report significant genes
    sig_genes <- sum(results_genes$qval < 0.05, na.rm = TRUE)
    cat("    Significant genes (q < 0.05): ", sig_genes, "\n")
    
}, error = function(e) {
    cat("    ERROR in gene-level analysis:\n")
    cat("    ", as.character(e), "\n")
})

cat("\n")

# Create summary report
cat("Creating summary report...\n")
summary_df <- data.frame(
    analysis = "ballgown_dge",
    status = "complete",
    samples = nrow(metadata),
    conditions = length(conditions),
    total_transcripts = nrow(texpr(bg)),
    filtered_transcripts = nrow(texpr(bg_filt)),
    sig_transcripts = if (exists("sig_transcripts")) sig_transcripts else NA,
    sig_genes = if (exists("sig_genes")) sig_genes else NA,
    stringsAsFactors = FALSE
)

summary_file <- file.path(output_dir, "dge_summary.csv")
write.csv(summary_df, summary_file, row.names = FALSE)
cat("  Saved summary to: ", summary_file, "\n\n")

# Generate simple MA plot if ggplot2 is available
tryCatch({
    suppressPackageStartupMessages(library(ggplot2))
    
    if (exists("results_transcripts") && nrow(results_transcripts) > 0) {
        cat("Generating MA plot...\n")
        
        # Prepare data
        plot_data <- data.frame(
            avg_expr = log10(results_transcripts$fc + 1),
            log_fc = log2(results_transcripts$fc + 0.1),
            significant = results_transcripts$qval < 0.05
        )
        
        # Remove infinite values
        plot_data <- plot_data[is.finite(plot_data$avg_expr) & is.finite(plot_data$log_fc), ]
        
        # Create plot
        p <- ggplot(plot_data, aes(x = avg_expr, y = log_fc, color = significant)) +
            geom_point(alpha = 0.5, size = 1) +
            scale_color_manual(values = c("FALSE" = "gray", "TRUE" = "red")) +
            labs(
                title = "MA Plot - Transcript Differential Expression",
                x = "Log10(Average Expression + 1)",
                y = "Log2(Fold Change)",
                color = "FDR < 0.05"
            ) +
            theme_minimal() +
            theme(legend.position = "top")
        
        # Save plot
        plot_file <- file.path(output_dir, "transcript_ma_plot.png")
        ggsave(plot_file, p, width = 8, height = 6, dpi = 300)
        cat("  Saved MA plot to: ", plot_file, "\n\n")
    }
}, error = function(e) {
    cat("NOTE: Could not generate MA plot (ggplot2 may not be installed)\n\n")
})

cat(rep("=", 80), "\n", sep = "")
cat("Ballgown DGE analysis complete!\n")
cat(rep("=", 80), "\n", sep = "")

# Close log
sink(type = "message")
sink(type = "output")
close(log_conn)

# Print summary to stdout (after closing log redirect)
cat("\nResults saved to:", output_dir, "\n")
cat("  - transcript_dge_results.csv\n")
cat("  - gene_dge_results.csv\n")
cat("  - dge_summary.csv\n")
cat("  - ballgown_dge.log\n\n")
