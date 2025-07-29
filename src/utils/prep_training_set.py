import os
import logging
from scipy.sparse import load_npz
import numpy as np
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfTransformer

def load_kmer_results(base_path, verbose=True):
    """
    Loads k-mer results that were saved in the sparse format
    (matrix.npz, rows.txt, cols.txt).

    Args:
        base_path (str): The base path used when saving (e.g., 'output/my_kmers').
                         The function will look for '{base_path}_sparse.npz', etc.
                         If using binary k-mers, ensure base_path includes the '_binary' tag
                         if it was added during saving (e.g., 'output/my_kmers_binary').
        verbose (bool): If True, log loading messages.
        log_level (str): Logging level to use. Defaults to "INFO".

    Returns:
        tuple: (scipy.sparse.csr_matrix, list of row names/IDs, list of col names/k-mers)
               Returns (None, None, None) if files are not found.
    """
    logger = logging.getLogger(__name__)
    # Determine file paths based on whether it's likely binary or ATGC output
    # This is a simple heuristic; a more robust way would be to pass the exact filenames
    # or have a metadata file.
    sparse_matrix_file = f"{base_path}_sparse.npz"
    rows_file = f"{base_path}_rows.txt"
    cols_file = f"{base_path}_cols.txt"

    if not (os.path.exists(sparse_matrix_file) and os.path.exists(rows_file) and os.path.exists(cols_file)):
         # Try with _binary suffix if primary files not found
         binary_sparse_matrix_file = f"{base_path}_binary_sparse.npz"
         binary_rows_file = f"{base_path}_binary_rows.txt"
         binary_cols_file = f"{base_path}_binary_cols.txt"
         if os.path.exists(binary_sparse_matrix_file) and os.path.exists(binary_rows_file) and os.path.exists(binary_cols_file):
             sparse_matrix_file = binary_sparse_matrix_file
             rows_file = binary_rows_file
             cols_file = binary_cols_file
             if verbose: logger.info(f"Loading binary k-mer results from: {base_path}_binary*")
         else:
            if verbose: logger.error(f"One or more result files not found for base path '{base_path}' (tried with and without '_binary' suffix).")
            return None, None, None

    if verbose and sparse_matrix_file.startswith(base_path + "_binary"):
        pass # Already logged above
    elif verbose:
        logger.info(f"Loading k-mer results from: {base_path}*")


    try:
        sparse_matrix = load_npz(sparse_matrix_file)
        with open(rows_file, 'r') as f:
            row_names = [line.strip() for line in f]
        with open(cols_file, 'r') as f:
            col_names = [line.strip() for line in f]
        if verbose:
            logger.info(f"Loaded sparse matrix ({sparse_matrix.shape}), {len(row_names)} row names, {len(col_names)} column names.")
        return sparse_matrix, row_names, col_names
    except Exception as e:
        if verbose: logger.error(f"Error loading k-mer results: {e}")
        return None, None, None
    
def prep_training_data(
    target,
    base_parquet_path,
    bwq_parquet_path,
    cpat_parquet_path,
    mfe_parquet_path,
    kmer_base_path,
    sparse=True,
    use_tfidf=True
):
    
    base = pd.read_parquet(base_parquet_path)
    # generate a unique ID for each row of each dataframe
    base['id'] = base.apply(lambda x: f"{x['transcript_id']}_{x['exon_number']}_{x['Start']}_{x['End']}" if pd.notnull(x['exon_number']) else f"{x['transcript_id']}_{x['Start']}_{x['End']}", axis=1)
    # get relevant columns from base
    df = base[['id', 'Chromosome', 'Start', 'End', 'length', 'transcript_id', 'exon_number', 'gene_name', 'Sequence']]
    df.columns = [x.lower() for x in df.columns]
    # clear base dataframe
    del base

    # load bwq dataframe and generate a unique ID for each row
    bwq = pd.read_parquet(bwq_parquet_path)
    bwq['id'] = bwq.apply(lambda x: f"{x['name']}_{x['start']}_{x['end']}", axis=1)
    # drop unnecessary columns
    cols_to_drop = ['chromosome', 'start', 'end', 'name']
    bwq = bwq.drop(columns=cols_to_drop)
    # inner join df with bwq and assert that the number of rows is the same
    df = df.merge(bwq, how='inner', left_on=['id'], right_on=['id'])
    del bwq

    cpat = pd.read_parquet(cpat_parquet_path)
    cols_to_keep = ['Transcript_ID', 'Coding_prob', 'Fickett_Score', 'Hexamer_Score', 'ORF_Len']
    cpat = cpat[cols_to_keep]
    cpat.rename(columns={
        'Transcript_ID': 'id',
        'Coding_prob': 'cpat_cod_prob',
        'Fickett_Score': 'cpat_fickett_score',
        'Hexamer_Score': 'cpat_hexamer_score',
        'ORF_Len': 'cpat_orf_len'
    }, inplace=True)
    # inner join df with cpat and assert that the number of rows is the same
    df = df.merge(cpat, how='inner', left_on=['id'], right_on=['id'])
    del cpat

    mfe = pd.read_parquet(mfe_parquet_path)
    mfe['id'] = mfe.apply(lambda x: f"{x['transcript_id']}_{x['exon_number']}_{x['Start']}_{x['End']}" if pd.notnull(x['exon_number']) else f"{x['transcript_id']}_{x['Start']}_{x['End']}", axis=1)
    cols_to_keep = ['id', 'MFE', 'Structure']
    mfe = mfe[cols_to_keep]
    mfe.rename(columns={
        'MFE': 'ss_mfe',
        'Structure': 'ss_structure'
    }, inplace=True)
    df = df.merge(mfe, how='inner', left_on=['id'], right_on=['id'])
    del mfe

    npz_mtx, row_names, col_names = load_kmer_results(kmer_base_path)
    assert len(row_names) == len(set(row_names)), f"Duplicate row names found: {len(row_names) - len(set(row_names))} duplicates"

    # Apply TF-IDF transformation if requested
    if use_tfidf:
        tfidf = TfidfTransformer()
        npz_mtx = tfidf.fit_transform(npz_mtx)

    kmer = pd.DataFrame.sparse.from_spmatrix(npz_mtx, columns=col_names, index=row_names)

    if not sparse:
        try:
            # Convert sparse matrix to dense
            kmer = kmer.sparse.to_dense()
        except Exception as e:
            raise ValueError(f"Error converting sparse matrix to dense: {e}")

    # set index of df as 'id' to merge with kmer sparse matrix
    df.set_index('id', inplace=True)
    df = df.merge(kmer, how='inner', left_index=True, right_index=True)
    del kmer

    if target == 'ncr':
        df['y'] = True
        return df
    elif target == 'pcg':
        df['y'] = False
        return df
    else:
        raise ValueError(f"Invalid target: {target}. Must be 'ncr' or 'pcg'.")

def main():
    args = argparse.ArgumentParser(description="Prepare ML training dataset.")
    args.add_argument("--target", type=str, default="ncr", choices=["ncr", "pcg"], help="Target variable")
    args.add_argument("--base_parquet_path", type=str, required=True, help="Path to base parquet file")
    args.add_argument("--bwq_parquet_path", type=str, required=True, help="Path to bwq parquet file")
    args.add_argument("--cpat_parquet_path", type=str, required=True, help="Path to cpat parquet file")
    args.add_argument("--mfe_parquet_path", type=str, required=True, help="Path to mfe parquet file")
    args.add_argument("--kmer_base_path", type=str, required=True, help="Base path for kmer sparse matrix files (no extension)")
    args.add_argument("--output_path", type=str, required=True, help="Output file path for the training data")
    args.add_argument("--write_dense", action="store_true", help="Try to write dense matrix to disk. WARNING: This may take a long time and use a lot of memory.")
    args.add_argument("--write_sparse", action="store_true", help="Write output as a sparse parquet file (default is dense if --write_dense is set, otherwise sparse).")
    args.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args.add_argument("--no_tfidf", action="store_true", help="Disable TF-IDF transformation of k-mer counts.")
    args = args.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting training set preparation...")
    logger.info(f"Target: {args.target}")
    logger.info(f"Verbose mode: {'enabled' if args.verbose else 'disabled'}")
    # Prepare the training data
    training_data = prep_training_data(
        args.target,
        args.base_parquet_path,
        args.bwq_parquet_path,
        args.cpat_parquet_path,
        args.mfe_parquet_path,
        args.kmer_base_path,
        sparse=not args.write_dense,
        use_tfidf=not args.no_tfidf
    )
    # Save the training data to a file
    try:
        if args.write_dense:
            training_data.to_parquet(args.output_path)
            logger.info(f"Saved dense training data to {args.output_path}")
        elif args.write_sparse:
            training_data.to_parquet(args.output_path, engine="pyarrow", compression="snappy")
            logger.info(f"Saved sparse training data to {args.output_path}")
        else:
            # Default: save as sparse if possible
            training_data.to_parquet(args.output_path, engine="pyarrow", compression="snappy")
            logger.info(f"Saved sparse training data to {args.output_path}")
    except Exception as e:
        logger.error(f"Error saving training data: {e}")

    # Return the training data
    if args.verbose:
        logger.info(f"Training data shape: {training_data.shape}")
    logger.info("Training set preparation complete.")

    return training_data
# Example usage:
# python prep_training_set.py --target ncr --verbose
# python prep_training_set.py --target pcg --verbose

if __name__ == "__main__":
    main()