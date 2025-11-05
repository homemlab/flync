import os
import logging
from collections import defaultdict
from typing import List, Tuple, Union, Dict, Optional

import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

logger = logging.getLogger(__name__)


def dim_redux(sparse_matrix: csr_matrix, row_names: List[str], col_names: List[str], n_components: int = 2, verbose: bool = True):
    if verbose:
        logger.info(f"Performing dimensionality reduction with {n_components} components.")
    try:
        svd = TruncatedSVD(n_components=n_components)
        reduced_matrix = svd.fit_transform(sparse_matrix)
        if verbose:
            logger.info(f"Reduced matrix shape: {reduced_matrix.shape}")
        return reduced_matrix, row_names, col_names
    except Exception as e:
        if verbose:
            logger.error(f"Error during dimensionality reduction: {e}")
        return None, None, None


def dim_redux_by_kmer_length(sparse_matrix: csr_matrix, col_names: List[str], n_components_per_k: Union[int, Dict[int, int]] = 1, verbose: bool = True):
    kmer_groups: Dict[int, List[int]] = defaultdict(list)
    for idx, name in enumerate(col_names):
        kmer_groups[len(name)].append(idx)

    reduced_matrices = []
    reduced_col_names: List[str] = []
    for k, idxs in sorted(kmer_groups.items()):
        if verbose:
            logger.info(f"Reducing {len(idxs)} columns of {k}-mers")
        submatrix = sparse_matrix[:, idxs]
        n_comp = n_components_per_k[k] if isinstance(n_components_per_k, dict) and k in n_components_per_k else n_components_per_k
        # Handle tiny submatrix columns
        if submatrix.shape[1] <= 1:
            reduced = submatrix.toarray()
            reduced_matrices.append(reduced)
            reduced_col_names.extend([f"{k}mer_SVD1"])
            continue
        svd = TruncatedSVD(n_components=min(int(n_comp), submatrix.shape[1]-1))
        reduced = svd.fit_transform(submatrix)
        reduced_matrices.append(reduced)
        reduced_col_names.extend([f"{k}mer_SVD{i+1}" for i in range(reduced.shape[1])])
    reduced_matrix = np.hstack(reduced_matrices) if reduced_matrices else np.empty((sparse_matrix.shape[0], 0))
    return reduced_matrix, reduced_col_names


def load_kmer_results(base_path: str, redux_n_components: Union[int, Dict[int, int]], redux: bool = True, group_redux_kmer_len: bool = True, tfidf: bool = True, verbose: bool = True):
    logger = logging.getLogger(__name__)
    sparse_matrix_file = f"{base_path}_sparse.npz"
    rows_file = f"{base_path}_rows.txt"
    cols_file = f"{base_path}_cols.txt"

    if not (os.path.exists(sparse_matrix_file) and os.path.exists(rows_file) and os.path.exists(cols_file)):
        binary_sparse_matrix_file = f"{base_path}_binary_sparse.npz"
        binary_rows_file = f"{base_path}_binary_rows.txt"
        binary_cols_file = f"{base_path}_binary_cols.txt"
        if os.path.exists(binary_sparse_matrix_file) and os.path.exists(binary_rows_file) and os.path.exists(binary_cols_file):
            sparse_matrix_file = binary_sparse_matrix_file
            rows_file = binary_rows_file
            cols_file = binary_cols_file
            if verbose:
                logger.info(f"Loading binary k-mer results from: {base_path}_binary*")
        else:
            if verbose:
                logger.error(f"One or more result files not found for base path '{base_path}' (tried with and without '_binary' suffix).")
            return None, None, None

    if verbose and sparse_matrix_file.startswith(base_path + "_binary"):
        pass
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
    except FileNotFoundError as e:
        if verbose:
            logger.error(f"File not found: {e}")
        return None, None, None

    if redux:
        try:
            if group_redux_kmer_len:
                reduced_matrix, col_names = dim_redux_by_kmer_length(sparse_matrix, col_names, n_components_per_k=redux_n_components, verbose=verbose)
                if verbose:
                    logger.info(f"Dimensionality reduction by k-mer length completed. Reduced matrix shape: {reduced_matrix.shape}")
            else:
                reduced_matrix, row_names, col_names = dim_redux(sparse_matrix, row_names, col_names, n_components=redux_n_components, verbose=verbose)
                if verbose:
                    logger.info(f"Dimensionality reduction completed. Reduced matrix shape: {reduced_matrix.shape}")
        except Exception as e:
            if verbose:
                logger.error(f"Dimensionality reduction failed: {e}")
            return None, None, None
        if reduced_matrix is not None:
            sparse_matrix = reduced_matrix
        else:
            if verbose:
                logger.error("Dimensionality reduction failed.")
            return None, None, None

    if tfidf:
        if verbose:
            logger.info("Applying TF-IDF transformation.")
        tfidf_transformer = TfidfTransformer()
        sparse_matrix = tfidf_transformer.fit_transform(sparse_matrix)
        if verbose:
            logger.info(f"TF-IDF transformed matrix shape: {sparse_matrix.shape}")

    return sparse_matrix, row_names, col_names


def apply_kmer_transformations(kmer_sparse: csr_matrix, ids: List[str], kmer_names: List[str], use_dim_redux: bool, redux_n_components: Union[int, Dict[int, int]], use_tfidf: bool, sparse: bool, group_redux_kmer_len: bool = True):
    transformed = kmer_sparse
    transformed_names = kmer_names

    if use_tfidf:
        tfidf = TfidfTransformer()
        transformed = tfidf.fit_transform(transformed)
    
    if use_dim_redux:
        if group_redux_kmer_len:
            reduced_matrix, reduced_names = dim_redux_by_kmer_length(transformed, transformed_names, n_components_per_k=redux_n_components, verbose=True)
            transformed = reduced_matrix
            transformed_names = reduced_names
        else:
            reduced_matrix, _, _ = dim_redux(transformed, ids, transformed_names, n_components=int(redux_n_components) if not isinstance(redux_n_components, dict) else 1, verbose=True)
            transformed = reduced_matrix
            transformed_names = [f"SVD_{i}" for i in range(transformed.shape[1])] if transformed is not None else []

    import pandas as pd
    if sparse and hasattr(transformed, 'shape') and not isinstance(transformed, np.ndarray):
        return transformed, transformed_names
    else:
        if not isinstance(transformed, np.ndarray):
            transformed = transformed.toarray()
        df = pd.DataFrame(transformed, index=ids, columns=transformed_names)
        return df, transformed_names
