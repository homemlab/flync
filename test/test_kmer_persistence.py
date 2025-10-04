import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from features.feature_wrapper import FeatureWrapper


def write_dummy_fasta(tmp_path):
    fasta_path = os.path.join(tmp_path, "seqs.fasta")
    with open(fasta_path, "w") as f:
        for i in range(5):
            f.write(f">tx{i}\n")
            f.write("A" * 60 + "\n")
    return fasta_path


def test_kmer_save_and_load(tmp_path):
    fw = FeatureWrapper(quiet=True, show_progress=False)
    fasta = write_dummy_fasta(tmp_path)
    base = os.path.join(tmp_path, "kmer_raw")
    df_or_tuple = fw.run_kmer(
        input_path=fasta,
        k_min=3,
        k_max=3,
        output_format="sparse_dataframe",
        return_sparse_paths=True,
        sparse_base_name=base,
    )
    assert isinstance(df_or_tuple, tuple)
    kmer_df, paths = df_or_tuple
    # Files should exist
    for key in ["sparse_matrix", "rows", "cols"]:
        assert os.path.exists(paths[key])

    # Use aggregate_features with saved paths (simulate later run)
    # Build minimal other feature frames
    ids = kmer_df.index.tolist()
    bwq_df = pd.DataFrame({"transcript_id": ids, "bwq_val": np.arange(len(ids))})
    cpat_df = pd.DataFrame({
        "transcript_id": ids,
        "coding_prob": np.linspace(0.1, 0.9, len(ids)),
        "fickett_score": np.random.rand(len(ids)),
        "hexamer_score": np.random.rand(len(ids)),
        "orf_len": np.random.randint(50, 100, len(ids)),
        "Sequence": ["A"*60]*len(ids),
    })
    mfe_df = pd.DataFrame({"transcript_id": ids, "mfe": np.random.randn(len(ids)), "structure": ["."*10]*len(ids)})

    merged = fw.aggregate_features(
        bwq_df=bwq_df,
        cpat_df=cpat_df,
        mfe_df=mfe_df,
        kmer_df=None,
        kmer_sparse_paths=paths,
        use_dim_redux=True,
        redux_n_components=1,
        use_tfidf=False,
        sparse=True,
        group_kmer_redux_by_length=True,
    )
    assert "bwq_val" in merged.columns
    assert any(col.startswith("cpat_") for col in merged.columns)
    # Transformed sparse stored when utilities available
    if "kmer_transformed_sparse" in merged.attrs:
        assert merged.attrs["kmer_transformed_sparse"].shape[0] == len(ids)
