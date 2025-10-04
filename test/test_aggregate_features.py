import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from features.feature_wrapper import FeatureWrapper

def build_dummy_inputs():
    ids = [f"tx{i}" for i in range(5)]
    bwq = pd.DataFrame({"transcript_id": ids, "bwq_val": np.arange(5)})
    cpat = pd.DataFrame({
        "transcript_id": ids,
        "coding_prob": np.linspace(0.1, 0.9, 5),
        "fickett_score": np.random.rand(5),
        "hexamer_score": np.random.rand(5),
        "orf_len": np.random.randint(100, 500, 5),
        "Sequence": ["A"*50]*5,
    })
    mfe = pd.DataFrame({"transcript_id": ids, "mfe": np.random.randn(5), "structure": ["."*10]*5})
    # 5 sequences x 8 k-mers (pretend mixture of lengths)
    kmer_sparse = csr_matrix(np.random.randint(0,4,(5,8)))
    return bwq, cpat, mfe, kmer_sparse


def test_aggregate_features_basic():
    bwq, cpat, mfe, kmer_sparse = build_dummy_inputs()
    fw = FeatureWrapper(quiet=True, show_progress=False)
    merged = fw.aggregate_features(
        bwq_df=bwq,
        cpat_df=cpat,
        mfe_df=mfe,
        kmer_df=kmer_sparse,
        use_dim_redux=True,
        redux_n_components=1,
        use_tfidf=False,
        sparse=False,
        group_kmer_redux_by_length=True,
    )
    assert "cpat_coding_prob" in merged.columns
    assert "ss_mfe" in merged.columns
    # Either SVD or group naming
    assert any(col.startswith("SVD_") or col.endswith("mer_SVD1") for col in merged.columns)
    assert len(merged) == 5
