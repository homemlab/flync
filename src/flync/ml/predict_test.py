"""Minimal smoke tests for predictor module.

This file previously contained a syntax error due to an incomplete import.
Conda's byte-compilation reported it but did not fail the build. We keep a
very small executable snippet here so packaging/import checks succeed cleanly.
"""

from .predictor import predict_lncrna  # type: ignore  # Avoid circular type refs


def _smoke() -> bool:
    """Return True if predictor symbol is importable.

    We intentionally avoid executing heavy feature extraction; this runs during
    conda recipe test import phase to ensure module loads without syntax errors.
    """
    return callable(predict_lncrna)


if __name__ == "__main__":  # pragma: no cover
    ok = _smoke()
    print(f"predict_test smoke={ok}")
