"""
High-level predictor module for lncRNA classification

This module provides a simplified interface to the FLYNC ML pipeline,
integrating feature extraction and prediction into a single function call.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import click
import tempfile
import os

# Import from the migrated modules
from flync.features.feature_cleaning import FeatureCleaning
from flync.ml.ebm_predictor import EBMPredictorWithScaler


def predict_lncrna(
    gtf_file: str,
    model_file: str,
    output_file: str,
    ref_genome: Optional[str] = None,
    bwq_config: Optional[str] = None,
    scaler_path: Optional[str] = None,
    schema_path: Optional[str] = None,
    threads: int = 8,
    cache_dir: Optional[str] = None,
    clear_cache: bool = False,
    cov_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Predict lncRNA candidates from a GTF file.

    This function performs the complete ML inference pipeline:
    1. Feature extraction from GTF and genomic tracks
    2. Feature cleaning and alignment to model schema
    3. ML-based classification with confidence scores

    Parameters
    ----------
    gtf_file : str
        Path to input GTF file (e.g., merged.gtf or merged-new-transcripts.gtf)
    model_file : str
        Path to trained ML model file (.pkl or .model)
    output_file : str
        Path to save predictions (CSV format)
    ref_genome : str, optional
        Path to reference genome FASTA file
    bwq_config : str, optional
        Path to BigWig query configuration file
    scaler_path : str, optional
        Path to saved feature scaler (if not provided, model directory is searched)
    schema_path : str, optional
        Path to model schema file (if not provided, model directory is searched)
    threads : int
        Number of threads for feature extraction
    cache_dir : str, optional
        Directory for caching downloaded genomic tracks. If None, uses system temp directory.
    clear_cache : bool
        If True, clears the cache directory before starting
    cov_dir : str, optional
        Directory containing coverage GTF files (<sample_id>.rna.gtf or <sample_id>.gtf).
        If None, defaults to the 'cov' folder relative to the input GTF parent directory.
    verbose : bool
        Print progress messages

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and confidence scores
    """

    if verbose:
        click.echo("=" * 60)
        click.echo("FLYNC lncRNA Prediction Pipeline")
        click.echo("=" * 60)

    # Step 1: Feature Extraction
    if verbose:
        click.echo("\n[1/3] Extracting features from GTF...")

    # Use default BWQ config if not provided
    if bwq_config is None:
        # Use bundled config from src/config
        import flync

        package_dir = Path(flync.__file__).parent
        bwq_config = str(package_dir / "config" / "bwq_config.yaml")
        if verbose:
            click.echo(f"  Using bundled BWQ config: {bwq_config}")

    try:
        # Use Python API instead of subprocess
        from flync.features.feature_wrapper import FeatureWrapper

        # Use system temp directory if no cache_dir specified
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "flync_cache")

        # Initialize feature wrapper with appropriate settings
        fw = FeatureWrapper(
            log_level="WARNING" if verbose else "ERROR",
            cache_dir=cache_dir,
            keep_downloaded_files=True,
            clear_cache_on_startup=clear_cache,
            show_progress=verbose,
            quiet=not verbose,
            threads=threads,
        )

        # Run full feature extraction pipeline with default ML parameters
        features_df = fw.run_all(
            gtf_file=gtf_file,
            ref_genome_path=ref_genome,
            config_file=bwq_config,
            k_min=3,
            k_max=12,
            use_dim_redux=True,
            redux_n_components=1,
            use_tfidf=True,
            sparse=False,
            group_kmer_redux_by_length=True,
        )

        if verbose:
            click.echo(f"  Extracted {len(features_df)} transcripts")
            click.echo(f"  Features: {features_df.shape[1]} columns")

            # Debug: Show which columns were extracted
            feature_cols = [
                c
                for c in features_df.columns
                if c
                not in [
                    "transcript_id",
                    "chrom",
                    "start",
                    "end",
                    "strand",
                    "ss_sequence",
                    "ss_structure",
                ]
            ]
            click.echo(
                f"  Feature types: {', '.join(sorted(set([c.split('_')[0] for c in feature_cols if '_' in c])))}"
            )

    except Exception as e:
        click.secho(f"  Error during feature extraction: {str(e)}", fg="red")
        raise

    # Step 2: Feature Cleaning and Schema Alignment
    if verbose:
        click.echo("\n[2/3] Cleaning features and aligning to model schema...")

    try:
        # Infer paths if not provided
        model_dir = Path(model_file).parent

        if schema_path is None:
            # Look for schema in model directory
            schema_candidates = list(model_dir.glob("*_schema.json"))
            if schema_candidates:
                schema_path = str(schema_candidates[0])
                if verbose:
                    click.echo(f"  Found schema: {schema_path}")
            else:
                # Fallback to bundled assets
                import flync

                assets_dir = Path(flync.__file__).parent / "assets"
                candidate = assets_dir / "flync_ebm_model_schema.json"
                if candidate.exists():
                    schema_path = str(candidate)
                    if verbose:
                        click.echo(f"  Using bundled schema: {schema_path}")

        if scaler_path is None:
            # Look for scaler in model directory
            scaler_candidates = list(model_dir.glob("*_scaler.pkl"))
            if scaler_candidates:
                scaler_path = str(scaler_candidates[0])
                if verbose:
                    click.echo(f"  Found scaler: {scaler_path}")
            else:
                # Fallback to bundled assets
                import flync

                assets_dir = Path(flync.__file__).parent / "assets"
                candidate = assets_dir / "flync_ebm_scaler.pkl"
                if candidate.exists():
                    scaler_path = str(candidate)
                    if verbose:
                        click.echo(f"  Using bundled scaler: {scaler_path}")

        # Clean features for inference using proper pipeline
        # This will apply multi-hot encoding, schema alignment, AND scaling
        from flync.features.feature_cleaning import (
            PipelineConfig,
            execute_data_pipeline,
        )

        # Extract metadata from GTF file and features_df
        # Note: FeatureWrapper drops BED coordinate columns (chromosome, start, end, strand)
        # to prevent data leakage, so we must extract them from the GTF directly
        metadata_df = _extract_metadata_from_gtf_and_features(gtf_file, features_df)

        # The cleaning pipeline expects an "index" column containing transcript IDs
        if "transcript_id" in features_df.columns:
            features_df = features_df.rename(columns={"transcript_id": "index"})

        # Save features to temporary file for feature_cleaning pipeline
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", delete=False, mode="wb"
        ) as tmp_features:
            features_temp_path = tmp_features.name
            features_df.to_parquet(features_temp_path)

        # Create temporary output file
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", delete=False, mode="wb"
        ) as tmp_cleaned:
            cleaned_temp_path = tmp_cleaned.name

        try:
            # Create config for inference mode
            # The cleaning pipeline will:
            # 1. Apply multi-hot encoding
            # 2. Align to model schema (keep only expected features)
            # 3. Apply the scaler
            config = PipelineConfig(
                dataset_path=Path(features_temp_path),
                cleaned_output_path=Path(cleaned_temp_path),
                metadata_path=Path(schema_path) if schema_path else None,
                scaler_path=Path(scaler_path)
                if scaler_path
                else None,  # Scaler applied here
                enable_multi_hot=True,  # CRITICAL: Model was trained with multi-hot encoding
                mode="inference",
            )

            # Run inference preparation (includes multi-hot encoding + schema alignment + scaling)
            result = execute_data_pipeline(config)
            cleaned_df = (
                result.inference_data
                if hasattr(result, "inference_data")
                and result.inference_data is not None
                else result.cleaned_df
            )

            if verbose:
                click.echo(f"  Cleaned features: {cleaned_df.shape}")
                click.echo(
                    f"  Transcripts after cleaning: {len(cleaned_df)} (filtered {len(features_df) - len(cleaned_df)})"
                )
                click.echo(f"  Final feature columns: {len(cleaned_df.columns)}")

                # Debug: Check if index was preserved
                if hasattr(cleaned_df.index, "name"):
                    click.echo(f"  Index name: {cleaned_df.index.name}")

        finally:
            # Clean up temporary files
            if os.path.exists(features_temp_path):
                os.unlink(features_temp_path)
            if os.path.exists(cleaned_temp_path):
                os.unlink(cleaned_temp_path)

    except Exception as e:
        click.secho(f"  Error during feature cleaning: {str(e)}", fg="red")
        raise

    # Step 3: ML Prediction
    if verbose:
        click.echo("\n[3/3] Running ML prediction...")

    try:
        # Initialize predictor WITHOUT scaler (scaling already applied in cleaning pipeline)
        if schema_path is None:
            raise RuntimeError(
                "Model schema not found. Please provide --schema-path or ensure bundled schema exists."
            )

        # Use base EBMPredictor since data is already scaled
        from flync.ml.ebm_predictor import EBMPredictor

        predictor = EBMPredictor(
            model_path=model_file,
            schema_path=schema_path,
            auto_validate=False,  # Already validated in cleaning pipeline
        )

        # Get class predictions and probabilities (data is already scaled)
        preds = predictor.predict(cleaned_df, validate=False)
        probas = predictor.predict_proba(cleaned_df, validate=False)

        # Build output DataFrame (only include prob_lncrna, not prob_coding)
        # Use cleaned_df index to maintain alignment with surviving transcripts
        predictions_df = pd.DataFrame(
            {
                "prediction": preds,
                "prob_lncrna": probas[:, 1]
                if probas.ndim == 2 and probas.shape[1] > 1
                else 1 - probas.ravel(),
            },
            index=cleaned_df.index,
        )

        # Add transcript_id column from index
        predictions_df.insert(0, "transcript_id", predictions_df.index)

        # Join metadata using the index (only keeps transcripts that survived cleaning)
        if len(metadata_df) > 0:
            # Reindex metadata_df to match cleaned_df index (only surviving transcripts)
            metadata_aligned = metadata_df.reindex(cleaned_df.index)

            # Add metadata columns to predictions
            for col in metadata_aligned.columns:
                predictions_df[col] = metadata_aligned[col].values

        if verbose:
            n_lncrna = int((predictions_df["prediction"] == 1).sum())
            n_coding = int((predictions_df["prediction"] == 0).sum())
            click.echo(f"  Predicted lncRNAs: {n_lncrna}")
            click.echo(f"  Predicted coding: {n_coding}")

        # Defer saving/returning until after enrichment step

    except Exception as e:
        click.secho(f"  Error during prediction: {str(e)}", fg="red")
        raise

    # Augment predictions with FPKM values if coverage directory is available
    try:
        if verbose:
            click.echo("\n[4/4] Augmenting predictions with FPKM values...")

        # Use provided cov_dir or infer from GTF path (standard pipeline structure)
        if cov_dir is None:
            cov_dir = Path(gtf_file).parent.parent / "cov"
        else:
            cov_dir = Path(cov_dir)

        # Collect per-sample FPKM from coverage GTF files
        fpkm_df = _collect_fpkm_from_gtf(
            cov_dir, transcript_ids=predictions_df.get("transcript_id")
        )

        # Merge FPKM data
        out_df = predictions_df
        if fpkm_df is not None and not fpkm_df.empty:
            out_df = out_df.merge(fpkm_df, how="left", on="transcript_id")
            if verbose:
                fpkm_cols = [
                    c for c in fpkm_df.columns if c.upper().startswith("FPKM_")
                ]
                click.echo(f"  Added {len(fpkm_cols)} FPKM columns")

        # Save final
        out_df.to_csv(output_file, index=False)

        if verbose:
            click.echo(f"\n✓ Predictions saved to: {output_file}")

        return out_df

    except Exception as e:
        # Non-fatal enrichment failure: keep base predictions
        click.secho(
            f"  Warning: failed to enrich predictions with FPKM: {e}",
            fg="yellow",
        )
        predictions_df.to_csv(output_file, index=False)
        if verbose:
            click.echo(f"\n✓ Predictions saved (base) to: {output_file}")
        return predictions_df


def predict_from_features(
    features_file: str,
    model_file: str,
    output_file: str,
    scaler_path: Optional[str] = None,
    schema_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Predict lncRNA candidates from pre-extracted features.

    This is a shortcut function for when features have already been extracted
    and saved to a Parquet or CSV file.

    Parameters
    ----------
    features_file : str
        Path to features file (Parquet or CSV)
    model_file : str
        Path to trained ML model file
    output_file : str
        Path to save predictions
    scaler_path : str, optional
        Path to saved feature scaler
    schema_path : str, optional
        Path to model schema file
    verbose : bool
        Print progress messages

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and confidence scores
    """

    if verbose:
        click.echo("Loading pre-extracted features...")

    # Load features
    features_path = Path(features_file)
    if features_path.suffix == ".parquet":
        features_df = pd.read_parquet(features_file)
    else:
        features_df = pd.read_csv(features_file)

    if verbose:
        click.echo(
            f"  Loaded {len(features_df)} transcripts with {features_df.shape[1]} features"
        )

    # Clean and predict
    if verbose:
        click.echo("Cleaning features...")

    cleaner = FeatureCleaning(mode="inference")
    cleaned_df = cleaner.clean(
        features_df, metadata_path=schema_path, scaler_path=scaler_path
    )

    if verbose:
        click.echo("Running prediction...")

    if schema_path is None:
        raise RuntimeError(
            "Model schema not found. Please provide schema_path for prediction from features."
        )

    predictor = EBMPredictorWithScaler(
        model_path=model_file, schema_path=schema_path, scaler_path=scaler_path
    )

    preds, validation = predictor.predict(cleaned_df, return_validation=True)
    probas, _ = predictor.predict_proba(cleaned_df, return_validation=True)

    predictions_df = pd.DataFrame(
        {
            "prediction": preds,
            "prob_coding": probas[:, 0]
            if probas.ndim == 2 and probas.shape[1] > 1
            else probas.ravel(),
            "prob_lncrna": probas[:, 1]
            if probas.ndim == 2 and probas.shape[1] > 1
            else 1 - probas.ravel(),
        }
    )

    # Save predictions
    predictions_df.to_csv(output_file, index=False)

    if verbose:
        n_lncrna = int((predictions_df["prediction"] == 1).sum())
        click.echo(f"\n✓ Predicted {n_lncrna} lncRNAs")
        click.echo(f"  Results saved to: {output_file}")

    return predictions_df


# -----------------------------
# Helper functions
# -----------------------------


def _extract_metadata_from_gtf_and_features(
    gtf_file: str, features_df: pd.DataFrame
) -> pd.DataFrame:
    """Extract metadata from GTF file and features DataFrame.

    Returns DataFrame with transcript_id as index and columns:
    - chromosome, start, end, strand (from GTF)
    - ss_sequence, ss_structure (from features_df if available)

    Parameters
    ----------
    gtf_file : str
        Path to GTF file
    features_df : pd.DataFrame
        Features DataFrame with transcript_id column

    Returns
    -------
    pd.DataFrame
        Metadata with transcript_id as index
    """
    try:
        import pyranges as pr
    except ImportError:
        # If pyranges not available, return empty metadata
        if "transcript_id" in features_df.columns:
            return pd.DataFrame(index=features_df["transcript_id"])
        return pd.DataFrame(index=features_df.index)

    # Read GTF and extract transcript-level features
    gr = pr.read_gtf(gtf_file)
    transcripts = gr.df[gr.df["Feature"] == "transcript"].copy()

    # Select relevant columns and rename to lowercase
    metadata = transcripts[
        ["transcript_id", "Chromosome", "Start", "End", "Strand"]
    ].copy()
    metadata.columns = ["transcript_id", "chromosome", "start", "end", "strand"]
    metadata = metadata.set_index("transcript_id")

    # Add secondary structure columns from features_df if available
    ss_cols = ["ss_sequence", "ss_structure"]
    if "transcript_id" in features_df.columns:
        features_with_ss = features_df.set_index("transcript_id")[
            [col for col in ss_cols if col in features_df.columns]
        ].copy()

        # Merge with GTF metadata
        metadata = metadata.join(features_with_ss, how="left")

    return metadata


def _collect_fpkm_from_gtf(
    cov_dir: Path, transcript_ids: Optional[pd.Series]
) -> Optional[pd.DataFrame]:
    """Collect per-sample FPKM from coverage GTF files and compute mean/std.

    Looks for GTF files matching patterns:
    - <cov_dir>/<sample_id>.rna.gtf
    - <cov_dir>/<sample_id>.gtf
    - <cov_dir>/<sample_id>/<sample_id>.rna.gtf (standard pipeline structure)

    Returns DataFrame with columns: transcript_id, FPKM_<sample>..., fpkm_mean, fpkm_std.
    """
    if not cov_dir.exists():
        return None

    wanted = (
        set(transcript_ids.dropna().astype(str)) if transcript_ids is not None else None
    )

    per_sample: list[pd.DataFrame] = []

    # Try to import pyranges for GTF parsing
    try:
        import pyranges as pr
    except ImportError:
        pr = None  # Fallback to manual parsing below

    # Pattern 1: Direct GTF files in cov_dir
    gtf_files = list(cov_dir.glob("*.rna.gtf")) + list(cov_dir.glob("*.gtf"))

    # Pattern 2: GTF files in subdirectories (standard pipeline structure)
    for subdir in cov_dir.iterdir():
        if subdir.is_dir():
            gtf_files.extend(subdir.glob("*.rna.gtf"))
            gtf_files.extend(subdir.glob("*.gtf"))

    # Remove duplicates
    gtf_files = list(set(gtf_files))

    for gtf_path in gtf_files:
        # Extract sample name from filename
        sample_name = gtf_path.stem.replace(".rna", "")

        parsed_ok = False

        # Preferred: pyranges-based parsing
        if pr is not None:
            try:
                gr = pr.read_gtf(str(gtf_path))
                df = gr.df
                if "Feature" in df.columns:
                    tdf = df[df["Feature"] == "transcript"].copy()
                else:
                    tdf = pd.DataFrame()
                if not tdf.empty:
                    # Normalize column names: sometimes attributes are lowercased
                    cols = {c.lower(): c for c in tdf.columns}
                    has_tid = "transcript_id" in tdf.columns
                    has_fpkm = "FPKM" in tdf.columns or ("fpkm" in cols)

                    if not has_tid and "transcript_id" in cols:
                        tdf.rename(
                            columns={cols["transcript_id"]: "transcript_id"},
                            inplace=True,
                        )
                        has_tid = True

                    if "fpkm" in cols and "FPKM" not in tdf.columns:
                        tdf.rename(columns={cols["fpkm"]: "FPKM"}, inplace=True)
                        has_fpkm = True

                    if has_tid and has_fpkm:
                        work = tdf[["transcript_id", "FPKM"]].copy()
                        # Filter to wanted transcript IDs if specified
                        if wanted is not None:
                            work = work[work["transcript_id"].astype(str).isin(wanted)]
                        if not work.empty:
                            sdf = work.groupby("transcript_id", as_index=False)[
                                "FPKM"
                            ].mean()
                            sdf = sdf.rename(columns={"FPKM": f"FPKM_{sample_name}"})
                            per_sample.append(sdf)
                            parsed_ok = True
            except Exception:
                parsed_ok = False

        if parsed_ok:
            continue

        # Fallback: manual attribute parsing without pyranges
        try:
            import csv
            import re

            rows = []
            with open(gtf_path, "r") as fh:
                r = csv.reader(fh, delimiter="\t")
                for parts in r:
                    if not parts or parts[0].startswith("#"):
                        continue
                    if len(parts) < 9:
                        continue
                    feature = parts[2]
                    if feature != "transcript":
                        continue
                    attrs = parts[8]
                    # Extract transcript_id and FPKM via regex
                    m_tid = re.search(r'transcript_id\s+"([^"]+)"', attrs)
                    m_fpkm = re.search(r'FPKM\s+"([^"]+)"', attrs)
                    if not m_tid or not m_fpkm:
                        continue
                    tid = m_tid.group(1)
                    try:
                        fpkm_val = float(m_fpkm.group(1))
                    except ValueError:
                        continue
                    rows.append((tid, fpkm_val))
            if rows:
                sdf = pd.DataFrame(
                    rows, columns=["transcript_id", f"FPKM_{sample_name}"]
                )
                if wanted is not None:
                    sdf = sdf[sdf["transcript_id"].astype(str).isin(wanted)]
                if not sdf.empty:
                    # Group in case of duplicates
                    sdf = sdf.groupby("transcript_id", as_index=False).mean(
                        numeric_only=True
                    )
                    per_sample.append(sdf)
        except Exception:
            # Skip on failure
            continue

    if not per_sample:
        return None

    # Merge all per-sample frames on transcript_id
    from functools import reduce

    fpkm_df = reduce(
        lambda left, right: pd.merge(left, right, on="transcript_id", how="outer"),
        per_sample,
    )

    # Compute mean/std across FPKM_* columns
    fpkm_cols = [c for c in fpkm_df.columns if str(c).startswith("FPKM_")]
    if fpkm_cols:
        fpkm_df["fpkm_mean"] = fpkm_df[fpkm_cols].mean(axis=1, skipna=True)
        fpkm_df["fpkm_std"] = fpkm_df[fpkm_cols].std(axis=1, ddof=0, skipna=True)

    return fpkm_df
