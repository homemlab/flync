"""Workflow package initialization."""

from .main_workflow import run_pipeline

__all__ = ["run_pipeline"]


def run_pipeline(config: PipelineConfig) -> bool:
    """Run the complete FLYNC pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        True if pipeline succeeded, False otherwise
    """
    logger = get_logger("flync.workflow")
    
    # Create output directory
    config.output.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = Pipeline(config)
    
    # Add pipeline steps in order
    pipeline.add_step(GenomePreparationStep(config))
    pipeline.add_step(InputPreparationStep(config))
    pipeline.add_step(MappingStep(config))
    pipeline.add_step(AssemblyStep(config))
    pipeline.add_step(CodingProbabilityStep(config))
    
    # Only add differential expression if metadata is provided
    if config.input.metadata:
        pipeline.add_step(DifferentialExpressionStep(config))
    
    pipeline.add_step(FeatureExtractionStep(config))
    pipeline.add_step(PredictionStep(config))
    
    # Determine starting step for resume functionality
    start_step = 0
    if config.resume:
        start_step = find_resume_step(config)
        logger.info(f"Resuming pipeline from step {start_step}")
    
    # Run pipeline
    if config.dry_run:
        logger.info("Dry run mode - showing pipeline steps:")
        for i, step in enumerate(pipeline.steps):
            logger.info(f"Step {i}: {step.step_name}")
        return True
    
    return pipeline.run(start_from_step=start_step)


def find_resume_step(config: PipelineConfig) -> int:
    """Find the appropriate step to resume from based on existing outputs.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Step index to resume from (0-based)
    """
    logger = get_logger("flync.workflow")
    output_dir = config.output.output_dir
    
    # Check for various output files to determine completion
    checkpoints = [
        (output_dir / "genome" / "genome.fa", 0),  # Genome preparation
        (output_dir / "sra_info.txt", 1),  # Input preparation (for SRA mode)
        (output_dir / "mapped", 2),  # Mapping
        (output_dir / "assembled", 3),  # Assembly
        (output_dir / "results" / "new-non-coding.gtf", 4),  # Coding probability
        (output_dir / "results" / "ballgown_results.csv", 5),  # Differential expression
        (output_dir / "results" / "non-coding" / "features", 6),  # Feature extraction
        (output_dir / "results" / "final_predictions.csv", 7),  # Prediction
    ]
    
    # Find the last completed step
    last_completed = -1
    for checkpoint_file, step_index in checkpoints:
        if checkpoint_file.exists():
            last_completed = step_index
    
    # Resume from next step
    resume_step = last_completed + 1
    logger.info(f"Last completed step: {last_completed}, resuming from step: {resume_step}")
    
    return min(resume_step, 7)  # Don't exceed max steps
