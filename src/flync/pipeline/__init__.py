"""Base classes for FLYNC pipeline steps."""

import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import PipelineConfig
from ..utils import LoggerMixin


class PipelineStep(ABC, LoggerMixin):
    """Abstract base class for pipeline steps."""
    
    def __init__(self, config: PipelineConfig, step_name: str):
        self.config = config
        self.step_name = step_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    @abstractmethod
    def run(self) -> bool:
        """Execute the pipeline step.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate that all required inputs are available.
        
        Returns:
            True if inputs are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_outputs(self) -> Dict[str, Path]:
        """Get the output files produced by this step.
        
        Returns:
            Dictionary mapping output names to file paths
        """
        pass
    
    def execute(self) -> bool:
        """Execute the pipeline step with timing and logging."""
        self.logger.info(f"Starting step: {self.step_name}")
        self.start_time = time.time()
        
        try:
            # Validate inputs
            if not self.validate_inputs():
                self.logger.error(f"Input validation failed for step: {self.step_name}")
                return False
            
            # Run the step
            success = self.run()
            
            if success:
                self.logger.info(f"Successfully completed step: {self.step_name}")
            else:
                self.logger.error(f"Step failed: {self.step_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in step {self.step_name}: {str(e)}")
            return False
        finally:
            self.end_time = time.time()
            if self.start_time:
                duration = self.end_time - self.start_time
                self.logger.info(f"Step {self.step_name} took {duration:.2f} seconds")
    
    def run_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        capture_output: bool = False
    ) -> subprocess.CompletedProcess:
        """Run a shell command with proper logging.
        
        Args:
            command: Command and arguments as list
            cwd: Working directory for command
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            Completed process object
        """
        cmd_str = " ".join(str(arg) for arg in command)
        self.logger.debug(f"Running command: {cmd_str}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                self.logger.debug(f"Command succeeded: {cmd_str}")
            else:
                self.logger.error(f"Command failed with code {result.returncode}: {cmd_str}")
                if capture_output and result.stderr:
                    self.logger.error(f"Command stderr: {result.stderr}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running command {cmd_str}: {str(e)}")
            raise


class Pipeline(LoggerMixin):
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.steps: List[PipelineStep] = []
        self.current_step = 0
    
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline."""
        self.steps.append(step)
    
    def run(self, start_from_step: int = 0) -> bool:
        """Run the pipeline from a specific step.
        
        Args:
            start_from_step: Step index to start from (0-based)
            
        Returns:
            True if all steps succeeded, False otherwise
        """
        self.logger.info(f"Starting FLYNC pipeline with {len(self.steps)} steps")
        
        for i, step in enumerate(self.steps[start_from_step:], start_from_step):
            self.current_step = i
            
            if not step.execute():
                self.logger.error(f"Pipeline failed at step {i}: {step.step_name}")
                return False
        
        self.logger.info("Pipeline completed successfully")
        return True
    
    def get_step_outputs(self, step_name: str) -> Optional[Dict[str, Path]]:
        """Get outputs from a specific step by name."""
        for step in self.steps:
            if step.step_name == step_name:
                return step.get_outputs()
        return None
