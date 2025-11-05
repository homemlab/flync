"""
Unified progress reporting interface for CLI and Python APIs.

This module provides a consistent interface for progress tracking across all
feature extraction modules, with automatic TTY detection, logging integration,
and graceful fallbacks.

Example usage:
    >>> from utils.progress import get_progress_manager, create_progress_bar
    >>> 
    >>> # Simple progress bar
    >>> with create_progress_bar(total=100, desc="Processing") as pbar:
    ...     for i in range(100):
    ...         # Do work
    ...         pbar.update(1)
    >>> 
    >>> # Progress manager for complex workflows
    >>> pm = get_progress_manager(show_progress=True)
    >>> main_bar = pm.create_bar(total=3, desc="Main workflow")
    >>> sub_bar = pm.create_bar(total=100, desc="Sub-task", parent_id=main_bar.task_id)
"""

import atexit
import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

# Optional imports with fallbacks
try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import rich
    from rich.console import Console
    from rich.progress import (
        Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn,
        MofNCompleteColumn, SpinnerColumn, TimeElapsedColumn
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Create dummy types for type hints when rich is not available
    class TaskID:
        pass
    class Progress:
        pass

# Thread-local storage for progress state
_local = threading.local()

# Global registry for cleanup
_progress_instances: List['BaseProgressManager'] = []
_cleanup_registered = False


def _register_cleanup():
    """Register cleanup function to run on program exit."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_all)
        _cleanup_registered = True


def _cleanup_all():
    """Clean up all progress instances on program exit."""
    for instance in _progress_instances[:]:  # Copy list to avoid modification during iteration
        try:
            instance.cleanup()
        except Exception:
            pass  # Ignore cleanup errors


class ProgressConfig:
    """Configuration for progress reporting behavior."""
    
    def __init__(
        self,
        enabled: bool = None,
        force_terminal: bool = False,
        prefer_rich: bool = True,
        quiet_mode: bool = False,
        log_milestones: bool = True,
        milestone_intervals: List[float] = None,
    ):
        """
        Initialize progress configuration.
        
        Args:
            enabled: Whether to show progress. None means auto-detect TTY.
            force_terminal: Force terminal output even if not TTY.
            prefer_rich: Prefer rich output over tqdm when available.
            quiet_mode: Suppress all progress output.
            log_milestones: Log milestone messages in non-interactive mode.
            milestone_intervals: Progress percentages to log (default: [25, 50, 75]).
        """
        if enabled is None:
            # Auto-detect: show progress if we're in an interactive terminal
            enabled = (
                not quiet_mode and
                (force_terminal or (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()))
            )
        
        self.enabled = enabled and not quiet_mode
        self.force_terminal = force_terminal
        self.prefer_rich = prefer_rich and RICH_AVAILABLE
        self.quiet_mode = quiet_mode
        self.log_milestones = log_milestones
        self.milestone_intervals = milestone_intervals or [25, 50, 75]


class BaseProgressBar(ABC):
    """Abstract base class for progress bars."""
    
    def __init__(self, total: Optional[int], desc: str, task_id: Optional[str] = None, unit: str = 'it'):
        self.total = total
        self.desc = desc
        self.task_id = task_id or f"task_{id(self)}"
        self.current = 0
        self.completed = False
        self.unit = unit
        self._start_time = time.time()
        self._milestone_logged = set()
        
    @abstractmethod
    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        pass
    
    @abstractmethod
    def set_description(self, desc: str) -> None:
        """Update the description."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the progress bar."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _log_milestone(self, config: ProgressConfig):
        """Log progress milestones if enabled."""
        if not config.log_milestones or self.total is None:
            return
            
        progress_pct = (self.current / self.total) * 100
        
        for milestone in config.milestone_intervals:
            if milestone not in self._milestone_logged and progress_pct >= milestone:
                elapsed = time.time() - self._start_time
                logging.info(
                    f"{self.desc}: {milestone}% complete "
                    f"({self.current:,}/{self.total:,}) - "
                    f"elapsed: {elapsed:.1f}s"
                )
                self._milestone_logged.add(milestone)


class TqdmProgressBar(BaseProgressBar):
    """Progress bar implementation using tqdm."""
    
    def __init__(self, total: Optional[int], desc: str, task_id: Optional[str] = None, 
                 config: Optional[ProgressConfig] = None, unit: str = 'it'):
        super().__init__(total, desc, task_id, unit)
        self.config = config or ProgressConfig()
        
        if self.config.enabled and TQDM_AVAILABLE:
            self._pbar = tqdm.tqdm(
                total=total,
                desc=desc,
                unit=unit,
                disable=False,
                leave=True,
                file=sys.stdout
            )
        else:
            self._pbar = None
            
        # Log start if milestones enabled
        if self.config.log_milestones:
            logging.info(f"Starting {desc}" + (f" ({total:,} items)" if total else ""))
    
    def update(self, n: int = 1) -> None:
        self.current += n
        if self._pbar is not None:
            self._pbar.update(n)
        self._log_milestone(self.config)
    
    def set_description(self, desc: str) -> None:
        self.desc = desc
        if self._pbar is not None:
            self._pbar.set_description(desc)
    
    def close(self) -> None:
        if self.completed:
            return
        self.completed = True
        
        if self._pbar is not None:
            self._pbar.close()
        
        # Log completion
        if self.config.log_milestones:
            elapsed = time.time() - self._start_time
            logging.info(
                f"Completed {self.desc}" +
                (f" ({self.current:,}" + (f"/{self.total:,}" if self.total else "") + " items)" if self.current > 0 else "") +
                f" - elapsed: {elapsed:.1f}s"
            )


class RichProgressBar(BaseProgressBar):
    """Progress bar implementation using rich."""
    
    def __init__(self, total: Optional[int], desc: str, task_id: Optional[str] = None,
                 progress_instance: Optional['Progress'] = None, rich_task_id: Optional[TaskID] = None,
                 config: Optional[ProgressConfig] = None, unit: str = 'it'):
        super().__init__(total, desc, task_id, unit)
        self.config = config or ProgressConfig()
        self.progress_instance = progress_instance
        self.rich_task_id = rich_task_id
        
        # Log start if milestones enabled
        if self.config.log_milestones:
            logging.info(f"Starting {desc}" + (f" ({total:,} items)" if total else ""))
    
    def update(self, n: int = 1) -> None:
        self.current += n
        if self.progress_instance and self.rich_task_id:
            self.progress_instance.update(self.rich_task_id, advance=n)
        self._log_milestone(self.config)
    
    def set_description(self, desc: str) -> None:
        self.desc = desc
        if self.progress_instance and self.rich_task_id:
            self.progress_instance.update(self.rich_task_id, description=desc)
    
    def close(self) -> None:
        if self.completed:
            return
        self.completed = True
        
        # Rich progress bars are managed by the progress instance
        # Log completion
        if self.config.log_milestones:
            elapsed = time.time() - self._start_time
            logging.info(
                f"Completed {self.desc}" +
                (f" ({self.current:,}" + (f"/{self.total:,}" if self.total else "") + " items)" if self.current > 0 else "") +
                f" - elapsed: {elapsed:.1f}s"
            )


class NullProgressBar(BaseProgressBar):
    """No-op progress bar for disabled progress reporting."""
    
    def __init__(self, total: Optional[int], desc: str, task_id: Optional[str] = None,
                 config: Optional[ProgressConfig] = None, unit: str = 'it'):
        super().__init__(total, desc, task_id, unit)
        self.config = config or ProgressConfig()
        
        # Log start if milestones enabled (even in null mode)
        if self.config.log_milestones:
            logging.info(f"Starting {desc}" + (f" ({total:,} items)" if total else ""))
    
    def update(self, n: int = 1) -> None:
        self.current += n
        self._log_milestone(self.config)
    
    def set_description(self, desc: str) -> None:
        self.desc = desc
    
    def close(self) -> None:
        if self.completed:
            return
        self.completed = True
        
        # Log completion even in null mode
        if self.config.log_milestones:
            elapsed = time.time() - self._start_time
            logging.info(
                f"Completed {self.desc}" +
                (f" ({self.current:,}" + (f"/{self.total:,}" if self.total else "") + " items)" if self.current > 0 else "") +
                f" - elapsed: {elapsed:.1f}s"
            )


class BaseProgressManager(ABC):
    """Abstract base class for progress managers."""
    
    def __init__(self, config: ProgressConfig):
        self.config = config
        self.bars: Dict[str, BaseProgressBar] = {}
        self._lock = threading.Lock()
        
        # Register for cleanup
        _progress_instances.append(self)
        _register_cleanup()
    
    @abstractmethod
    def create_bar(self, total: Optional[int], desc: str, parent_id: Optional[str] = None, unit: str = 'it') -> BaseProgressBar:
        """Create a new progress bar."""
        pass
    
    def get_bar(self, task_id: str) -> Optional[BaseProgressBar]:
        """Get a progress bar by task ID."""
        with self._lock:
            return self.bars.get(task_id)
    
    def remove_bar(self, task_id: str) -> None:
        """Remove a progress bar by task ID."""
        with self._lock:
            if task_id in self.bars:
                self.bars[task_id].close()
                del self.bars[task_id]
    
    def cleanup(self) -> None:
        """Clean up all progress bars."""
        with self._lock:
            for bar in self.bars.values():
                try:
                    bar.close()
                except Exception:
                    pass
            self.bars.clear()
        
        # Remove from global registry
        if self in _progress_instances:
            _progress_instances.remove(self)


class TqdmProgressManager(BaseProgressManager):
    """Progress manager using individual tqdm progress bars."""
    
    def create_bar(self, total: Optional[int], desc: str, parent_id: Optional[str] = None, unit: str = 'it') -> BaseProgressBar:
        bar = TqdmProgressBar(total, desc, config=self.config, unit=unit)
        with self._lock:
            self.bars[bar.task_id] = bar
        return bar


class RichProgressManager(BaseProgressManager):
    """Progress manager using rich's consolidated progress display."""
    
    def __init__(self, config: ProgressConfig):
        super().__init__(config)
        
        if config.enabled and RICH_AVAILABLE:
            self.console = Console(file=sys.stdout)
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.start()
        else:
            self.console = None
            self.progress = None
    
    def create_bar(self, total: Optional[int], desc: str, parent_id: Optional[str] = None, unit: str = 'it') -> BaseProgressBar:
        if self.progress and self.config.enabled:
            rich_task_id = self.progress.add_task(desc, total=total)
            bar = RichProgressBar(total, desc, progress_instance=self.progress, 
                                rich_task_id=rich_task_id, config=self.config, unit=unit)
        else:
            bar = NullProgressBar(total, desc, config=self.config, unit=unit)
        
        with self._lock:
            self.bars[bar.task_id] = bar
        return bar
    
    def cleanup(self) -> None:
        super().cleanup()
        if self.progress:
            self.progress.stop()


class NullProgressManager(BaseProgressManager):
    """No-op progress manager for disabled progress reporting."""
    
    def create_bar(self, total: Optional[int], desc: str, parent_id: Optional[str] = None, unit: str = 'it') -> BaseProgressBar:
        bar = NullProgressBar(total, desc, config=self.config, unit=unit)
        with self._lock:
            self.bars[bar.task_id] = bar
        return bar


# Global progress manager instance (thread-local)
def _get_thread_local_manager() -> Optional[BaseProgressManager]:
    """Get the thread-local progress manager."""
    return getattr(_local, 'progress_manager', None)


def _set_thread_local_manager(manager: Optional[BaseProgressManager]) -> None:
    """Set the thread-local progress manager."""
    _local.progress_manager = manager


def configure_progress(
    show_progress: bool = None,
    quiet: bool = False,
    force_terminal: bool = False,
    prefer_rich: bool = True,
    log_milestones: bool = True,
    milestone_intervals: List[float] = None,
) -> ProgressConfig:
    """
    Configure global progress settings.
    
    Args:
        show_progress: Whether to show progress bars. None for auto-detect.
        quiet: Suppress all progress output.
        force_terminal: Force terminal output even if not TTY.
        prefer_rich: Prefer rich output over tqdm when available.
        log_milestones: Log milestone messages in non-interactive mode.
        milestone_intervals: Progress percentages to log.
    
    Returns:
        ProgressConfig instance.
    """
    return ProgressConfig(
        enabled=show_progress,
        force_terminal=force_terminal,
        prefer_rich=prefer_rich,
        quiet_mode=quiet,
        log_milestones=log_milestones,
        milestone_intervals=milestone_intervals,
    )


def get_progress_manager(
    show_progress: bool = None,
    quiet: bool = False,
    force_terminal: bool = False,
    prefer_rich: bool = True,
    **kwargs
) -> BaseProgressManager:
    """
    Get or create a progress manager for the current thread.
    
    Args:
        show_progress: Whether to show progress bars. None for auto-detect.
        quiet: Suppress all progress output.
        force_terminal: Force terminal output even if not TTY.
        prefer_rich: Prefer rich output over tqdm when available.
        **kwargs: Additional configuration passed to configure_progress.
    
    Returns:
        BaseProgressManager instance.
    """
    # Check if we already have a manager for this thread
    existing = _get_thread_local_manager()
    if existing is not None:
        return existing
    
    config = configure_progress(
        show_progress=show_progress,
        quiet=quiet,
        force_terminal=force_terminal,
        prefer_rich=prefer_rich,
        **kwargs
    )
    
    # Select manager implementation
    if config.enabled:
        if config.prefer_rich and RICH_AVAILABLE:
            manager = RichProgressManager(config)
        elif TQDM_AVAILABLE:
            manager = TqdmProgressManager(config)
        else:
            manager = NullProgressManager(config)
    else:
        manager = NullProgressManager(config)
    
    _set_thread_local_manager(manager)
    return manager


@contextmanager
def create_progress_bar(
    total: Optional[int],
    desc: str = "Processing",
    show_progress: bool = None,
    quiet: bool = False,
    **kwargs
) -> BaseProgressBar:
    """
    Context manager for creating a simple progress bar.
    
    Args:
        total: Total number of items to process.
        desc: Description for the progress bar.
        show_progress: Whether to show progress. None for auto-detect.
        quiet: Suppress progress output.
        **kwargs: Additional configuration options.
    
    Yields:
        BaseProgressBar instance.
    
    Example:
        >>> with create_progress_bar(100, "Processing files") as pbar:
        ...     for i in range(100):
        ...         # Do work
        ...         pbar.update(1)
    """
    manager = get_progress_manager(show_progress=show_progress, quiet=quiet, **kwargs)
    bar = manager.create_bar(total=total, desc=desc)
    try:
        yield bar
    finally:
        bar.close()
        manager.remove_bar(bar.task_id)


def update_cli_args(parser, add_quiet: bool = True):
    """
    Add standard progress-related arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance to modify.
        add_quiet: Whether to add the --quiet flag.
    
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> update_cli_args(parser)
        >>> args = parser.parse_args()
        >>> show_progress = resolve_progress_settings(args)
    """
    progress_group = parser.add_argument_group('Progress Options')
    
    progress_group.add_argument(
        '--progress',
        action='store_true',
        help='Force enable progress bars even in non-interactive mode'
    )
    
    progress_group.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )
    
    if add_quiet:
        progress_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress progress output and reduce logging'
        )


def resolve_progress_settings(args) -> Dict[str, Any]:
    """
    Resolve progress settings from parsed command-line arguments.
    
    Args:
        args: Parsed arguments from ArgumentParser.
    
    Returns:
        Dictionary of progress configuration settings.
    
    Example:
        >>> args = parser.parse_args(['--no-progress'])
        >>> settings = resolve_progress_settings(args)
        >>> manager = get_progress_manager(**settings)
    """
    show_progress = None
    quiet = getattr(args, 'quiet', False)
    
    if hasattr(args, 'no_progress') and args.no_progress:
        show_progress = False
    elif hasattr(args, 'progress') and args.progress:
        show_progress = True
    
    # --quiet implies --no-progress
    if quiet:
        show_progress = False
    
    return {
        'show_progress': show_progress,
        'quiet': quiet,
    }


# Convenience functions for backward compatibility
def create_progress_context(total: Optional[int], desc: str = "Processing", **kwargs):
    """Legacy alias for create_progress_bar."""
    return create_progress_bar(total=total, desc=desc, **kwargs)


def get_progress_context(**kwargs):
    """Legacy alias for get_progress_manager."""
    return get_progress_manager(**kwargs)


# Module-level cleanup on import
_register_cleanup()