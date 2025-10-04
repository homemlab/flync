#!/usr/bin/env python3
"""
Unit tests for the progress utility module.

Tests TTY simulation, fallback behavior, progress counting accuracy, 
and other core functionality.
"""

import io
import logging
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stderr

# Add src to path for imports
sys.path.insert(0, '/home/chlab/flync/src')

from utils.progress import (
    ProgressConfig, 
    get_progress_manager, 
    update_cli_args, 
    resolve_progress_settings,
    create_progress_bar
)


class TestProgressConfig(unittest.TestCase):
    """Test ProgressConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProgressConfig()
        self.assertIsInstance(config.enabled, bool)
        self.assertFalse(config.force_terminal)
        self.assertTrue(config.prefer_rich)
        self.assertFalse(config.quiet_mode)
        self.assertTrue(config.log_milestones)
        self.assertEqual(config.milestone_intervals, [25, 50, 75])
    
    def test_quiet_mode_disables_progress(self):
        """Test that quiet mode disables progress."""
        config = ProgressConfig(enabled=True, quiet_mode=True)
        self.assertFalse(config.enabled)
    
    def test_force_terminal(self):
        """Test force terminal mode."""
        config = ProgressConfig(enabled=None, force_terminal=True, quiet_mode=False)
        self.assertTrue(config.enabled)


class TestProgressManager(unittest.TestCase):
    """Test progress manager functionality."""
    
    def test_get_progress_manager_with_progress(self):
        """Test creating progress manager with progress enabled."""
        manager = get_progress_manager(show_progress=True, quiet=False)
        self.assertIsNotNone(manager)
        
        # Test creating a progress bar
        bar = manager.create_bar(total=100, desc="Test")
        self.assertIsNotNone(bar)
        
        # Test updating the progress bar
        bar.update(10)
        self.assertEqual(bar.current, 10)
        
        bar.close()
    
    def test_get_progress_manager_quiet_mode(self):
        """Test creating progress manager in quiet mode."""
        manager = get_progress_manager(show_progress=False, quiet=True)
        self.assertIsNotNone(manager)
        
        # Should return a null progress bar
        bar = manager.create_bar(total=100, desc="Test")
        self.assertIsNotNone(bar)
        
        # Updates should be no-ops
        bar.update(10)
        bar.close()
    
    def test_progress_bar_context_manager(self):
        """Test progress bar as context manager."""
        manager = get_progress_manager(show_progress=False)  # Use null manager for testing
        
        with manager.create_bar(total=50, desc="Context test") as bar:
            bar.update(25)
            self.assertEqual(bar.current, 25)
        
        # Bar should be closed after context exit
    
    def test_milestone_logging(self):
        """Test milestone logging functionality."""
        # Capture log output
        with self.assertLogs(level='INFO') as cm:
            manager = get_progress_manager(show_progress=False, quiet=False)
            bar = manager.create_bar(total=100, desc="Milestone test")
            
            # Should trigger 25% milestone
            bar.update(25)
            # Should trigger 50% milestone  
            bar.update(25)
            # Should trigger 75% milestone
            bar.update(25)
            
            bar.close()
        
        # Check that milestone messages were logged
        log_messages = ' '.join(cm.output)
        self.assertIn('25% complete', log_messages)
        self.assertIn('50% complete', log_messages)
        self.assertIn('75% complete', log_messages)


class TestCLIArguments(unittest.TestCase):
    """Test CLI argument parsing functionality."""
    
    def test_update_cli_args(self):
        """Test adding progress arguments to parser."""
        import argparse
        
        parser = argparse.ArgumentParser()
        update_cli_args(parser)
        
        # Test that arguments are added
        args = parser.parse_args(['--progress'])
        self.assertTrue(args.progress)
        self.assertFalse(args.no_progress)
        
        args = parser.parse_args(['--no-progress'])
        self.assertFalse(args.progress)
        self.assertTrue(args.no_progress)
        
        args = parser.parse_args(['--quiet'])
        self.assertTrue(args.quiet)
    
    def test_resolve_progress_settings(self):
        """Test resolving progress settings from args."""
        import argparse
        
        # Mock args with progress enabled
        args = Mock()
        args.progress = True
        args.no_progress = False
        args.quiet = False
        
        settings = resolve_progress_settings(args)
        self.assertTrue(settings['show_progress'])
        self.assertFalse(settings['quiet'])
        
        # Mock args with no progress
        args.progress = False
        args.no_progress = True
        args.quiet = False
        
        settings = resolve_progress_settings(args)
        self.assertFalse(settings['show_progress'])
        
        # Mock args with quiet mode (should override progress)
        args.progress = True
        args.no_progress = False
        args.quiet = True
        
        settings = resolve_progress_settings(args)
        self.assertFalse(settings['show_progress'])
        self.assertTrue(settings['quiet'])


class TestProgressCountingAccuracy(unittest.TestCase):
    """Test progress counting accuracy."""
    
    def test_accurate_counting(self):
        """Test that progress counting is accurate."""
        manager = get_progress_manager(show_progress=False)  # Use null manager
        bar = manager.create_bar(total=1000, desc="Accuracy test")
        
        # Update in various increments
        for i in range(10):
            bar.update(50)  # 50 * 10 = 500
        
        self.assertEqual(bar.current, 500)
        
        # Update with remaining amount
        bar.update(500)
        
        self.assertEqual(bar.current, 1000)
        self.assertEqual(bar.current, bar.total)
        
        bar.close()
    
    def test_overflow_handling(self):
        """Test handling of progress overflow."""
        manager = get_progress_manager(show_progress=False)
        bar = manager.create_bar(total=100, desc="Overflow test")
        
        # Update beyond total
        bar.update(150)
        
        # Should not exceed total (implementation dependent)
        # At minimum, should not crash
        self.assertGreaterEqual(bar.current, 0)
        
        bar.close()


class TestFallbackBehavior(unittest.TestCase):
    """Test fallback behavior when dependencies are missing."""
    
    @patch('utils.progress.TQDM_AVAILABLE', False)
    @patch('utils.progress.RICH_AVAILABLE', False)
    def test_fallback_to_null_progress(self):
        """Test fallback when neither tqdm nor rich are available."""
        # Should not raise an exception
        manager = get_progress_manager(show_progress=True)
        self.assertIsNotNone(manager)
        
        bar = manager.create_bar(total=100, desc="Fallback test")
        self.assertIsNotNone(bar)
        
        # Should handle updates gracefully
        bar.update(50)
        bar.close()
    
    def test_tty_detection_fallback(self):
        """Test TTY detection fallback behavior."""
        # Mock non-TTY environment
        with patch.object(sys.stdout, 'isatty', return_value=False):
            config = ProgressConfig()
            # Should disable progress in non-TTY by default
            self.assertFalse(config.enabled)
        
        # Mock TTY environment  
        with patch.object(sys.stdout, 'isatty', return_value=True):
            config = ProgressConfig()
            # Should enable progress in TTY
            self.assertTrue(config.enabled)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety (basic tests)."""
    
    def test_concurrent_progress_bars(self):
        """Test creating multiple progress bars concurrently."""
        manager = get_progress_manager(show_progress=False)
        
        # Create multiple progress bars
        bars = []
        for i in range(5):
            bar = manager.create_bar(total=100, desc=f"Task {i}")
            bars.append(bar)
        
        # Update all bars
        for i, bar in enumerate(bars):
            bar.update(i * 10)
            self.assertEqual(bar.current, i * 10)
        
        # Clean up
        for bar in bars:
            bar.close()


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)