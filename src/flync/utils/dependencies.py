"""System dependency management for FLYNC."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..utils import LoggerMixin


class DependencyManager(LoggerMixin):
    """Manages external system dependencies for FLYNC."""
    
    # Required tools for basic functionality
    REQUIRED_TOOLS = {
        "hisat2": {
            "command": "hisat2",
            "version_flag": "--version",
            "description": "Read alignment tool",
            "install_hint": "conda install hisat2"
        },
        "hisat2-build": {
            "command": "hisat2-build", 
            "version_flag": "--version",
            "description": "HISAT2 index builder",
            "install_hint": "conda install hisat2"
        },
        "stringtie": {
            "command": "stringtie",
            "version_flag": "--version", 
            "description": "Transcriptome assembler",
            "install_hint": "conda install stringtie"
        },
        "cpat.py": {
            "command": "cpat.py",
            "version_flag": "--version",
            "description": "Coding probability assessment",
            "install_hint": "pip install CPAT"
        },
        "gffread": {
            "command": "gffread",
            "version_flag": "--version",
            "description": "GTF/GFF file processor", 
            "install_hint": "conda install gffread"
        }
    }
    
    # Optional tools for enhanced functionality
    OPTIONAL_TOOLS = {
        "samtools": {
            "command": "samtools",
            "version_flag": "--version",
            "description": "SAM/BAM file processor",
            "install_hint": "conda install samtools"
        },
        "bedtools": {
            "command": "bedtools",
            "version_flag": "--version", 
            "description": "Genomic intervals toolkit",
            "install_hint": "conda install bedtools"
        },
        "fastq-dump": {
            "command": "fastq-dump",
            "version_flag": "--version",
            "description": "SRA data downloader",
            "install_hint": "conda install sra-tools"
        },
        "Rscript": {
            "command": "Rscript",
            "version_flag": "--version",
            "description": "R scripting (for ballgown)",
            "install_hint": "conda install r-base r-ballgown"
        }
    }
    
    def __init__(self):
        self.console = Console()
        
    def check_tool_availability(self, tool_name: str, tool_config: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """Check if a tool is available and get its version.
        
        Returns:
            (is_available, version_string)
        """
        try:
            # Check if command exists in PATH
            if not shutil.which(tool_config["command"]):
                return False, None
                
            # Try to get version information
            result = subprocess.run(
                [tool_config["command"], tool_config["version_flag"]],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Extract version from output (usually first line)
                version_line = result.stdout.strip().split('\n')[0] if result.stdout else "Unknown version"
                return True, version_line
            else:
                # Some tools might output version to stderr
                version_line = result.stderr.strip().split('\n')[0] if result.stderr else "Unknown version" 
                return True, version_line
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return False, None
    
    def check_all_dependencies(self) -> Dict[str, Dict[str, any]]:
        """Check all required and optional dependencies.
        
        Returns:
            Dictionary with tool status information
        """
        results = {
            "required": {},
            "optional": {},
            "summary": {"required_missing": 0, "optional_missing": 0}
        }
        
        # Check required tools
        for tool_name, tool_config in self.REQUIRED_TOOLS.items():
            is_available, version = self.check_tool_availability(tool_name, tool_config)
            results["required"][tool_name] = {
                "available": is_available,
                "version": version,
                "config": tool_config
            }
            if not is_available:
                results["summary"]["required_missing"] += 1
        
        # Check optional tools  
        for tool_name, tool_config in self.OPTIONAL_TOOLS.items():
            is_available, version = self.check_tool_availability(tool_name, tool_config)
            results["optional"][tool_name] = {
                "available": is_available,
                "version": version,
                "config": tool_config
            }
            if not is_available:
                results["summary"]["optional_missing"] += 1
                
        return results
    
    def display_dependency_status(self, results: Dict[str, Dict[str, any]]) -> None:
        """Display dependency status in a formatted table."""
        
        # Required tools table
        required_table = Table(title="Required System Dependencies", show_header=True)
        required_table.add_column("Tool", style="bold")
        required_table.add_column("Status", justify="center")
        required_table.add_column("Version", style="dim")
        required_table.add_column("Description")
        required_table.add_column("Install Command", style="cyan")
        
        for tool_name, info in results["required"].items():
            status = "✅ Found" if info["available"] else "❌ Missing"
            version = info["version"] or "N/A"
            description = info["config"]["description"]
            install_hint = info["config"]["install_hint"]
            
            required_table.add_row(tool_name, status, version, description, install_hint)
        
        # Optional tools table
        optional_table = Table(title="Optional System Dependencies", show_header=True)
        optional_table.add_column("Tool", style="bold")
        optional_table.add_column("Status", justify="center") 
        optional_table.add_column("Version", style="dim")
        optional_table.add_column("Description")
        optional_table.add_column("Install Command", style="cyan")
        
        for tool_name, info in results["optional"].items():
            status = "✅ Found" if info["available"] else "⚠️ Missing"
            version = info["version"] or "N/A"
            description = info["config"]["description"]
            install_hint = info["config"]["install_hint"]
            
            optional_table.add_row(tool_name, status, version, description, install_hint)
        
        # Display tables
        self.console.print(required_table)
        self.console.print()
        self.console.print(optional_table)
        
        # Summary panel
        summary = results["summary"]
        if summary["required_missing"] > 0:
            panel_style = "red"
            status_text = f"❌ {summary['required_missing']} required tools missing"
        else:
            panel_style = "green"
            status_text = "✅ All required tools available"
            
        if summary["optional_missing"] > 0:
            status_text += f"\n⚠️ {summary['optional_missing']} optional tools missing"
        else:
            status_text += "\n✅ All optional tools available"
            
        self.console.print(Panel(status_text, title="Dependency Summary", style=panel_style))
    
    def generate_install_commands(self, results: Dict[str, Dict[str, any]], package_manager: str = "conda") -> List[str]:
        """Generate installation commands for missing dependencies.
        
        Args:
            results: Results from check_all_dependencies()
            package_manager: 'conda', 'apt', or 'brew'
            
        Returns:
            List of installation commands
        """
        missing_tools = []
        
        # Collect missing required tools
        for tool_name, info in results["required"].items():
            if not info["available"]:
                missing_tools.append(info["config"]["install_hint"])
        
        # Collect missing optional tools
        for tool_name, info in results["optional"].items():
            if not info["available"]:
                missing_tools.append(info["config"]["install_hint"])
        
        if package_manager == "conda":
            # Group conda installs
            conda_packages = []
            pip_packages = []
            
            for hint in missing_tools:
                if hint.startswith("conda install"):
                    package = hint.replace("conda install ", "")
                    conda_packages.append(package)
                elif hint.startswith("pip install"):
                    pip_packages.append(hint)
            
            commands = []
            if conda_packages:
                commands.append(f"conda install {' '.join(conda_packages)}")
            commands.extend(pip_packages)
            
            return commands
        else:
            return list(set(missing_tools))  # Remove duplicates
    
    def is_ready_for_pipeline(self) -> bool:
        """Check if all required dependencies are available for running the pipeline."""
        results = self.check_all_dependencies()
        return results["summary"]["required_missing"] == 0
