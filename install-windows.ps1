# FLYNC Dependency Installation Script for Windows PowerShell
# Run this in PowerShell as Administrator

Write-Host "🚀 Installing FLYNC dependencies on Windows..." -ForegroundColor Green

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Install Chocolatey if not present
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "🍫 Installing Chocolatey..." -ForegroundColor Blue
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Install Git (needed for development)
Write-Host "🔧 Installing Git..." -ForegroundColor Blue
choco install git -y

# Install Python
Write-Host "🐍 Installing Python..." -ForegroundColor Blue
choco install python311 -y

# Install R
Write-Host "📊 Installing R..." -ForegroundColor Blue
choco install r.project -y

# Install Windows Subsystem for Linux (WSL2) for bioinformatics tools
Write-Host "🐧 Enabling WSL2..." -ForegroundColor Blue
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host "⚠️ WSL2 requires a system restart." -ForegroundColor Yellow
Write-Host "After restart, install Ubuntu from Microsoft Store and run:" -ForegroundColor Yellow
Write-Host "  wsl --set-default-version 2" -ForegroundColor Cyan
Write-Host "  wsl --install -d Ubuntu" -ForegroundColor Cyan

# Install Miniconda
Write-Host "🐍 Installing Miniconda..." -ForegroundColor Blue
choco install miniconda3 -y

# Refresh environment
refreshenv

Write-Host "✅ Basic installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your computer to complete WSL2 installation"
Write-Host "2. Install Ubuntu from Microsoft Store"
Write-Host "3. In Ubuntu WSL, run the Linux installation script:"
Write-Host "   curl -O https://raw.githubusercontent.com/yourrepo/flync/main/install-ubuntu.sh"
Write-Host "   bash install-ubuntu.sh"
Write-Host ""
Write-Host "💡 Alternative: Use Docker for FLYNC (recommended for Windows):"
Write-Host "   docker pull your-registry/flync:latest"
