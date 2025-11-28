# PowerShell script to start both backend and frontend servers

Write-Host "🏥 Starting Medical Chatbot Application..." -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Navigate to project directory
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

Write-Host ""
Write-Host "📦 Installing backend dependencies..." -ForegroundColor Yellow

# Install backend dependencies
Set-Location "backend"
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  No .env file found. Creating from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✓ Created .env file. Please configure your API keys if needed." -ForegroundColor Green
    Write-Host ""
}

Write-Host "🚀 Starting servers..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Green
Write-Host "Frontend UI: http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers" -ForegroundColor Yellow
Write-Host ""

# Start backend server in background
$backendJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    Set-Location "backend"
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
} -ArgumentList $projectDir

Write-Host "✓ Backend server started (Job ID: $($backendJob.Id))" -ForegroundColor Green

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start frontend server
Set-Location $projectDir
Set-Location "frontend"

Write-Host "✓ Frontend server starting..." -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Medical Chatbot is now running!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    # Start frontend server (this will block)
    python -m http.server 3000
} finally {
    # Cleanup: Stop backend job when frontend is stopped
    Write-Host ""
    Write-Host "Stopping servers..." -ForegroundColor Yellow
    Stop-Job $backendJob
    Remove-Job $backendJob
    Write-Host "✓ Servers stopped" -ForegroundColor Green
}
