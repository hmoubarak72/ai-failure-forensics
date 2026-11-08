<#
.SYNOPSIS
  Package AI Failure Forensics with Docker on Windows (Docker Desktop), step by step.

.DESCRIPTION
  1. Makes sure Docker Desktop is running
  2. Builds the image from the Dockerfile (official python:3.11-slim base)
  3. Starts the API + UI containers with docker compose and waits for the healthchecks
  4. Shows the seeded demo results and opens the trace explorer
  5. -Export : also saves the image to a .tar you can copy to another machine (docker load -i ...)
  6. -Publish: commits and pushes the Docker publishing files, so GitHub Actions publishes
               ghcr.io/hmoubarak72/ai-failure-forensics

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\package.ps1
  powershell -ExecutionPolicy Bypass -File scripts\package.ps1 -Export -Publish
#>
param(
    [switch]$Export,
    [switch]$Publish,
    [string]$Image = "ai-failure-forensics:latest"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)   # repo root

function Step($n, $title, $why) {
    Write-Host ""
    Write-Host "=== Step $n : $title ===" -ForegroundColor Cyan
    Write-Host "    $why" -ForegroundColor DarkGray
}
function Run($cmd) {
    Write-Host "  > $cmd" -ForegroundColor Yellow
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $cmd" }
}

# ---------------------------------------------------------------------------
Step 1 "Check Docker Desktop" "The docker CLI talks to the Docker engine that Docker Desktop runs in the background."
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $dd = "$Env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $dd)) { throw "Docker Desktop not found. Install it from https://www.docker.com/products/docker-desktop/" }
    Write-Host "  Docker engine not running - starting Docker Desktop..."
    Start-Process $dd
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 3
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { break }
    }
    if ($LASTEXITCODE -ne 0) { throw "Docker engine did not start within 3 minutes." }
}
Run "docker version --format 'Client {{.Client.Version}} / Engine {{.Server.Version}}'"

# ---------------------------------------------------------------------------
Step 2 "Build the image" "Reads the Dockerfile: stage 1 builds Python wheels, stage 2 copies only what is needed to run (smaller, non-root image)."
Run "docker build -t $Image ."
Run "docker image ls ai-failure-forensics"

# ---------------------------------------------------------------------------
Step 3 "Start the containers" "docker compose starts two containers from the same image: 'api' (port 8000) and 'ui' (port 8501). Data lives in a named volume, so it survives restarts."
Run "docker compose up -d"
Write-Host "  Waiting for the API healthcheck (first start also generates the 150-trace demo)..."
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-RestMethod http://localhost:8000/health -TimeoutSec 3 | Out-Null; $ok = $true; break } catch { Start-Sleep -Seconds 3 }
}
if (-not $ok) { docker compose logs api --tail 40; throw "API did not become healthy." }
Run "docker compose ps"

# ---------------------------------------------------------------------------
Step 4 "Check the results" "Queries the running API from your machine, through the port that compose published."
$a = Invoke-RestMethod http://localhost:8000/analytics
Write-Host ("  Traces: {0}   Flagged: {1}   Eval cases: {2}" -f $a.totals.traces, $a.totals.flagged, $a.totals.eval_cases) -ForegroundColor Green
$last = $a.regression_trend[-1]
Write-Host ("  Latest regression: {0} resolves {1}/{2} known issues" -f $last.pipeline_version, $last.resolved, $last.total) -ForegroundColor Green
Start-Process "http://localhost:8501"
Write-Host "  Trace explorer opened at http://localhost:8501  (API docs: http://localhost:8000/docs)"

# ---------------------------------------------------------------------------
if ($Export) {
    Step 5 "Export the image" "docker save writes the image to one file. On another machine: docker load -i ai-failure-forensics-image.tar"
    Run "docker save -o ai-failure-forensics-image.tar $Image"
    Get-Item ai-failure-forensics-image.tar | Select-Object Name, @{n = "SizeMB"; e = { [math]::Round($_.Length / 1MB) } } | Format-Table
}

if ($Publish) {
    Step 6 "Publish via GitHub Actions" "Pushing the docker-publish workflow makes GitHub build the image and push it to ghcr.io on every push to main."
    Run "git add .github/workflows/docker-publish.yml docker-compose.ghcr.yml README.md scripts/package.ps1 .gitignore .dockerignore"
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { Run "git commit -m 'Publish Docker image to GHCR; add package.ps1 and docker-compose.ghcr.yml'" }
    Run "git push"
    Write-Host "  Watch it build: https://github.com/hmoubarak72/ai-failure-forensics/actions" -ForegroundColor Green
    Write-Host "  When it finishes: docker pull ghcr.io/hmoubarak72/ai-failure-forensics:latest" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. Useful commands:" -ForegroundColor Cyan
Write-Host "  docker compose logs -f api     # follow API logs"
Write-Host "  docker compose down            # stop (data kept in the volume)"
Write-Host "  docker compose down -v         # stop and delete the data volume"
