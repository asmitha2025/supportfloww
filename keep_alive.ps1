# keep_alive.ps1
# SupportMind - Crash-safe overnight DistilBERT training wrapper
# Prevents Windows sleep, auto-resumes from last checkpoint on crash.
#
# Usage:  powershell -ExecutionPolicy Bypass -File keep_alive.ps1
# Stop:   Ctrl+C  (training will stop at next checkpoint boundary)

$ProjectDir  = $PSScriptRoot
$PythonExe   = "python"
$TrainScript = Join-Path $ProjectDir "src\train_router.py"
$LogFile     = Join-Path $ProjectDir "keepalive_log.txt"
$ErrFile     = Join-Path $ProjectDir "train_err_v5.txt"
$ModelDir    = Join-Path $ProjectDir "models\ticket_classifier"

# -- Prevent sleep via SetThreadExecutionState --------------------------------
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class SleepGuard {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS        = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED   = 0x00000001;
    public const uint ES_DISPLAY_REQUIRED  = 0x00000002;
    public static void PreventSleep() {
        SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
    }
    public static void AllowSleep() {
        SetThreadExecutionState(ES_CONTINUOUS);
    }
}
"@
[SleepGuard]::PreventSleep()
Write-Host "[OK] Sleep prevention ACTIVE" -ForegroundColor Green

# -- Helper ------------------------------------------------------------------
function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $LogFile -Value $line
}

function Model-Saved {
    # DistilBERT model is fully saved when pytorch_model.bin (or model.safetensors) exists
    $bin = Join-Path $ModelDir "pytorch_model.bin"
    $safe = Join-Path $ModelDir "model.safetensors"
    return ((Test-Path $bin) -or (Test-Path $safe))
}

# -- Main retry loop ---------------------------------------------------------
$attempt     = 0
$maxAttempts = 20

Write-Log "======================================================="
Write-Log "SupportMind DistilBERT Training - Keep-Alive Started"
Write-Log "  Script : $TrainScript"
Write-Log "  Log    : $LogFile"
Write-Log "  Model  : $ModelDir"
Write-Log "======================================================="

while (-not (Model-Saved) -and $attempt -lt $maxAttempts) {
    $attempt++
    Write-Log "--- Attempt $attempt / $maxAttempts ---"

    # Re-arm sleep prevention each loop
    [SleepGuard]::PreventSleep()

    # Run training, tee stdout to log file, stderr to err file
    $proc = Start-Process -FilePath $PythonExe `
        -ArgumentList "-u `"$TrainScript`"" `
        -WorkingDirectory $ProjectDir `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput $LogFile `
        -RedirectStandardError  $ErrFile

    Write-Log "Training PID: $($proc.Id) - waiting for completion..."

    # Poll every 60 s
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 60
        [SleepGuard]::PreventSleep()

        if (Model-Saved) {
            Write-Log "[DONE] Model file detected - training finished!"
            break
        }

        $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if ($p) {
            $ramMB = [math]::Round($p.WorkingSet64 / 1MB, 0)
            Write-Log "  ... still running | RAM: ${ramMB} MB"
        }
    }

    if (Model-Saved) {
        Write-Log "[DONE] Model saved to $ModelDir - ALL DONE!"
        break
    }

    $exitCode = if ($proc.HasExited) { $proc.ExitCode } else { -1 }
    Write-Log "[WARN] Process exited with code $exitCode"

    if ($attempt -lt $maxAttempts) {
        Write-Log "   Waiting 30s before restart (will resume from last checkpoint)..."
        Start-Sleep -Seconds 30
    }
}

# -- Cleanup -----------------------------------------------------------------
[SleepGuard]::AllowSleep()

if (Model-Saved) {
    Write-Log "[SUCCESS] Training complete! Model is at: $ModelDir"
    Write-Host ""
    Write-Host ">>> TRAINING COMPLETE - DistilBERT model is ready!" -ForegroundColor Green
    Write-Host "    Next: python src\evaluate.py" -ForegroundColor Yellow
} else {
    Write-Log "[FAIL] Max attempts ($maxAttempts) reached without saving model."
    Write-Host "[FAIL] Training failed after $maxAttempts attempts." -ForegroundColor Red
    Write-Host "       Check: $ErrFile" -ForegroundColor Red
}
