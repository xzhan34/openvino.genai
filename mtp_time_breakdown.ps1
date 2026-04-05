# mtp_time_breakdown.ps1 - Per-step MTP inference time breakdown
# Usage: powershell -ExecutionPolicy Bypass -File mtp_time_breakdown.ps1 [-MtpK 2] [-OutputTokens 64] [-Mode vl]

param(
    [int]$MtpK = 2,
    [int]$OutputTokens = 64,
    [string]$Mode = "vl",           # "text" or "vl"
    [string]$Prompt = "",           # custom prompt (auto-selected if empty)
    [string]$Model = "C:\work\models\Qwen3.5-9B"
)

# --- Environment Setup ---
$GENAI_DIR = "C:\work\openvino_ws\openvino.genai.liangali"
$OV_DIR    = "C:\work\openvino_ws\openvino.liangali"
$env:PATH = "$OV_DIR\bin\intel64\RelWithDebInfo;$OV_DIR\temp\Windows_AMD64\tbb\bin;$GENAI_DIR\build-master\openvino_genai;" + $env:PATH
$env:OPENVINO_TOKENIZERS_PATH_GENAI = "$GENAI_DIR\build-master\openvino_genai\openvino_tokenizers.dll"
$env:OV_GENAI_USE_MODELING_API = "1"
$env:OV_GENAI_INFLIGHT_QUANT_MODE = "int4_asym"
$env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE = "128"
$env:OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE = "int4_asym"
$env:OV_GENAI_MTP_SNAPSHOT = "1"
$env:OV_GENAI_SNAPSHOT_RESTORE = "3"
# oneDNN FC accumulation mode fix (fully_connected_onednn.cpp) ensures batch-size-invariant
# INT4 GEMM results.  No need to disable oneDNN anymore.
$env:OV_GENAI_STEP_PROFILE = "1"

$EXE       = "$GENAI_DIR\build-master\src\cpp\src\modeling\samples\RelWithDebInfo\modeling_qwen3_5.exe"
$IMAGE_PATH = "$OV_DIR\docs\articles_en\assets\images\get_started_with_cpp.jpg"
$OUT_DIR   = "C:\work\openvino_ws"

# --- Build args ---
$argList = @(
    "--model", $Model,
    "--device", "GPU",
    "--mode", $Mode,
    "--output-tokens", "$OutputTokens",
    "--think", "0",
    "--temperature", "0",
    "--mtp", "1",
    "--pure-batch", "1",
    "--mtp-k", "$MtpK"
)

if ($Mode -eq "vl") {
    $argList += @("--image", $IMAGE_PATH)
    if (-not $Prompt) { $Prompt = "describe this picture in details" }
} else {
    if (-not $Prompt) { $Prompt = "Hello, please write a short story about a robot learning to paint." }
}
$argList += @("--prompt", "`"$Prompt`"")

$outFile = Join-Path $OUT_DIR "step_profile_out.txt"
$errFile = Join-Path $OUT_DIR "step_profile_err.txt"

# --- Run ---
Write-Host "MTP Time Breakdown: K=$MtpK, mode=$Mode, tokens=$OutputTokens" -ForegroundColor Cyan
Write-Host "Command: $EXE $($argList -join ' ')" -ForegroundColor DarkGray
Write-Host ""

$proc = Start-Process -FilePath $EXE -ArgumentList $argList -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $outFile -RedirectStandardError $errFile
$exitCode = $proc.ExitCode

if ($exitCode -ne 0) {
    Write-Host "FAILED (exit code $exitCode)" -ForegroundColor Red
    $err = Get-Content $errFile -Raw -ErrorAction SilentlyContinue
    if ($err) { Write-Host ($err.Substring(0, [Math]::Min(500, $err.Length))) -ForegroundColor Red }
    exit 1
}

# --- Print summary from stdout ---
$output = Get-Content $outFile -Raw
foreach ($line in $output -split "`r?`n") {
    $l = $line.Trim()
    if ($l -match '^(TTFT|TPOT|Throughput|Decode time|MTP |--- |Main verify|Main GPU|MTP draft|KV trim|Dead KV|Snapshot save|Restore|State refresh|State restore|Accept|Avg step|MTP reset|\-{5})') {
        Write-Host $line
    }
}

# --- Print per-step breakdown from stderr ---
Write-Host ""
Write-Host "=== Per-Step Breakdown ===" -ForegroundColor Yellow
$stepLines = Select-String -Path $errFile -Pattern "^\[STEP" | ForEach-Object { $_.Line }
if ($stepLines.Count -gt 0) {
    foreach ($sl in $stepLines) { Write-Host $sl }
} else {
    Write-Host "(no per-step lines found in stderr)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Full stdout: $outFile" -ForegroundColor DarkGray
Write-Host "Full stderr: $errFile" -ForegroundColor DarkGray
