# bench_mtp.ps1 - Benchmark MTP performance comparison for Qwen3.5
# Usage: powershell -ExecutionPolicy Bypass -File bench_mtp.ps1

param(
    [int]$NumRuns = 3,
    [int]$OutputTokens = 64,
    [string]$Mode = "all"  # "text", "vl", or "all"
)

# --- Environment Setup ---
$GENAI_DIR = "C:\work\openvino_ws\openvino.genai.liangali"
$OV_DIR    = "C:\work\openvino_ws\openvino.liangali"
$env:OPENVINO_TOKENIZERS_PATH_GENAI = "$GENAI_DIR\build-master\openvino_genai\openvino_tokenizers.dll"
$env:PATH = "$OV_DIR\bin\intel64\RelWithDebInfo;$OV_DIR\temp\Windows_AMD64\tbb\bin;$GENAI_DIR\build-master\openvino_genai;$env:PATH"
$env:OV_GENAI_USE_MODELING_API = "1"
$env:OV_GENAI_INFLIGHT_QUANT_MODE = "int4_asym"
$env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE = "128"
$env:OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE = "int4_asym"
$env:DEVICE = "GPU"

$EXE       = "$GENAI_DIR\build-master\src\cpp\src\modeling\samples\RelWithDebInfo\modeling_qwen3_5.exe"
$MODEL_DIR = "C:\work\openvino_ws\openvino.genai.xzhan34\tests\module_genai\cpp\test_models\Qwen3.5-0.8B"
$IMAGE_PATH = "$OV_DIR\docs\articles_en\assets\images\get_started_with_cpp.jpg"

# --- Helper: parse metrics from stdout ---
function Parse-Metrics([string]$output) {
    $metrics = @{
        TTFT       = "N/A"
        DecodeTime = "N/A"
        TPOT       = "N/A"
        Throughput = "N/A"
        MTPHits    = "N/A"
        MTPRate    = "N/A"
        OutputSize = "N/A"
        MTPInfers  = "N/A"
        TokensPerInfer = "N/A"
    }
    foreach ($line in $output -split "`r?`n") {
        $line = $line.Trim()
        if ($line -match '^TTFT:\s+([\d.]+)\s+ms') {
            $metrics.TTFT = [double]$Matches[1]
        }
        elseif ($line -match '^Decode time:\s+([\d.]+)\s+ms') {
            $metrics.DecodeTime = [double]$Matches[1]
        }
        elseif ($line -match '^TPOT:\s+([\d.]+)\s+ms/token') {
            $metrics.TPOT = [double]$Matches[1]
        }
        elseif ($line -match '^Throughput:\s+([\d.]+)\s+tokens/s') {
            $metrics.Throughput = [double]$Matches[1]
        }
        elseif ($line -match '^Output token size:\s+(\d+)') {
            $metrics.OutputSize = [int]$Matches[1]
        }
        elseif ($line -match '^MTP hits:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)') {
            $metrics.MTPHits = "$($Matches[1])/$($Matches[2])"
            $metrics.MTPRate = [double]$Matches[3]
        }
        elseif ($line -match '^MTP main model infers:\s+(\d+)') {
            $metrics.MTPInfers = [int]$Matches[1]
        }
        elseif ($line -match '^MTP tokens/infer:\s+([\d.]+)') {
            $metrics.TokensPerInfer = [double]$Matches[1]
        }
    }
    return $metrics
}

# --- Helper: run one benchmark ---
function Run-SingleBenchmark([string]$label, [string[]]$exeArgs) {
    Write-Host "  Running: $label ..." -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # Run exe and capture both stdout and stderr via temp files
    $tempOut = [System.IO.Path]::GetTempFileName()
    $tempErr = [System.IO.Path]::GetTempFileName()
    $proc = Start-Process -FilePath $EXE -ArgumentList $exeArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr
    $sw.Stop()
    $exitCode = $proc.ExitCode
    $output = Get-Content $tempOut -Raw -ErrorAction SilentlyContinue
    $errOutput = Get-Content $tempErr -Raw -ErrorAction SilentlyContinue
    Remove-Item $tempOut, $tempErr -Force -ErrorAction SilentlyContinue

    if ($exitCode -ne 0) {
        Write-Host "  FAILED (exit code $exitCode)" -ForegroundColor Red
        if ($errOutput) { Write-Host "  stderr: $($errOutput.Substring(0, [Math]::Min(200, $errOutput.Length)))" -ForegroundColor Red }
        return $null
    }
    $metrics = Parse-Metrics $output
    $metrics["WallTime"] = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    Write-Host ("  Done ({0}s) TTFT={1}ms TPOT={2}ms/tok TP={3}tok/s" -f $metrics.WallTime, $metrics.TTFT, $metrics.TPOT, $metrics.Throughput) -ForegroundColor Green
    return $metrics
}

# --- Helper: compute averages ---
function Get-Averages([System.Collections.ArrayList]$resultList) {
    $valid = [System.Collections.ArrayList]@()
    foreach ($item in $resultList) {
        if ($null -ne $item) { [void]$valid.Add($item) }
    }
    if ($valid.Count -eq 0) { return $null }

    $avg = @{
        TTFT = "N/A"; TPOT = "N/A"; Throughput = "N/A"
        DecodeTime = "N/A"; MTPRate = "N/A"; MTPHits = "N/A"
        TokensPerInfer = "N/A"
        Count = $valid.Count
    }
    foreach ($key in @("TTFT", "TPOT", "Throughput", "DecodeTime", "MTPRate", "TokensPerInfer")) {
        $vals = [System.Collections.ArrayList]@()
        foreach ($item in $valid) {
            $v = $item[$key]
            if ($v -ne "N/A" -and $v -is [double]) { [void]$vals.Add($v) }
        }
        if ($vals.Count -gt 0) {
            $sum = 0.0
            foreach ($v in $vals) { $sum += $v }
            $avg[$key] = [math]::Round($sum / $vals.Count, 2)
        }
    }
    $lastMtp = $valid[$valid.Count - 1].MTPHits
    if ($lastMtp -and $lastMtp -ne "N/A") { $avg.MTPHits = $lastMtp }
    return $avg
}

# --- Configs ---
$configs = @(
    @{ Name = "No MTP (baseline)"; MTP = 0 }
    @{ Name = "MTP enabled";       MTP = 1 }
)

$modeList = [System.Collections.ArrayList]@()
if ($Mode -eq "all" -or $Mode -eq "text") { [void]$modeList.Add("text") }
if ($Mode -eq "all" -or $Mode -eq "vl")   { [void]$modeList.Add("vl") }

# --- Run benchmarks ---
$allResults = @{}

foreach ($m in $modeList) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host " Mode: $m  |  Output tokens: $OutputTokens  |  Runs: $NumRuns" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow

    foreach ($cfg in $configs) {
        $key = "$m|$($cfg.Name)"
        $runResults = [System.Collections.ArrayList]::new()

        # Build argument list
        $argList = [System.Collections.ArrayList]@(
            "--model", $MODEL_DIR,
            "--mode", $m,
            "--output-tokens", "$OutputTokens",
            "--think", "0"
        )
        if ($m -eq "text") {
            [void]$argList.AddRange(@("--prompt", "`"Hello, please write a short story about a robot learning to paint.`""))
        } else {
            [void]$argList.AddRange(@("--image", $IMAGE_PATH, "--prompt", "`"describe this picture in details: `""))
        }
        if ($cfg.MTP -gt 0) {
            [void]$argList.AddRange(@("--mtp", "1"))
        }

        Write-Host "`n--- $($cfg.Name) ($m mode) ---" -ForegroundColor White
        for ($i = 1; $i -le $NumRuns; $i++) {
            $label = "$($cfg.Name) run $i/$NumRuns"
            $result = Run-SingleBenchmark $label ([string[]]$argList)
            [void]$runResults.Add($result)
        }
        $allResults[$key] = $runResults
    }
}

# --- Print Summary Table ---
Write-Host "`n`n" -NoNewline
Write-Host "=================================================================" -ForegroundColor Yellow
Write-Host " PERFORMANCE COMPARISON SUMMARY" -ForegroundColor Yellow
Write-Host " Model: Qwen3.5-0.8B | Quant: INT4_ASYM g128 | Device: GPU" -ForegroundColor Yellow
Write-Host " Output tokens: $OutputTokens | Runs per config: $NumRuns" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow

foreach ($m in $modeList) {
    Write-Host "`n--- Mode: $($m.ToUpper()) ---" -ForegroundColor Cyan

    $header = "{0,-22} {1,10} {2,14} {3,16} {4,14} {5,12} {6,10}" -f "Config", "TTFT(ms)", "TPOT(ms/tok)", "Throughput(t/s)", "Decode(ms)", "MTP Rate", "Tok/Infer"
    $sep = "-" * 104
    Write-Host $sep
    Write-Host $header
    Write-Host $sep

    $baselineAvg = $null
    foreach ($cfg in $configs) {
        $key = "$m|$($cfg.Name)"
        $avg = Get-Averages $allResults[$key]

        if ($null -eq $avg) {
            $row = "{0,-22} {1,10} {2,14} {3,16} {4,14} {5,12}" -f $cfg.Name, "FAILED", "FAILED", "FAILED", "FAILED", "N/A"
            Write-Host $row -ForegroundColor Red
            continue
        }

        if ($cfg.MTP -eq 0) { $baselineAvg = $avg }

        $mtpStr = if ($avg.MTPRate -ne "N/A") { "$($avg.MTPRate)%" } else { "N/A" }
        $tpiStr = if ($avg.TokensPerInfer -ne "N/A") { "$($avg.TokensPerInfer)" } else { "N/A" }
        $row = "{0,-22} {1,10} {2,14} {3,16} {4,14} {5,12} {6,10}" -f $cfg.Name, $avg.TTFT, $avg.TPOT, $avg.Throughput, $avg.DecodeTime, $mtpStr, $tpiStr
        Write-Host $row
    }
    Write-Host $sep

    # Speedup/overhead summary
    if ($null -ne $baselineAvg) {
        foreach ($cfg in $configs) {
            if ($cfg.MTP -eq 0) { continue }
            $key = "$m|$($cfg.Name)"
            $avg = Get-Averages $allResults[$key]
            if ($null -eq $avg -or $avg.TPOT -eq "N/A" -or $baselineAvg.TPOT -eq "N/A") { continue }
            if ($baselineAvg.TPOT -gt 0 -and $avg.TPOT -gt 0) {
                $tpotPct = [math]::Round(($avg.TPOT - $baselineAvg.TPOT) / $baselineAvg.TPOT * 100, 1)
                $tpPct = "N/A"
                if ($avg.Throughput -ne "N/A" -and $baselineAvg.Throughput -ne "N/A" -and $baselineAvg.Throughput -gt 0) {
                    $tpPct = [math]::Round(($avg.Throughput - $baselineAvg.Throughput) / $baselineAvg.Throughput * 100, 1)
                }
                $sign = if ($tpotPct -ge 0) { "+" } else { "" }
                Write-Host ("  {0} vs baseline: TPOT {1}{2}%, Throughput {3}%" -f $cfg.Name, $sign, $tpotPct, $(if($tpPct -ne "N/A"){if($tpPct -ge 0){"+$tpPct"}else{"$tpPct"}}else{"N/A"})) -ForegroundColor Magenta
            }
        }
    }
}

# --- Per-Run Details ---
Write-Host "`n`n--- PER-RUN DETAILS ---" -ForegroundColor Yellow
foreach ($m in $modeList) {
    foreach ($cfg in $configs) {
        $key = "$m|$($cfg.Name)"
        $runResults = $allResults[$key]
        Write-Host ("`n  {0} ({1}):" -f $cfg.Name, $m) -ForegroundColor White
        for ($i = 0; $i -lt $runResults.Count; $i++) {
            $r = $runResults[$i]
            if ($null -eq $r) {
                Write-Host "    Run $($i+1): FAILED"
                continue
            }
            $mtpStr = if ($r.MTPHits -ne "N/A") { " MTP=$($r.MTPHits)($($r.MTPRate)%)" } else { "" }
            $tpiStr = if ($r.TokensPerInfer -ne "N/A") { " Tok/Infer=$($r.TokensPerInfer)" } else { "" }
            Write-Host ("    Run {0}: TTFT={1}ms TPOT={2}ms/tok TP={3}tok/s Decode={4}ms{5}{6}" -f ($i+1), $r.TTFT, $r.TPOT, $r.Throughput, $r.DecodeTime, $mtpStr, $tpiStr)
        }
    }
}

Write-Host "`n`nBenchmark complete." -ForegroundColor Green
