# bench_mtp.ps1 - Benchmark MTP performance comparison for Qwen3.5
# Usage: powershell -ExecutionPolicy Bypass -File bench_mtp.ps1

param(
    [int]$NumRuns = 3,
    [int]$OutputTokens = 256,
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
$IMAGE_PATH = "$OV_DIR\docs\articles_en\assets\images\get_started_with_cpp.jpg"

# --- Models to benchmark ---
$MODELS = @(
    #@{ Name = "Qwen3.5-0.8B"; Path = "C:\work\openvino_ws\openvino.genai.xzhan34\tests\module_genai\cpp\test_models\Qwen3.5-0.8B" }
    @{ Name = "Qwen3.5-9B";   Path = "C:\work\models\Qwen3.5-9B" }
)

# --- Logging Setup ---
$LOG_DIR = Join-Path $GENAI_DIR "OV_Logs"
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }
$BENCH_TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# --- Helper: parse metrics from stdout ---
function Parse-Metrics([string]$output) {
    $metrics = @{
        TTFT       = "N/A"
        DecodeTime = "N/A"
        TPOT       = "N/A"
        Throughput = "N/A"
        MTPHits    = "N/A"
        MTPRate    = "N/A"
        MTPDraftAcceptRate = "N/A"
        MTPMeanAccepted = "N/A"
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
        elseif ($line -match '^MTP draft acceptance:\s+\d+/\d+\s+\(([\d.]+)%\)') {
            $metrics.MTPDraftAcceptRate = [double]$Matches[1]
        }
        elseif ($line -match '^MTP mean accepted/step:\s+([\d.]+)') {
            $metrics.MTPMeanAccepted = [double]$Matches[1]
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

# --- Helper: check generated text quality (detect degenerate repetition) ---
function Check-TextQuality([string]$output) {
    # Extract generated text: everything after the last metric line (Throughput/Spec decode/MTP lines)
    # The text is at the end of stdout, after all metric/profiling lines.
    $lines = $output -split "`r?`n"
    $textStartIdx = -1
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        $l = $lines[$i].Trim()
        if ($l -match '^(TTFT|TPOT|Throughput|Decode time|MTP |--- Spec|  Main verify|  MTP draft|  KV trim|Output token|Prompt token|Mode:|\[)' -or $l -eq '') {
            $textStartIdx = $i + 1
            break
        }
    }
    if ($textStartIdx -lt 0 -or $textStartIdx -ge $lines.Count) {
        return @{ OK = $false; Reason = "NO_TEXT"; Text = "" }
    }
    $genText = ($lines[$textStartIdx..($lines.Count - 1)] | Where-Object { $_.Trim() -ne '' }) -join ' '
    $genText = $genText.Trim()

    if ($genText.Length -lt 10) {
        return @{ OK = $false; Reason = "TOO_SHORT"; Text = $genText }
    }

    # Check 1: repeated word/phrase patterns (e.g., "about her about her about her")
    # Split into words and look for runs of the same word
    $words = $genText -split '\s+' | Where-Object { $_ -ne '' }
    $maxRepeatRun = 0
    $repeatWord = ""
    $currentRun = 1
    for ($w = 1; $w -lt $words.Count; $w++) {
        if ($words[$w] -eq $words[$w - 1]) {
            $currentRun++
            if ($currentRun -gt $maxRepeatRun) {
                $maxRepeatRun = $currentRun
                $repeatWord = $words[$w]
            }
        } else {
            $currentRun = 1
        }
    }

    # Check 2: repeated bigrams (e.g., "about her" x N)
    $maxBigramRepeat = 0
    $repeatBigram = ""
    if ($words.Count -ge 4) {
        $bigramRun = 1
        for ($w = 2; $w -lt $words.Count - 1; $w += 2) {
            $prev = "$($words[$w - 2]) $($words[$w - 1])"
            $curr = "$($words[$w]) $($words[$w + 1])"
            if ($curr -eq $prev) {
                $bigramRun++
                if ($bigramRun -gt $maxBigramRepeat) {
                    $maxBigramRepeat = $bigramRun
                    $repeatBigram = $curr
                }
            } else {
                $bigramRun = 1
            }
        }
    }

    # Check 3: unique word ratio in the last 50% of text
    $halfIdx = [math]::Floor($words.Count / 2)
    $lastHalf = $words[$halfIdx..($words.Count - 1)]
    $uniqueRatio = if ($lastHalf.Count -gt 0) {
        ($lastHalf | Sort-Object -Unique).Count / $lastHalf.Count
    } else { 1.0 }

    # Thresholds for degenerate detection
    $isDegenerate = $false
    $reason = "OK"

    if ($maxRepeatRun -ge 5) {
        $isDegenerate = $true
        $reason = "WORD_REPEAT: '$repeatWord' x$maxRepeatRun"
    } elseif ($maxBigramRepeat -ge 4) {
        $isDegenerate = $true
        $reason = "BIGRAM_REPEAT: '$repeatBigram' x$maxBigramRepeat"
    } elseif ($uniqueRatio -lt 0.20 -and $lastHalf.Count -ge 20) {
        $isDegenerate = $true
        $reason = "LOW_DIVERSITY: unique_ratio=$([math]::Round($uniqueRatio, 3)) in last $($lastHalf.Count) words"
    }

    $preview = if ($genText.Length -gt 80) { $genText.Substring(0, 80) + "..." } else { $genText }
    return @{ OK = (-not $isDegenerate); Reason = $reason; Text = $preview; UniqueRatio = [math]::Round($uniqueRatio, 3) }
}

# --- Helper: run one benchmark ---
function Run-SingleBenchmark([string]$label, [string[]]$exeArgs, [string]$logTag) {
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

    # Save log with timestamp
    $runTs = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeTag = $logTag -replace '[\s/|]+', '_'
    $logFile = Join-Path $LOG_DIR "${BENCH_TIMESTAMP}_${safeTag}_${runTs}.log"
    $cmdLine = "$EXE $($exeArgs -join ' ')"
    $logContent = @"
=== Benchmark Run Log ===
Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Label: $label
Command: $cmdLine
Exit Code: $exitCode
Wall Time: $([math]::Round($sw.Elapsed.TotalSeconds, 2))s

--- STDOUT ---
$output
--- STDERR ---
$errOutput
"@
    Set-Content -Path $logFile -Value $logContent -Encoding UTF8
    Write-Host "  Log saved: $logFile" -ForegroundColor DarkGray

    # Print generated text (everything after the last metric/profiling line)
    if ($output) {
        $outLines = $output -split "`r?`n"
        $txtStart = -1
        for ($ti = $outLines.Count - 1; $ti -ge 0; $ti--) {
            $tl = $outLines[$ti].Trim()
            if ($tl -match '^(TTFT|TPOT|Throughput|Decode time|MTP |--- Spec|  Main verify|  MTP draft|  KV trim|Output token|Prompt token|Mode:|\[)' -or $tl -eq '') {
                $txtStart = $ti + 1
                break
            }
        }
        if ($txtStart -ge 0 -and $txtStart -lt $outLines.Count) {
            $genLines = $outLines[$txtStart..($outLines.Count - 1)] | Where-Object { $_.Trim() -ne '' }
            if ($genLines.Count -gt 0) {
                $genPreview = ($genLines -join ' ').Trim()
                if ($genPreview.Length -gt 200) { $genPreview = $genPreview.Substring(0, 200) + "..." }
                Write-Host "  Generated: $genPreview" -ForegroundColor DarkCyan
            }
        }
    }

    if ($exitCode -ne 0) {
        Write-Host "  FAILED (exit code $exitCode)" -ForegroundColor Red
        if ($errOutput) { Write-Host "  stderr: $($errOutput.Substring(0, [Math]::Min(200, $errOutput.Length)))" -ForegroundColor Red }
        return $null
    }
    $metrics = Parse-Metrics $output
    $metrics["WallTime"] = [math]::Round($sw.Elapsed.TotalSeconds, 2)

    # Functional quality check
    $qc = Check-TextQuality $output
    $metrics["QualityOK"] = $qc.OK
    $metrics["QualityReason"] = $qc.Reason
    $metrics["QualityPreview"] = $qc.Text
    $metrics["UniqueRatio"] = $qc.UniqueRatio

    if ($qc.OK) {
        Write-Host ("  Done ({0}s) TTFT={1}ms TPOT={2}ms/tok TP={3}tok/s  [QUALITY: OK]" -f $metrics.WallTime, $metrics.TTFT, $metrics.TPOT, $metrics.Throughput) -ForegroundColor Green
    } else {
        Write-Host ("  Done ({0}s) TTFT={1}ms TPOT={2}ms/tok TP={3}tok/s" -f $metrics.WallTime, $metrics.TTFT, $metrics.TPOT, $metrics.Throughput) -ForegroundColor Green
        Write-Host ("  [QUALITY: FAIL] {0}" -f $qc.Reason) -ForegroundColor Red
        Write-Host ("  Preview: {0}" -f $qc.Text) -ForegroundColor DarkYellow
    }
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
        MTPDraftAcceptRate = "N/A"; MTPMeanAccepted = "N/A"
        TokensPerInfer = "N/A"
        Count = $valid.Count
    }
    foreach ($key in @("TTFT", "TPOT", "Throughput", "DecodeTime", "MTPRate", "MTPDraftAcceptRate", "MTPMeanAccepted", "TokensPerInfer")) {
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

    # Aggregate quality check results
    $qOkCount = 0; $qFailCount = 0; $qReasons = [System.Collections.ArrayList]@()
    foreach ($item in $valid) {
        if ($item["QualityOK"] -eq $true) { $qOkCount++ }
        else { $qFailCount++; if ($item["QualityReason"]) { [void]$qReasons.Add($item["QualityReason"]) } }
    }
    $avg["QualityOK"] = $qOkCount
    $avg["QualityFail"] = $qFailCount
    $avg["QualityStatus"] = if ($qFailCount -eq 0) { "OK" } else { "FAIL($qFailCount/$($valid.Count))" }
    $avg["QualityReasons"] = ($qReasons | Sort-Object -Unique) -join "; "

    return $avg
}

# --- Configs ---
$configs = @(
    @{ Name = "No MTP (baseline)"; MTP = 0; MtpK = 0 }
    @{ Name = "MTP K=1";           MTP = 1; MtpK = 1 }
    @{ Name = "MTP K=2";           MTP = 1; MtpK = 2 }
    @{ Name = "MTP K=3";           MTP = 1; MtpK = 3 }
)

$modeList = [System.Collections.ArrayList]@()
if ($Mode -eq "all" -or $Mode -eq "text") { [void]$modeList.Add("text") }
if ($Mode -eq "all" -or $Mode -eq "vl")   { [void]$modeList.Add("vl") }

# --- Run benchmarks ---
$allResults = @{}

foreach ($mdl in $MODELS) {
    $MODEL_DIR = $mdl.Path
    $modelName = $mdl.Name

    Write-Host "`n╔══════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║  Model: $modelName" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Yellow

    foreach ($m in $modeList) {
        Write-Host "`n========================================" -ForegroundColor Yellow
        Write-Host " $modelName | Mode: $m  |  Output tokens: $OutputTokens  |  Runs: $NumRuns" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow

        foreach ($cfg in $configs) {
            $key = "$modelName|$m|$($cfg.Name)"
            $runResults = [System.Collections.ArrayList]::new()

            # Build argument list
            $argList = [System.Collections.ArrayList]@(
                "--model", $MODEL_DIR,
                "--device", $env:DEVICE,
                "--mode", $m,
                "--output-tokens", "$OutputTokens",
                "--think", "0",
                "--temperature", "0"
            )
            if ($m -eq "text") {
                [void]$argList.AddRange(@("--prompt", "`"Hello, please write a short story about a robot learning to paint.`""))
            } else {
                [void]$argList.AddRange(@("--image", $IMAGE_PATH, "--prompt", "`"describe this picture in details: `""))
            }
            if ($cfg.MTP -gt 0) {
                [void]$argList.AddRange(@("--mtp", "1", "--mtp-k", "$($cfg.MtpK)", "--seq-verify", "1"))
            }

            Write-Host "`n--- $($cfg.Name) ($modelName / $m mode) ---" -ForegroundColor White
            for ($i = 1; $i -le $NumRuns; $i++) {
                $label = "$modelName $($cfg.Name) run $i/$NumRuns"
                $logTag = "${modelName}_${m}_$($cfg.Name)_run${i}"
                $result = Run-SingleBenchmark $label ([string[]]$argList) $logTag
                [void]$runResults.Add($result)
            }
            $allResults[$key] = $runResults
        }
    }
}

# --- Print Summary Table ---
Write-Host "`n`n" -NoNewline
Write-Host "=================================================================" -ForegroundColor Yellow
Write-Host " PERFORMANCE COMPARISON SUMMARY" -ForegroundColor Yellow
Write-Host " Quant: INT4_ASYM g128 | Device: $($env:DEVICE) | Sampling: greedy (T=0)" -ForegroundColor Yellow
Write-Host " Output tokens: $OutputTokens | Runs per config: $NumRuns" -ForegroundColor Yellow
Write-Host " Logs: $LOG_DIR" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow

foreach ($mdl in $MODELS) {
    $modelName = $mdl.Name

    foreach ($m in $modeList) {
        Write-Host "`n--- $modelName / $($m.ToUpper()) ---" -ForegroundColor Cyan

        $header = "{0,-22} {1,10} {2,14} {3,16} {4,14} {5,12} {6,12} {7,10} {8,12}" -f "Config", "TTFT(ms)", "TPOT(ms/tok)", "Throughput(t/s)", "Decode(ms)", "Accept%", "Tok/Infer", "Avg Acc", "Quality"
        $sep = "-" * 132
        Write-Host $sep
        Write-Host $header
        Write-Host $sep

        $baselineAvg = $null
        foreach ($cfg in $configs) {
            $key = "$modelName|$m|$($cfg.Name)"
            $avg = Get-Averages $allResults[$key]

            if ($null -eq $avg) {
                $row = "{0,-22} {1,10} {2,14} {3,16} {4,14} {5,12} {6,12} {7,10} {8,12}" -f $cfg.Name, "FAILED", "FAILED", "FAILED", "FAILED", "N/A", "N/A", "N/A", "N/A"
                Write-Host $row -ForegroundColor Red
                continue
            }

            if ($cfg.MTP -eq 0) { $baselineAvg = $avg }

            $accStr = if ($avg.MTPDraftAcceptRate -ne "N/A") { "$($avg.MTPDraftAcceptRate)%" } else { "N/A" }
            $tpiStr = if ($avg.TokensPerInfer -ne "N/A") { "$($avg.TokensPerInfer)" } else { "N/A" }
            $maStr = if ($avg.MTPMeanAccepted -ne "N/A") { "$($avg.MTPMeanAccepted)" } else { "N/A" }
            $qStr = $avg.QualityStatus
            $row = "{0,-22} {1,10} {2,14} {3,16} {4,14} {5,12} {6,12} {7,10} {8,12}" -f $cfg.Name, $avg.TTFT, $avg.TPOT, $avg.Throughput, $avg.DecodeTime, $accStr, $tpiStr, $maStr, $qStr
            if ($avg.QualityFail -gt 0) {
                Write-Host $row -ForegroundColor Red
            } else {
                Write-Host $row
            }
        }
        Write-Host $sep

        # Speedup/overhead summary
        if ($null -ne $baselineAvg) {
            foreach ($cfg in $configs) {
                if ($cfg.MTP -eq 0) { continue }
                $key = "$modelName|$m|$($cfg.Name)"
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
}

# --- Per-Run Details ---
Write-Host "`n`n--- PER-RUN DETAILS ---" -ForegroundColor Yellow
foreach ($mdl in $MODELS) {
    $modelName = $mdl.Name
    foreach ($m in $modeList) {
        foreach ($cfg in $configs) {
            $key = "$modelName|$m|$($cfg.Name)"
            $runResults = $allResults[$key]
            Write-Host ("`n  {0} ({1} / {2}):" -f $cfg.Name, $modelName, $m) -ForegroundColor White
        for ($i = 0; $i -lt $runResults.Count; $i++) {
            $r = $runResults[$i]
            if ($null -eq $r) {
                Write-Host "    Run $($i+1): FAILED"
                continue
            }
            $mtpStr = if ($r.MTPHits -ne "N/A") { " Cond=$($r.MTPHits)($($r.MTPRate)%)" } else { "" }
            $accStr = if ($r.MTPDraftAcceptRate -ne "N/A") { " Accept=$($r.MTPDraftAcceptRate)%" } else { "" }
            $maStr = if ($r.MTPMeanAccepted -ne "N/A") { " AvgAcc=$($r.MTPMeanAccepted)" } else { "" }
            $tpiStr = if ($r.TokensPerInfer -ne "N/A") { " Tok/Infer=$($r.TokensPerInfer)" } else { "" }
            $qStr = if ($r["QualityOK"] -eq $true) { " [OK]" } else { " [FAIL: $($r['QualityReason'])]" }
            $lineColor = if ($r["QualityOK"] -eq $true) { "White" } else { "Red" }
            Write-Host ("    Run {0}: TTFT={1}ms TPOT={2}ms/tok TP={3}tok/s Decode={4}ms{5}{6}{7}{8}{9}" -f ($i+1), $r.TTFT, $r.TPOT, $r.Throughput, $r.DecodeTime, $mtpStr, $accStr, $maStr, $tpiStr, $qStr) -ForegroundColor $lineColor
        }
    }
}
}

Write-Host "`n`nBenchmark complete. Logs saved to: $LOG_DIR" -ForegroundColor Green
