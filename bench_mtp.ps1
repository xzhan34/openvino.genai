# bench_mtp.ps1 - Benchmark MTP performance comparison for Qwen3.5
# Usage: powershell -ExecutionPolicy Bypass -File bench_mtp.ps1

param(
    [int]$NumRuns = 1,
    [int]$OutputTokens = 256,
    [string]$Mode = "all",  # "text", "vl", or "all"
    [string]$ConfigFilter = "*"  # Wildcard filter on config Name, e.g. "*f16*" or "baseline*"
)

# --- Environment Setup ---
$GENAI_DIR = "C:\work\openvino_ws\openvino.genai.liangali"
$OV_DIR    = "C:\work\openvino_ws\openvino.liangali"
$env:OPENVINO_TOKENIZERS_PATH_GENAI = "$GENAI_DIR\build-master\openvino_genai\openvino_tokenizers.dll"
$env:PATH = "$OV_DIR\bin\intel64\RelWithDebInfo;$OV_DIR\temp\Windows_AMD64\tbb\bin;$GENAI_DIR\build-master\openvino_genai;$env:PATH"
$env:OV_GENAI_USE_MODELING_API = "1"
# Quant env vars are set per-config (int4 vs f16). Defaults cleared here.
Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE -ErrorAction SilentlyContinue
$env:OV_GENAI_MTP_SNAPSHOT = "1"
$env:OV_GENAI_SNAPSHOT_RESTORE = "3"   # GPU-side state restore on draft rejection (critical for MTP perf)
$env:OV_GENAI_VALIDATE_SNAPSHOT = "0"  # Disable snapshot validation overhead in benchmarks
# GPU kernel batch-invariance for MTP speculative decoding:
#   oneDNN batch-1 loop: For INT4 compressed weights with verify batch M>1,
#     the GPU plugin now executes the M=1 oneDNN matmul primitive M times with
#     per-row offsets, ensuring bit-identical computation to decode (M=1).
#     Controlled by OV_GPU_ONEDNN_FC_BATCH1_MAX (default 8, 0=disable).
#   OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD=K+1: Force single-token SDPA kernel for
#     verify batch to eliminate batch-dependent SDPA tiling divergence.
# Clean up stale env vars that could interfere with auto-configuration
Remove-Item Env:\OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD -ErrorAction SilentlyContinue
Remove-Item Env:\OV_GPU_USE_ONEDNN -ErrorAction SilentlyContinue
Remove-Item Env:\OV_GPU_FC_SINGLE_BATCH_THRESHOLD -ErrorAction SilentlyContinue
Remove-Item Env:\OV_GPU_ONEDNN_FC_BATCH1_MAX -ErrorAction SilentlyContinue
# Enable per-step profiling breakdown (sub-step timing to stderr, summary to stdout)
$env:OV_GENAI_STEP_PROFILE = "1"
$env:DEVICE = "GPU"

$EXE       = "$GENAI_DIR\build-master\src\cpp\src\modeling\samples\RelWithDebInfo\modeling_qwen3_5.exe"
$IMAGE_PATH = "$OV_DIR\docs\articles_en\assets\images\get_started_with_cpp.jpg"

# --- Models to benchmark ---
$MODELS = @(
    #@{ Name = "Qwen3.5-0.8B"; Path = "C:\work\openvino_ws\openvino.genai.xzhan34\tests\module_genai\cpp\test_models\Qwen3.5-0.8B" }
    @{ Name = "Qwen3.5-9B";   Path = "C:\work\models\Qwen3.5-9B" }
)

# --- Logging Setup ---
$LOG_ROOT = Join-Path $GENAI_DIR "OV_Logs"
$BENCH_TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_DIR = Join-Path $LOG_ROOT $BENCH_TIMESTAMP
New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null

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
        AcceptedTokens = "N/A"
        DraftTokens    = "N/A"
        VerifyAvgMs    = "N/A"
        DraftAvgMs     = "N/A"
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
        elseif ($line -match '^MTP draft acceptance:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)') {
            $metrics.MTPDraftAcceptRate = [double]$Matches[3]
            $metrics.AcceptedTokens = [int]$Matches[1]
            $metrics.DraftTokens = [int]$Matches[2]
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
        elseif ($line -match 'Dead KV positions:\s+(\d+)') {
            $metrics["DeadKVPositions"] = [int]$Matches[1]
        }
        elseif ($line -match 'Main verify \(K\+1\):\s+[\d.]+\s+ms\s+\(\d+\s+calls\),\s+avg\s+([\d.]+)\s+ms') {
            $metrics.VerifyAvgMs = [math]::Round([double]$Matches[1], 2)
        }
        elseif ($line -match 'MTP draft \(x\d+\):\s+[\d.]+\s+ms\s+\(\d+\s+calls\),\s+avg\s+([\d.]+)\s+ms') {
            $metrics.DraftAvgMs = [math]::Round([double]$Matches[1], 2)
        }
    }
    return $metrics
}

# --- Helper: check generated text quality (detect degenerate repetition) ---
function Check-TextQuality([string]$output) {
    # Extract generated text: everything after the last metric/profiling line.
    # The text is at the end of stdout, after all metric/profiling lines.
    $lines = $output -split "`r?`n"
    $textStartIdx = -1
    # Scan from bottom up, skip trailing empty lines first, then find last metric line.
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        $l = $lines[$i].Trim()
        if ($l -eq '') { continue }  # skip trailing/inter-paragraph blanks
        if ($l -match '^(TTFT|TPOT|Throughput|Decode time|MTP |--- |Main verify|Main GPU|MTP draft|KV trim|Dead KV|Snapshot save|Restore|State refresh|State restore|Accept|Output token|Prompt token|Mode:|Avg step|\-{5}|\[)') {
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

    # Check 4: single-character repetition (e.g., "!!!!!!!!!" or "........")
    $maxCharRepeat = 0
    $repeatChar = ""
    if ($genText.Length -gt 0) {
        $currentCharRun = 1
        for ($ci = 1; $ci -lt $genText.Length; $ci++) {
            if ($genText[$ci] -eq $genText[$ci - 1] -and $genText[$ci] -ne ' ') {
                $currentCharRun++
                if ($currentCharRun -gt $maxCharRepeat) {
                    $maxCharRepeat = $currentCharRun
                    $repeatChar = $genText[$ci]
                }
            } else {
                $currentCharRun = 1
            }
        }
    }

    if ($maxCharRepeat -ge 10) {
        $isDegenerate = $true
        $reason = "CHAR_REPEAT: '$repeatChar' x$maxCharRepeat"
    } elseif ($maxRepeatRun -ge 5) {
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
    $logFile = Join-Path $LOG_DIR "${safeTag}.log"
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
            if ($tl -eq '') { continue }
            if ($tl -match '^(TTFT|TPOT|Throughput|Decode time|MTP |--- |Main verify|Main GPU|MTP draft|KV trim|Dead KV|Snapshot save|Restore|State refresh|State restore|Accept|Output token|Prompt token|Mode:|Avg step|\-{5}|\[)') {
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
        AcceptedTokens = "N/A"; DraftTokens = "N/A"; OutputSize = "N/A"
        MTPInfers = "N/A"
        VerifyAvgMs = "N/A"; DraftAvgMs = "N/A"
        Count = $valid.Count
    }
    foreach ($key in @("TTFT", "TPOT", "Throughput", "DecodeTime", "MTPRate", "MTPDraftAcceptRate", "MTPMeanAccepted", "TokensPerInfer", "VerifyAvgMs", "DraftAvgMs")) {
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

    # Sum-based fields (not averaged): AcceptedTokens, DraftTokens, OutputSize, MTPInfers
    foreach ($sumKey in @("AcceptedTokens", "DraftTokens", "OutputSize", "MTPInfers")) {
        $vals = [System.Collections.ArrayList]@()
        foreach ($item in $valid) {
            $v = $item[$sumKey]
            if ($v -ne "N/A" -and $v -is [int]) { [void]$vals.Add($v) }
        }
        if ($vals.Count -gt 0) {
            $sum = 0; foreach ($v in $vals) { $sum += $v }
            $avg[$sumKey] = [math]::Round($sum / $vals.Count, 0)
        }
    }

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

# --- Quant helper: set env vars for a given precision ---
function Set-QuantEnv([string]$quant) {
    if ($quant -eq "int4") {
        $env:OV_GENAI_INFLIGHT_QUANT_MODE = "int4_asym"
        $env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE = "128"
        $env:OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE = "int4_asym"
        # Force oneDNN ON: batch-1 loop in GPU plugin ensures decode/verify
        # numerical identity for INT4 compressed FC (no more auto-disable needed).
        $env:OV_GPU_USE_ONEDNN = "1"
        Remove-Item Env:\OV_GPU_ONEDNN_FC_BATCH1_MAX -ErrorAction SilentlyContinue
    } elseif ($quant -eq "int4+ocl") {
        # INT4 with oneDNN disabled (OCL bf_tiled path) for A/B comparison
        $env:OV_GENAI_INFLIGHT_QUANT_MODE = "int4_asym"
        $env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE = "128"
        $env:OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE = "int4_asym"
        $env:OV_GPU_USE_ONEDNN = "0"
        Remove-Item Env:\OV_GPU_ONEDNN_FC_BATCH1_MAX -ErrorAction SilentlyContinue
    } elseif ($quant -eq "int8") {
        $env:OV_GENAI_INFLIGHT_QUANT_MODE = "int8_asym"
        Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE -ErrorAction SilentlyContinue
        $env:OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE = "int8_asym"
        # Restore default oneDNN (exe auto-disables for pure-batch MTP)
        Remove-Item Env:\OV_GPU_USE_ONEDNN -ErrorAction SilentlyContinue
    } elseif ($quant -eq "f16+dnn") {
        # f16 with oneDNN enabled: A/B test to measure oneDNN f16 FC perf impact
        # WARNING: produces degenerate output on VL mode (oneDNN f16 FC bug)
        Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE -ErrorAction SilentlyContinue
        Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:\OV_GPU_USE_ONEDNN -ErrorAction SilentlyContinue
    } else {
        # f16: disable in-flight quantization
        Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE -ErrorAction SilentlyContinue
        Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE -ErrorAction SilentlyContinue
        # Disable oneDNN FC for f16: oneDNN f16 FC kernels produce degenerate output
        # for VL visual token distributions on Arc Pro 140T GPU.
        $env:OV_GPU_USE_ONEDNN = "0"
    }
}

# --- Configs ---
# Two verify modes: seq (--seq-verify 1, batch=1) vs batch (--pure-batch 1, batch=K+1)
# Two precisions: int4 (INT4_ASYM g128) vs f16 (native f16, no quantization)
$configs = @(
    # --- F16 (run first: no quantization, native precision) ---
    @{ Name = "baseline f16";       MTP = 0; MtpK = 0; Verify = "none";  Quant = "f16" }
    @{ Name = "baseline f16+dnn";   MTP = 0; MtpK = 0; Verify = "none";  Quant = "f16+dnn" }
    @{ Name = "K=1 seq f16";        MTP = 1; MtpK = 1; Verify = "seq";   Quant = "f16" }
    @{ Name = "K=1 batch f16";      MTP = 1; MtpK = 1; Verify = "batch"; Quant = "f16" }
    @{ Name = "K=2 seq f16";        MTP = 1; MtpK = 2; Verify = "seq";   Quant = "f16" }
    @{ Name = "K=2 batch f16";      MTP = 1; MtpK = 2; Verify = "batch"; Quant = "f16" }
    @{ Name = "K=3 seq f16";        MTP = 1; MtpK = 3; Verify = "seq";   Quant = "f16" }
    @{ Name = "K=3 batch f16";      MTP = 1; MtpK = 3; Verify = "batch"; Quant = "f16" }
    # --- INT8 ---
    @{ Name = "baseline int8";      MTP = 0; MtpK = 0; Verify = "none";  Quant = "int8" }
    @{ Name = "K=1 seq int8";       MTP = 1; MtpK = 1; Verify = "seq";   Quant = "int8" }
    @{ Name = "K=1 batch int8";     MTP = 1; MtpK = 1; Verify = "batch"; Quant = "int8" }
    @{ Name = "K=2 seq int8";       MTP = 1; MtpK = 2; Verify = "seq";   Quant = "int8" }
    @{ Name = "K=2 batch int8";     MTP = 1; MtpK = 2; Verify = "batch"; Quant = "int8" }
    @{ Name = "K=3 seq int8";       MTP = 1; MtpK = 3; Verify = "seq";   Quant = "int8" }
    @{ Name = "K=3 batch int8";     MTP = 1; MtpK = 3; Verify = "batch"; Quant = "int8" }
    # --- INT4 (oneDNN + batch-1 loop) ---
    @{ Name = "baseline int4";      MTP = 0; MtpK = 0; Verify = "none";  Quant = "int4" }
    @{ Name = "K=1 seq int4";       MTP = 1; MtpK = 1; Verify = "seq";   Quant = "int4" }
    @{ Name = "K=1 batch int4";     MTP = 1; MtpK = 1; Verify = "batch"; Quant = "int4" }
    @{ Name = "K=2 seq int4";       MTP = 1; MtpK = 2; Verify = "seq";   Quant = "int4" }
    @{ Name = "K=2 batch int4";     MTP = 1; MtpK = 2; Verify = "batch"; Quant = "int4" }
    @{ Name = "K=3 seq int4";       MTP = 1; MtpK = 3; Verify = "seq";   Quant = "int4" }
    @{ Name = "K=3 batch int4";     MTP = 1; MtpK = 3; Verify = "batch"; Quant = "int4" }
    # --- INT4+OCL (oneDNN disabled, OCL bf_tiled FC only) for A/B comparison ---
    @{ Name = "baseline int4+ocl";  MTP = 0; MtpK = 0; Verify = "none";  Quant = "int4+ocl" }
    @{ Name = "K=1 batch int4+ocl"; MTP = 1; MtpK = 1; Verify = "batch"; Quant = "int4+ocl" }
    @{ Name = "K=2 batch int4+ocl"; MTP = 1; MtpK = 2; Verify = "batch"; Quant = "int4+ocl" }
)

# Apply config filter
if ($ConfigFilter -ne "*") {
    $configs = @($configs | Where-Object { $_.Name -like $ConfigFilter })
    Write-Host "Config filter '$ConfigFilter' -> $($configs.Count) config(s): $($configs.Name -join ', ')" -ForegroundColor Cyan
}

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

            # Set quant env vars for this config
            Set-QuantEnv $cfg.Quant

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
                [void]$argList.AddRange(@("--image", $IMAGE_PATH, "--prompt", "`"describe this picture in details`""))
            }
            if ($cfg.MTP -gt 0) {
                [void]$argList.AddRange(@("--mtp", "1", "--mtp-k", "$($cfg.MtpK)"))
                if ($cfg.Verify -eq "batch") {
                    [void]$argList.AddRange(@("--pure-batch", "1"))
                    # OV_GPU_USE_ONEDNN=0 and OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD=K+1
                    # are auto-set by the exe for MTP pure-batch mode.
                } elseif ($cfg.Verify -eq "seq") {
                    [void]$argList.AddRange(@("--seq-verify", "1"))
                }
            } else {
                Remove-Item Env:\OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD -ErrorAction SilentlyContinue
                # OV_GPU_USE_ONEDNN is managed by Set-QuantEnv (disabled for f16 to avoid
                # oneDNN f16 FC degeneration on VL visual token distributions)
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
Write-Host " Precisions: INT4_ASYM g128 + INT8_ASYM + F16 | Device: $($env:DEVICE) | Sampling: greedy (T=0)" -ForegroundColor Yellow
Write-Host " Output tokens: $OutputTokens | Runs per config: $NumRuns" -ForegroundColor Yellow
Write-Host " Logs: $LOG_DIR" -ForegroundColor Yellow
Write-Host " Note: OutTok = MainInfers + AcceptTok  |  DraftTok = (MainInfers - 1) * K  |  Acc/Dft% = AcceptTok / DraftTok  |  Acc/Out% = AcceptTok / OutTok" -ForegroundColor Yellow
Write-Host "       Verify(ms) = avg main model K+1 batch verify infer  |  Draft(ms) = avg single MTP head infer (called K times/step)" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow

foreach ($mdl in $MODELS) {
    $modelName = $mdl.Name

    foreach ($m in $modeList) {
        Write-Host "`n--- $modelName / $($m.ToUpper()) ---" -ForegroundColor Cyan

        $header = "{0,-26} {1,10} {2,14} {3,16} {4,14} {5,12} {6,12} {7,12} {8,10} {9,10} {10,10} {11,8} {12,10} {13,13} {14,13} {15,10}" -f "Config", "TTFT(ms)", "TPOT(ms/tok)", "Throughput(t/s)", "Decode(ms)", "Acc/Dft%", "Acc/Out%", "Tok/Infer", "Avg Acc", "AcceptTok", "DraftTok", "OutTok", "MainInfers", "Verify(ms)", "Draft(ms)", "Quality"
        $sep = "-" * 225
        Write-Host $sep
        Write-Host $header
        Write-Host $sep

        $baselineInt4 = $null
        $baselineInt4Ocl = $null
        $baselineInt8 = $null
        $baselineF16 = $null
        foreach ($cfg in $configs) {
            $key = "$modelName|$m|$($cfg.Name)"
            $avg = Get-Averages $allResults[$key]

            if ($null -eq $avg) {
                $row = "{0,-26} {1,10} {2,14} {3,16} {4,14} {5,12} {6,12} {7,12} {8,10} {9,10} {10,10} {11,8} {12,10} {13,13} {14,13} {15,10}" -f $cfg.Name, "FAILED", "FAILED", "FAILED", "FAILED", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
                Write-Host $row -ForegroundColor Red
                continue
            }

            if ($cfg.MTP -eq 0 -and $cfg.Quant -eq "int4") { $baselineInt4 = $avg }
            if ($cfg.MTP -eq 0 -and $cfg.Quant -eq "int4+ocl") { $baselineInt4Ocl = $avg }
            if ($cfg.MTP -eq 0 -and $cfg.Quant -eq "int8") { $baselineInt8 = $avg }
            if ($cfg.MTP -eq 0 -and $cfg.Quant -eq "f16")  { $baselineF16 = $avg }

            # Acc/Dft% = AcceptedTokens / DraftTokens * 100 (draft acceptance rate)
            #   由 exe 输出的 "MTP draft acceptance: AcceptedTokens/DraftTokens (X%)" 直接解析得到。
            #   AcceptedTokens: 被主模型验证通过的 draft token 数量
            #   DraftTokens: K 个 MTP draft 头总共生成的候选 token 数量 = (MainInfers - 1) * K
            # Acc/Out% = AcceptedTokens / OutputSize * 100 (output中来自draft的比例)
            #   OutputSize = MainInfers + AcceptedTokens，所以 Acc/Out% 体现 speculative decoding 的整体收益。
            $accDftStr = if ($avg.MTPDraftAcceptRate -ne "N/A") { "$($avg.MTPDraftAcceptRate)%" } else { "N/A" }
            $accOutStr = if ($avg.AcceptedTokens -ne "N/A" -and $avg.OutputSize -ne "N/A" -and $avg.OutputSize -gt 0) {
                "$([math]::Round($avg.AcceptedTokens / $avg.OutputSize * 100, 1))%"
            } else { "N/A" }
            $tpiStr = if ($avg.TokensPerInfer -ne "N/A") { "$($avg.TokensPerInfer)" } else { "N/A" }
            $maStr = if ($avg.MTPMeanAccepted -ne "N/A") { "$($avg.MTPMeanAccepted)" } else { "N/A" }
            $aTokStr = if ($avg.AcceptedTokens -ne "N/A") { "$($avg.AcceptedTokens)" } else { "N/A" }
            $dTokStr = if ($avg.DraftTokens -ne "N/A") { "$($avg.DraftTokens)" } else { "N/A" }
            $oTokStr = if ($avg.OutputSize -ne "N/A") { "$($avg.OutputSize)" } else { "N/A" }
            $miStr = if ($avg.MTPInfers -ne "N/A") { "$($avg.MTPInfers)" } else { "N/A" }
            $vAvgStr = if ($avg.VerifyAvgMs -ne "N/A") { "$($avg.VerifyAvgMs)" } else { "N/A" }
            $dAvgStr = if ($avg.DraftAvgMs -ne "N/A") { "$($avg.DraftAvgMs)" } else { "N/A" }
            $qStr = $avg.QualityStatus
            $row = "{0,-26} {1,10} {2,14} {3,16} {4,14} {5,12} {6,12} {7,12} {8,10} {9,10} {10,10} {11,8} {12,10} {13,13} {14,13} {15,10}" -f $cfg.Name, $avg.TTFT, $avg.TPOT, $avg.Throughput, $avg.DecodeTime, $accDftStr, $accOutStr, $tpiStr, $maStr, $aTokStr, $dTokStr, $oTokStr, $miStr, $vAvgStr, $dAvgStr, $qStr
            if ($avg.QualityFail -gt 0) {
                Write-Host $row -ForegroundColor Red
            } else {
                Write-Host $row
            }
        }
        Write-Host $sep

        # Speedup/overhead summary (each MTP config vs its same-quant baseline)
        foreach ($cfg in $configs) {
            if ($cfg.MTP -eq 0) { continue }
            $baselineAvg = switch ($cfg.Quant) {
                "int4" { $baselineInt4 }
                "int4+ocl" { $baselineInt4Ocl }
                "int8" { $baselineInt8 }
                default { $baselineF16 }
            }
            if ($null -eq $baselineAvg) { continue }
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
