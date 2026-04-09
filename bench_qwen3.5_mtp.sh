#!/bin/bash
# bench_mtp.sh - Benchmark MTP performance comparison for Qwen3.5
# Converted from bench_mtp.ps1 (PowerShell) to Linux bash
# Usage: bash bench_mtp.sh [options]
#   -n NUM_RUNS       Number of runs per config (default: 1)
#   -t OUTPUT_TOKENS  Output tokens (default: 256)
#   -m MODE           "text", "vl", or "all" (default: "all")
#   -f CONFIG_FILTER  Wildcard filter on config Name, e.g. "*f16*" (default: "*")
#   -c COOLDOWN_SEC   Seconds between configs for GPU thermal stabilization (default: 15, 0=disabled)
#   -s                Include seq-verify configs (excluded by default)

set -uo pipefail

# --- Argument Parsing ---
NUM_RUNS=1
OUTPUT_TOKENS=256
MODE="all"
CONFIG_FILTER="*"
COOLDOWN_SEC=15
INCLUDE_SEQ=0

while getopts "n:t:m:f:c:s" opt; do
    case $opt in
        n) NUM_RUNS="$OPTARG" ;;
        t) OUTPUT_TOKENS="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        f) CONFIG_FILTER="$OPTARG" ;;
        c) COOLDOWN_SEC="$OPTARG" ;;
        s) INCLUDE_SEQ=1 ;;
        *) echo "Usage: $0 [-n runs] [-t tokens] [-m mode] [-f filter] [-c cooldown] [-s]"; exit 1 ;;
    esac
done

# --- Environment Setup ---
GENAI_DIR="/home/xzhan34/openvino_ws_2026.0/openvino.genai.xzhan34"
OV_DIR="/home/xzhan34/openvino_ws_2026.0/openvino.xzhan34"
IMAGE_PATH="${OV_DIR}/docs/articles_en/assets/images/get_started_with_cpp.jpg"

# --- Models to benchmark ---
declare -a MODEL_NAMES=("Qwen3.5-9B")
declare -A MODEL_PATHS=(
    ["Qwen3.5-9B"]="/home/intel/models/Qwen3.5-9B"
)

# Enable per-step profiling breakdown
export OV_GENAI_STEP_PROFILE=1
# Force GPU.1 — override by passing DEVICE=GPU.0 before the script
export DEVICE="GPU"

GENAI_BUILD="${GENAI_DIR}/build"
OV_LIB_DIR="${OV_DIR}/bin/intel64/RelWithDebInfo"
TBB_LIB_DIR="${OV_DIR}/temp/Linux_x86_64/tbb/lib"

export LD_LIBRARY_PATH="${GENAI_BUILD}/openvino_genai:${OV_LIB_DIR}:${TBB_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OV_GENAI_USE_MODELING_API=1
# Quant env vars are set per-config (int4 vs f16). Defaults cleared here.
unset OV_GENAI_INFLIGHT_QUANT_MODE 2>/dev/null || true
unset OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE 2>/dev/null || true
unset OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE 2>/dev/null || true
export OV_GENAI_MTP_SNAPSHOT=1
export OV_GENAI_SNAPSHOT_RESTORE=3    # linear + conv GPU kernel snapshot restore (memory-based f16 rounding fix)
export OV_GENAI_VALIDATE_SNAPSHOT=0   # Disable snapshot validation overhead
# Clean up stale env vars that may linger from manual testing
unset OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD 2>/dev/null || true
unset OV_GPU_USE_ONEDNN 2>/dev/null || true
unset OV_GPU_FC_SINGLE_BATCH_THRESHOLD 2>/dev/null || true
unset OV_GPU_ONEDNN_FC_BATCH1_MAX 2>/dev/null || true
unset OV_GENAI_USE_KERNEL_SNAPSHOT 2>/dev/null || true
unset OV_GENAI_GPU_RESTORE_CONV 2>/dev/null || true

EXE="${GENAI_BUILD}/bin/modeling_qwen3_5"

# --- Logging Setup ---
LOG_ROOT="${GENAI_DIR}/OV_Logs"
BENCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${LOG_ROOT}/${BENCH_TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# --- Sanity checks ---
if [[ ! -x "${EXE}" ]]; then
    echo "ERROR: Binary not found or not executable: ${EXE}"
    echo "       Build openvino.genai.xzhan34 first."
    exit 1
fi

# =============================================================================
# Helper: parse metrics from stdout
# Sets global associative array METRICS
# =============================================================================
declare -A METRICS
parse_metrics() {
    local output="$1"
    METRICS=(
        [TTFT]="N/A" [DecodeTime]="N/A" [TPOT]="N/A" [Throughput]="N/A"
        [MTPHits]="N/A" [MTPRate]="N/A" [MTPDraftAcceptRate]="N/A"
        [MTPMeanAccepted]="N/A" [OutputSize]="N/A" [MTPInfers]="N/A"
        [TokensPerInfer]="N/A" [AcceptedTokens]="N/A" [DraftTokens]="N/A"
        [VerifyAvgMs]="N/A" [DraftAvgMs]="N/A"
    )
    while IFS= read -r line; do
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [[ "$line" =~ ^TTFT:\ +([0-9.]+)\ +ms ]]; then
            METRICS[TTFT]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^Decode\ time:\ +([0-9.]+)\ +ms ]]; then
            METRICS[DecodeTime]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^TPOT:\ +([0-9.]+)\ +ms/token ]]; then
            METRICS[TPOT]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^Throughput:\ +([0-9.]+)\ +tokens/s ]]; then
            METRICS[Throughput]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^Output\ token\ size:\ +([0-9]+) ]]; then
            METRICS[OutputSize]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^MTP\ hits:\ +([0-9]+)/([0-9]+)\ +\(([0-9.]+)%\) ]]; then
            METRICS[MTPHits]="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
            METRICS[MTPRate]="${BASH_REMATCH[3]}"
        elif [[ "$line" =~ ^MTP\ draft\ acceptance:\ +([0-9]+)/([0-9]+)\ +\(([0-9.]+)%\) ]]; then
            METRICS[MTPDraftAcceptRate]="${BASH_REMATCH[3]}"
            METRICS[AcceptedTokens]="${BASH_REMATCH[1]}"
            METRICS[DraftTokens]="${BASH_REMATCH[2]}"
        elif [[ "$line" =~ ^MTP\ mean\ accepted/step:\ +([0-9.]+) ]]; then
            METRICS[MTPMeanAccepted]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^MTP\ main\ model\ infers:\ +([0-9]+) ]]; then
            METRICS[MTPInfers]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^MTP\ tokens/infer:\ +([0-9.]+) ]]; then
            METRICS[TokensPerInfer]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ Dead\ KV\ positions:\ +([0-9]+) ]]; then
            METRICS[DeadKVPositions]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ Main\ verify\ \(K\+1\):.*avg\ ([0-9.]+)\ ms ]]; then
            METRICS[VerifyAvgMs]="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ MTP\ draft\ \(x[0-9]+\):.*avg\ ([0-9.]+)\ ms ]]; then
            METRICS[DraftAvgMs]="${BASH_REMATCH[1]}"
        fi
    done <<< "$output"
}

# =============================================================================
# Helper: extract generated text from output
# Returns text via stdout
# =============================================================================
extract_generated_text() {
    local output="$1"
    local -a lines=()
    while IFS= read -r l; do
        lines+=("$l")
    done <<< "$output"

    local text_start=-1
    local i
    for (( i=${#lines[@]}-1; i>=0; i-- )); do
        local trimmed
        trimmed=$(echo "${lines[$i]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$trimmed" ]] && continue
        if [[ "$trimmed" =~ ^(TTFT|TPOT|Throughput|Decode\ time|MTP\ |---\ |Main\ verify|Main\ GPU|MTP\ draft|KV\ trim|Dead\ KV|Snapshot\ save|Restore|State\ refresh|State\ restore|Accept|Output\ token|Prompt\ token|Mode:|Avg\ step|-----|\[) ]]; then
            text_start=$((i + 1))
            break
        fi
    done

    if (( text_start < 0 || text_start >= ${#lines[@]} )); then
        echo ""
        return
    fi

    local result=""
    for (( i=text_start; i<${#lines[@]}; i++ )); do
        local trimmed
        trimmed=$(echo "${lines[$i]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -n "$trimmed" ]] && result="${result} ${trimmed}"
    done
    echo "${result}" | sed 's/^[[:space:]]//'
}

# =============================================================================
# Helper: check generated text quality (detect degenerate repetition)
# Sets global: QUALITY_OK (0/1), QUALITY_REASON, QUALITY_PREVIEW, QUALITY_UNIQUE_RATIO
# =============================================================================
QUALITY_OK=1
QUALITY_REASON="OK"
QUALITY_PREVIEW=""
QUALITY_UNIQUE_RATIO="1.0"

check_text_quality() {
    local output="$1"
    QUALITY_OK=1
    QUALITY_REASON="OK"
    QUALITY_PREVIEW=""
    QUALITY_UNIQUE_RATIO="1.0"

    local gen_text
    gen_text=$(extract_generated_text "$output")

    if [[ ${#gen_text} -lt 10 ]]; then
        QUALITY_OK=0
        QUALITY_REASON="TOO_SHORT"
        QUALITY_PREVIEW="$gen_text"
        return
    fi

    QUALITY_PREVIEW="${gen_text:0:80}"
    [[ ${#gen_text} -gt 80 ]] && QUALITY_PREVIEW="${QUALITY_PREVIEW}..."

    # Split into words
    local -a words=()
    read -ra words <<< "$gen_text"
    local word_count=${#words[@]}

    # Check 1: repeated word runs
    local max_repeat_run=0
    local repeat_word=""
    local current_run=1
    for (( w=1; w<word_count; w++ )); do
        if [[ "${words[$w]}" == "${words[$((w-1))]}" ]]; then
            ((current_run++))
            if (( current_run > max_repeat_run )); then
                max_repeat_run=$current_run
                repeat_word="${words[$w]}"
            fi
        else
            current_run=1
        fi
    done

    # Check 2: repeated bigrams
    local max_bigram_repeat=0
    local repeat_bigram=""
    if (( word_count >= 4 )); then
        local bigram_run=1
        for (( w=2; w<word_count-1; w+=2 )); do
            local prev="${words[$((w-2))]} ${words[$((w-1))]}"
            local curr="${words[$w]} ${words[$((w+1))]}"
            if [[ "$curr" == "$prev" ]]; then
                ((bigram_run++))
                if (( bigram_run > max_bigram_repeat )); then
                    max_bigram_repeat=$bigram_run
                    repeat_bigram="$curr"
                fi
            else
                bigram_run=1
            fi
        done
    fi

    # Check 3: unique word ratio in last 50% of text
    local half_idx=$(( word_count / 2 ))
    local -a last_half=("${words[@]:$half_idx}")
    local last_half_count=${#last_half[@]}
    local unique_count
    if (( last_half_count > 0 )); then
        unique_count=$(printf '%s\n' "${last_half[@]}" | sort -u | wc -l)
        QUALITY_UNIQUE_RATIO=$(awk "BEGIN { printf \"%.3f\", $unique_count / $last_half_count }")
    fi

    # Check 4: single-character repetition
    local max_char_repeat=0
    local repeat_char=""
    local text_len=${#gen_text}
    if (( text_len > 0 )); then
        local current_char_run=1
        for (( ci=1; ci<text_len; ci++ )); do
            local c="${gen_text:$ci:1}"
            local p="${gen_text:$((ci-1)):1}"
            if [[ "$c" == "$p" && "$c" != " " ]]; then
                ((current_char_run++))
                if (( current_char_run > max_char_repeat )); then
                    max_char_repeat=$current_char_run
                    repeat_char="$c"
                fi
            else
                current_char_run=1
            fi
        done
    fi

    # Thresholds for degenerate detection
    if (( max_char_repeat >= 10 )); then
        QUALITY_OK=0
        QUALITY_REASON="CHAR_REPEAT: '${repeat_char}' x${max_char_repeat}"
    elif (( max_repeat_run >= 5 )); then
        QUALITY_OK=0
        QUALITY_REASON="WORD_REPEAT: '${repeat_word}' x${max_repeat_run}"
    elif (( max_bigram_repeat >= 4 )); then
        QUALITY_OK=0
        QUALITY_REASON="BIGRAM_REPEAT: '${repeat_bigram}' x${max_bigram_repeat}"
    elif (( last_half_count >= 20 )); then
        local below
        below=$(awk "BEGIN { print ($QUALITY_UNIQUE_RATIO < 0.20) ? 1 : 0 }")
        if (( below == 1 )); then
            QUALITY_OK=0
            QUALITY_REASON="LOW_DIVERSITY: unique_ratio=${QUALITY_UNIQUE_RATIO} in last ${last_half_count} words"
        fi
    fi
}

# =============================================================================
# Helper: run one benchmark
# Returns metrics in RUN_RESULT_* globals; RUN_OK=0 if failed
# =============================================================================
RUN_OK=1

# Storage for per-run results (indexed by run number within a config)
# We store serialized key=value pairs in arrays for later aggregation.
declare -a RUN_RESULT_KEYS=()
declare -A RUN_RESULTS=()

run_single_benchmark() {
    local label="$1"
    shift
    local log_tag="$1"
    shift
    local -a exe_args=("$@")

    RUN_OK=1
    echo -e "  \033[36mRunning: ${label} ...\033[0m"

    local start_time
    start_time=$(date +%s%N)

    local temp_out
    temp_out=$(mktemp)
    local temp_err
    temp_err=$(mktemp)

    local exit_code=0
    "${EXE}" "${exe_args[@]}" > "$temp_out" 2> "$temp_err" || exit_code=$?

    local end_time
    end_time=$(date +%s%N)
    local wall_ms=$(( (end_time - start_time) / 1000000 ))
    local wall_sec
    wall_sec=$(awk "BEGIN { printf \"%.2f\", $wall_ms / 1000.0 }")

    local output
    output=$(cat "$temp_out" 2>/dev/null || true)
    local err_output
    err_output=$(cat "$temp_err" 2>/dev/null || true)
    rm -f "$temp_out" "$temp_err"

    # Save log
    local run_ts
    run_ts=$(date +"%Y%m%d_%H%M%S")
    local safe_tag
    safe_tag=$(echo "$log_tag" | tr ' /|' '___')
    local log_file="${LOG_DIR}/${safe_tag}.log"
    cat > "$log_file" << LOGEOF
=== Benchmark Run Log ===
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Label: ${label}
Command: ${EXE} ${exe_args[*]}
Exit Code: ${exit_code}
Wall Time: ${wall_sec}s

--- STDOUT ---
${output}
--- STDERR ---
${err_output}
LOGEOF
    echo -e "  \033[90mLog saved: ${log_file}\033[0m"

    # Print generated text preview
    if [[ -n "$output" ]]; then
        local gen_text
        gen_text=$(extract_generated_text "$output")
        if [[ -n "$gen_text" ]]; then
            local preview="${gen_text:0:200}"
            [[ ${#gen_text} -gt 200 ]] && preview="${preview}..."
            echo -e "  \033[96mGenerated: ${preview}\033[0m"
        fi
    fi

    if [[ $exit_code -ne 0 ]]; then
        echo -e "  \033[31mFAILED (exit code ${exit_code})\033[0m"
        if [[ -n "$err_output" ]]; then
            echo -e "  \033[31mstderr: ${err_output:0:200}\033[0m"
        fi
        RUN_OK=0
        return
    fi

    parse_metrics "$output"
    METRICS[WallTime]="$wall_sec"

    check_text_quality "$output"
    METRICS[QualityOK]="$QUALITY_OK"
    METRICS[QualityReason]="$QUALITY_REASON"
    METRICS[QualityPreview]="$QUALITY_PREVIEW"
    METRICS[UniqueRatio]="$QUALITY_UNIQUE_RATIO"

    if [[ $QUALITY_OK -eq 1 ]]; then
        printf "  \033[32mDone (%ss) TTFT=%sms TPOT=%sms/tok TP=%stok/s  [QUALITY: OK]\033[0m\n" \
            "${METRICS[WallTime]}" "${METRICS[TTFT]}" "${METRICS[TPOT]}" "${METRICS[Throughput]}"
    else
        printf "  \033[32mDone (%ss) TTFT=%sms TPOT=%sms/tok TP=%stok/s\033[0m\n" \
            "${METRICS[WallTime]}" "${METRICS[TTFT]}" "${METRICS[TPOT]}" "${METRICS[Throughput]}"
        echo -e "  \033[31m[QUALITY: FAIL] ${QUALITY_REASON}\033[0m"
        echo -e "  \033[33mPreview: ${QUALITY_PREVIEW}\033[0m"
    fi
}

# =============================================================================
# Helper: set quant env vars for a given precision
# =============================================================================
set_quant_env() {
    local quant="$1"
    case "$quant" in
        int4)
            export OV_GENAI_INFLIGHT_QUANT_MODE="int4_asym"
            export OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE="128"
            export OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE="int4_asym"
            export OV_GPU_USE_ONEDNN="1"
            unset OV_GPU_ONEDNN_FC_BATCH1_MAX 2>/dev/null || true
            ;;
        int4+ocl)
            export OV_GENAI_INFLIGHT_QUANT_MODE="int4_asym"
            export OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE="128"
            export OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE="int4_asym"
            export OV_GPU_USE_ONEDNN="0"
            unset OV_GPU_ONEDNN_FC_BATCH1_MAX 2>/dev/null || true
            ;;
        int8)
            export OV_GENAI_INFLIGHT_QUANT_MODE="int8_asym"
            unset OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE 2>/dev/null || true
            export OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE="int8_asym"
            unset OV_GPU_USE_ONEDNN 2>/dev/null || true
            ;;
        f16+dnn)
            unset OV_GENAI_INFLIGHT_QUANT_MODE 2>/dev/null || true
            unset OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE 2>/dev/null || true
            unset OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE 2>/dev/null || true
            unset OV_GPU_USE_ONEDNN 2>/dev/null || true
            ;;
        *)  # f16
            unset OV_GENAI_INFLIGHT_QUANT_MODE 2>/dev/null || true
            unset OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE 2>/dev/null || true
            unset OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE 2>/dev/null || true
            export OV_GPU_USE_ONEDNN="0"
            ;;
    esac
}

# =============================================================================
# Config definitions
# Each config is: "Name|MTP|MtpK|Verify|Quant|AdaptiveK"
# =============================================================================
CONFIGS=(
    # --- F16 ---
    "baseline f16|0|0|none|f16|0"
    #"baseline f16+dnn|0|0|none|f16+dnn|0"
    "K=1 batch f16|1|1|batch|f16|0"
    "K=2 batch f16|1|2|batch|f16|0"
    #"K=3 batch f16|1|3|batch|f16|0"
    # --- INT8 ---
    "baseline int8|0|0|none|int8|0"
    "K=1 batch int8|1|1|batch|int8|0"
    "K=2 batch int8|1|2|batch|int8|0"
    #"K=3 batch int8|1|3|batch|int8|0"
    # --- INT4 (oneDNN + batch-1 loop) ---
    "baseline int4|0|0|none|int4|0"
    "K=1 batch int4|1|1|batch|int4|0"
    "K=2 batch int4|1|2|batch|int4|0"
    #"K=3 batch int4|1|3|batch|int4|0"
    # --- Adaptive K ---
    #"K=2 adapt int4|1|2|batch|int4|1"
    # --- INT4+OCL ---
    #"baseline int4+ocl|0|0|none|int4+ocl|0"
    #"K=1 batch int4+ocl|1|1|batch|int4+ocl|0"
    #"K=2 batch int4+ocl|1|2|batch|int4+ocl|0"
)

SEQ_CONFIGS=(
    "K=1 seq f16|1|1|seq|f16|0"
    "K=2 seq f16|1|2|seq|f16|0"
    "K=3 seq f16|1|3|seq|f16|0"
    "K=1 seq int8|1|1|seq|int8|0"
    "K=2 seq int8|1|2|seq|int8|0"
    "K=3 seq int8|1|3|seq|int8|0"
    "K=1 seq int4|1|1|seq|int4|0"
    "K=2 seq int4|1|2|seq|int4|0"
    "K=3 seq int4|1|3|seq|int4|0"
)

if [[ $INCLUDE_SEQ -eq 1 ]]; then
    CONFIGS+=("${SEQ_CONFIGS[@]}")
    echo -e "\033[36mIncluding seq-verify configs (${#SEQ_CONFIGS[@]} additional)\033[0m"
fi

# Apply config filter
if [[ "$CONFIG_FILTER" != "*" ]]; then
    FILTERED=()
    IFS=',' read -ra PATTERNS <<< "$CONFIG_FILTER"
    for cfg in "${CONFIGS[@]}"; do
        cfg_name="${cfg%%|*}"
        for pat in "${PATTERNS[@]}"; do
            pat=$(echo "$pat" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            # Use case for reliable glob matching (handles * ? [] patterns)
            match=0
            eval "case \"\$cfg_name\" in $pat) match=1 ;; esac"
            if (( match )); then
                FILTERED+=("$cfg")
                break
            fi
        done
    done
    CONFIGS=("${FILTERED[@]}")
    echo -e "\033[36mConfig filter '${CONFIG_FILTER}' -> ${#CONFIGS[@]} config(s)\033[0m"
fi

# Build mode list
MODE_LIST=()
if [[ "$MODE" == "all" || "$MODE" == "text" ]]; then MODE_LIST+=("text"); fi
if [[ "$MODE" == "all" || "$MODE" == "vl" ]]; then MODE_LIST+=("vl"); fi

# =============================================================================
# Results storage
# Key: "modelName|mode|configName" => serialized run data
# We store results as files in a temp dir for simplicity
# =============================================================================
RESULTS_DIR=$(mktemp -d)
trap "rm -rf ${RESULTS_DIR}" EXIT

store_run_result() {
    local key="$1"
    local run_idx="$2"
    local ok="$3"
    local result_file="${RESULTS_DIR}/$(echo "$key" | tr ' /|' '___')_run${run_idx}.txt"
    if [[ "$ok" == "0" ]]; then
        echo "FAILED=1" > "$result_file"
        return
    fi
    # Save all METRICS
    for mk in "${!METRICS[@]}"; do
        echo "${mk}=${METRICS[$mk]}" >> "$result_file"
    done
}

# Helper: compute averages for a given key across all runs
# Sets AVG_* globals
declare -A AVG_METRICS
compute_averages() {
    local key="$1"
    local prefix
    prefix=$(echo "$key" | tr ' /|' '___')

    AVG_METRICS=(
        [TTFT]="N/A" [TPOT]="N/A" [Throughput]="N/A" [DecodeTime]="N/A"
        [MTPRate]="N/A" [MTPHits]="N/A" [MTPDraftAcceptRate]="N/A"
        [MTPMeanAccepted]="N/A" [OutputSize]="N/A" [MTPInfers]="N/A"
        [TokensPerInfer]="N/A" [AcceptedTokens]="N/A" [DraftTokens]="N/A"
        [VerifyAvgMs]="N/A" [DraftAvgMs]="N/A"
        [Count]="0" [QualityOK]="0" [QualityFail]="0" [QualityStatus]="N/A"
    )

    local valid_count=0
    local -a valid_files=()
    for (( ri=1; ri<=NUM_RUNS; ri++ )); do
        local rfile="${RESULTS_DIR}/${prefix}_run${ri}.txt"
        [[ ! -f "$rfile" ]] && continue
        if grep -q "FAILED=1" "$rfile" 2>/dev/null; then continue; fi
        valid_files+=("$rfile")
        ((valid_count++))
    done

    if (( valid_count == 0 )); then
        return 1
    fi
    AVG_METRICS[Count]="$valid_count"

    # Average numeric fields
    for field in TTFT TPOT Throughput DecodeTime MTPRate MTPDraftAcceptRate MTPMeanAccepted TokensPerInfer VerifyAvgMs DraftAvgMs; do
        local sum=0
        local count=0
        for rfile in "${valid_files[@]}"; do
            local val
            val=$(grep "^${field}=" "$rfile" 2>/dev/null | head -1 | cut -d= -f2-)
            if [[ -n "$val" && "$val" != "N/A" ]]; then
                sum=$(awk "BEGIN { printf \"%.4f\", $sum + $val }")
                ((count++))
            fi
        done
        if (( count > 0 )); then
            AVG_METRICS[$field]=$(awk "BEGIN { printf \"%.2f\", $sum / $count }")
        fi
    done

    # Sum-based integer fields (averaged over runs)
    for field in AcceptedTokens DraftTokens OutputSize MTPInfers; do
        local sum=0
        local count=0
        for rfile in "${valid_files[@]}"; do
            local val
            val=$(grep "^${field}=" "$rfile" 2>/dev/null | head -1 | cut -d= -f2-)
            if [[ -n "$val" && "$val" != "N/A" ]]; then
                sum=$((sum + val))
                ((count++))
            fi
        done
        if (( count > 0 )); then
            AVG_METRICS[$field]=$(awk "BEGIN { printf \"%.0f\", $sum / $count }")
        fi
    done

    # MTPHits from last valid run
    local last_file="${valid_files[-1]}"
    local last_hits
    last_hits=$(grep "^MTPHits=" "$last_file" 2>/dev/null | head -1 | cut -d= -f2-)
    [[ -n "$last_hits" && "$last_hits" != "N/A" ]] && AVG_METRICS[MTPHits]="$last_hits"

    # Quality aggregation
    local q_ok=0
    local q_fail=0
    for rfile in "${valid_files[@]}"; do
        local qval
        qval=$(grep "^QualityOK=" "$rfile" 2>/dev/null | head -1 | cut -d= -f2-)
        if [[ "$qval" == "1" ]]; then ((q_ok++)); else ((q_fail++)); fi
    done
    AVG_METRICS[QualityOK]="$q_ok"
    AVG_METRICS[QualityFail]="$q_fail"
    if (( q_fail == 0 )); then
        AVG_METRICS[QualityStatus]="OK"
    else
        AVG_METRICS[QualityStatus]="FAIL(${q_fail}/${valid_count})"
    fi

    return 0
}

# =============================================================================
# Parse config string into variables
# =============================================================================
parse_config() {
    local cfg_str="$1"
    IFS='|' read -r CFG_NAME CFG_MTP CFG_MTPK CFG_VERIFY CFG_QUANT CFG_ADAPTIVEK <<< "$cfg_str"
}

# =============================================================================
# MAIN BENCHMARK LOOP
# =============================================================================
FIRST_CONFIG=1

for model_name in "${MODEL_NAMES[@]}"; do
    MODEL_DIR="${MODEL_PATHS[$model_name]}"

    echo ""
    echo -e "\033[33m╔══════════════════════════════════════════════════════════╗\033[0m"
    echo -e "\033[33m║  Model: ${model_name}\033[0m"
    echo -e "\033[33m╚══════════════════════════════════════════════════════════╝\033[0m"

    for m in "${MODE_LIST[@]}"; do
        echo ""
        echo -e "\033[33m========================================\033[0m"
        echo -e "\033[33m ${model_name} | Mode: ${m}  |  Output tokens: ${OUTPUT_TOKENS}  |  Runs: ${NUM_RUNS}\033[0m"
        echo -e "\033[33m========================================\033[0m"

        for cfg_str in "${CONFIGS[@]}"; do
            parse_config "$cfg_str"
            local_key="${model_name}|${m}|${CFG_NAME}"

            # Set quant env vars for this config
            set_quant_env "$CFG_QUANT"

            # Build argument list
            arg_list=(
                "--model" "$MODEL_DIR"
                "--device" "$DEVICE"
                "--mode" "$m"
                "--output-tokens" "$OUTPUT_TOKENS"
                "--think" "0"
                "--temperature" "0"
            )
            if [[ "$m" == "text" ]]; then
                arg_list+=("--prompt" "Hello, please write a short story about a robot learning to paint.")
            else
                arg_list+=("--image" "$IMAGE_PATH" "--prompt" "describe this picture in details")
            fi

            if (( CFG_MTP > 0 )); then
                arg_list+=("--mtp" "1" "--mtp-k" "$CFG_MTPK")
                if (( CFG_ADAPTIVEK > 0 )); then
                    arg_list+=("--adaptive-k" "1")
                fi
                if [[ "$CFG_VERIFY" == "batch" ]]; then
                    arg_list+=("--pure-batch" "1")
                elif [[ "$CFG_VERIFY" == "seq" ]]; then
                    arg_list+=("--seq-verify" "1")
                fi
            else
                unset OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD 2>/dev/null || true
            fi

            # GPU thermal cooldown
            if (( COOLDOWN_SEC > 0 && FIRST_CONFIG == 0 )); then
                echo -e "  \033[90m[cooldown] Waiting ${COOLDOWN_SEC}s for GPU thermal stabilization...\033[0m"
                sleep "$COOLDOWN_SEC"
            fi
            FIRST_CONFIG=0

            echo ""
            echo -e "--- ${CFG_NAME} (${model_name} / ${m} mode) ---"
            for (( run_i=1; run_i<=NUM_RUNS; run_i++ )); do
                local_label="${model_name} ${CFG_NAME} run ${run_i}/${NUM_RUNS}"
                local_log_tag="${model_name}_${m}_${CFG_NAME}_run${run_i}"
                run_single_benchmark "$local_label" "$local_log_tag" "${arg_list[@]}"
                store_run_result "$local_key" "$run_i" "$RUN_OK"
            done
        done
    done
done

# =============================================================================
# SUMMARY TABLE
# =============================================================================
echo ""
echo ""
echo -e "\033[33m=================================================================\033[0m"
echo -e "\033[33m PERFORMANCE COMPARISON SUMMARY\033[0m"
echo -e "\033[33m Precisions: INT4_ASYM g128 + INT8_ASYM + F16 | Device: ${DEVICE} | Sampling: greedy (T=0)\033[0m"
echo -e "\033[33m Output tokens: ${OUTPUT_TOKENS} | Runs per config: ${NUM_RUNS}\033[0m"
echo -e "\033[33m Logs: ${LOG_DIR}\033[0m"
echo -e "\033[33m Note: OutTok = MainInfers + AcceptTok  |  DraftTok = (MainInfers-1)*K  |  Acc/Dft% = AcceptTok/DraftTok  |  Acc/Out% = AcceptTok/OutTok\033[0m"
echo -e "\033[33m       Verify(ms) = avg main model K+1 batch verify infer  |  Draft(ms) = avg single MTP head infer (called K times/step)\033[0m"
echo -e "\033[33m=================================================================\033[0m"

SEP=$(printf '%.0s-' {1..225})

for model_name in "${MODEL_NAMES[@]}"; do
    for m in "${MODE_LIST[@]}"; do
        echo ""
        echo -e "\033[36m--- ${model_name} / $(echo "$m" | tr '[:lower:]' '[:upper:]') ---\033[0m"
        echo "$SEP"
        printf "%-26s %10s %14s %16s %14s %12s %12s %12s %10s %10s %10s %8s %10s %13s %13s %10s\n" \
            "Config" "TTFT(ms)" "TPOT(ms/tok)" "Throughput(t/s)" "Decode(ms)" \
            "Acc/Dft%" "Acc/Out%" "Tok/Infer" "Avg Acc" \
            "AcceptTok" "DraftTok" "OutTok" "MainInfers" \
            "Verify(ms)" "Draft(ms)" "Quality"
        echo "$SEP"

        # Track baselines for speedup calc
        declare -A BASELINE_TPOT=()
        declare -A BASELINE_TP=()

        for cfg_str in "${CONFIGS[@]}"; do
            parse_config "$cfg_str"
            local_key="${model_name}|${m}|${CFG_NAME}"

            if ! compute_averages "$local_key"; then
                printf "%-26s %10s %14s %16s %14s %12s %12s %12s %10s %10s %10s %8s %10s %13s %13s %10s\n" \
                    "$CFG_NAME" "FAILED" "FAILED" "FAILED" "FAILED" \
                    "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A"
                continue
            fi

            # Store baselines
            if (( CFG_MTP == 0 )); then
                BASELINE_TPOT[$CFG_QUANT]="${AVG_METRICS[TPOT]}"
                BASELINE_TP[$CFG_QUANT]="${AVG_METRICS[Throughput]}"
            fi

            # Compute Acc/Dft% and Acc/Out%
            acc_dft_str="N/A"
            if [[ "${AVG_METRICS[MTPDraftAcceptRate]}" != "N/A" ]]; then
                acc_dft_str="${AVG_METRICS[MTPDraftAcceptRate]}%"
            fi
            acc_out_str="N/A"
            if [[ "${AVG_METRICS[AcceptedTokens]}" != "N/A" && "${AVG_METRICS[OutputSize]}" != "N/A" ]]; then
                out_sz="${AVG_METRICS[OutputSize]}"
                if (( out_sz > 0 )); then
                    acc_out_str=$(awk "BEGIN { printf \"%.1f%%\", ${AVG_METRICS[AcceptedTokens]} / $out_sz * 100 }")
                fi
            fi

            tpi_str="${AVG_METRICS[TokensPerInfer]}"
            ma_str="${AVG_METRICS[MTPMeanAccepted]}"
            a_tok_str="${AVG_METRICS[AcceptedTokens]}"
            d_tok_str="${AVG_METRICS[DraftTokens]}"
            o_tok_str="${AVG_METRICS[OutputSize]}"
            mi_str="${AVG_METRICS[MTPInfers]}"
            v_avg_str="${AVG_METRICS[VerifyAvgMs]}"
            d_avg_str="${AVG_METRICS[DraftAvgMs]}"
            q_str="${AVG_METRICS[QualityStatus]}"

            color=""
            reset="\033[0m"
            if [[ "${AVG_METRICS[QualityFail]}" != "0" ]]; then
                color="\033[31m"
            fi

            printf "${color}%-26s %10s %14s %16s %14s %12s %12s %12s %10s %10s %10s %8s %10s %13s %13s %10s${reset}\n" \
                "$CFG_NAME" "${AVG_METRICS[TTFT]}" "${AVG_METRICS[TPOT]}" "${AVG_METRICS[Throughput]}" "${AVG_METRICS[DecodeTime]}" \
                "$acc_dft_str" "$acc_out_str" "$tpi_str" "$ma_str" \
                "$a_tok_str" "$d_tok_str" "$o_tok_str" "$mi_str" \
                "$v_avg_str" "$d_avg_str" "$q_str"
        done
        echo "$SEP"

        # Speedup/overhead summary
        for cfg_str in "${CONFIGS[@]}"; do
            parse_config "$cfg_str"
            (( CFG_MTP == 0 )) && continue
            local_key="${model_name}|${m}|${CFG_NAME}"
            if ! compute_averages "$local_key"; then continue; fi
            base_quant="$CFG_QUANT"
            base_tpot="${BASELINE_TPOT[$base_quant]:-N/A}"
            base_tp="${BASELINE_TP[$base_quant]:-N/A}"
            if [[ "$base_tpot" == "N/A" || "${AVG_METRICS[TPOT]}" == "N/A" ]]; then continue; fi
            tpot_pct=$(awk "BEGIN { printf \"%.1f\", (${AVG_METRICS[TPOT]} - $base_tpot) / $base_tpot * 100 }")
            tp_pct="N/A"
            if [[ "$base_tp" != "N/A" && "${AVG_METRICS[Throughput]}" != "N/A" ]]; then
                tp_pct=$(awk "BEGIN { v = (${AVG_METRICS[Throughput]} - $base_tp) / $base_tp * 100; if (v >= 0) printf \"+%.1f\", v; else printf \"%.1f\", v }")
            fi
            sign=""
            [[ $(awk "BEGIN { print ($tpot_pct >= 0) ? 1 : 0 }") == "1" ]] && sign="+"
            echo -e "  \033[35m${CFG_NAME} vs baseline: TPOT ${sign}${tpot_pct}%, Throughput ${tp_pct}%\033[0m"
        done
    done
done

# =============================================================================
# PER-RUN DETAILS
# =============================================================================
echo ""
echo ""
echo -e "\033[33m--- PER-RUN DETAILS ---\033[0m"

for model_name in "${MODEL_NAMES[@]}"; do
    for m in "${MODE_LIST[@]}"; do
        for cfg_str in "${CONFIGS[@]}"; do
            parse_config "$cfg_str"
            echo ""
            echo -e "  ${CFG_NAME} (${model_name} / ${m}):"
            local_prefix=$(echo "${model_name}|${m}|${CFG_NAME}" | tr ' /|' '___')
            for (( ri=1; ri<=NUM_RUNS; ri++ )); do
                rfile="${RESULTS_DIR}/${local_prefix}_run${ri}.txt"
                if [[ ! -f "$rfile" ]] || grep -q "FAILED=1" "$rfile" 2>/dev/null; then
                    echo "    Run ${ri}: FAILED"
                    continue
                fi
                # Read metrics from file
                r_ttft=$(grep "^TTFT=" "$rfile" | head -1 | cut -d= -f2-)
                r_tpot=$(grep "^TPOT=" "$rfile" | head -1 | cut -d= -f2-)
                r_tp=$(grep "^Throughput=" "$rfile" | head -1 | cut -d= -f2-)
                r_decode=$(grep "^DecodeTime=" "$rfile" | head -1 | cut -d= -f2-)
                r_mtp_hits=$(grep "^MTPHits=" "$rfile" | head -1 | cut -d= -f2-)
                r_mtp_rate=$(grep "^MTPRate=" "$rfile" | head -1 | cut -d= -f2-)
                r_acc_rate=$(grep "^MTPDraftAcceptRate=" "$rfile" | head -1 | cut -d= -f2-)
                r_mean_acc=$(grep "^MTPMeanAccepted=" "$rfile" | head -1 | cut -d= -f2-)
                r_tpi=$(grep "^TokensPerInfer=" "$rfile" | head -1 | cut -d= -f2-)
                r_qok=$(grep "^QualityOK=" "$rfile" | head -1 | cut -d= -f2-)
                r_qreason=$(grep "^QualityReason=" "$rfile" | head -1 | cut -d= -f2-)

                detail="    Run ${ri}: TTFT=${r_ttft:-N/A}ms TPOT=${r_tpot:-N/A}ms/tok TP=${r_tp:-N/A}tok/s Decode=${r_decode:-N/A}ms"
                [[ -n "$r_mtp_hits" && "$r_mtp_hits" != "N/A" ]] && detail+=" Cond=${r_mtp_hits}(${r_mtp_rate}%)"
                [[ -n "$r_acc_rate" && "$r_acc_rate" != "N/A" ]] && detail+=" Accept=${r_acc_rate}%"
                [[ -n "$r_mean_acc" && "$r_mean_acc" != "N/A" ]] && detail+=" AvgAcc=${r_mean_acc}"
                [[ -n "$r_tpi" && "$r_tpi" != "N/A" ]] && detail+=" Tok/Infer=${r_tpi}"
                if [[ "$r_qok" == "1" ]]; then
                    detail+=" [OK]"
                else
                    detail+=" [FAIL: ${r_qreason}]"
                    echo -e "\033[31m${detail}\033[0m"
                    continue
                fi
                echo "$detail"
            done
        done
    done
done

echo ""
echo -e "\033[32mBenchmark complete. Logs saved to: ${LOG_DIR}\033[0m"
