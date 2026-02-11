#!/bin/bash
# GPU Monitor - Automatically starts experiments on idle GPUs
# Usage: nohup bash scripts/gpu_monitor.sh > results/gpu_monitor.log 2>&1 &

cd /home/lopedg/project/HyperGraph-Sparse-Attention

# Experiment queue (in order of priority)
declare -a EXPERIMENTS=(
    "isoflop_k6_deeper|K=6, 21 layers|6|21|512|1024|125000"
    "isoflop_k6_wider|K=6, dim=640|6|14|640|1024|125000"
    "isoflop_k6_longer_ctx|K=6, seq=1536|6|14|512|1536|125000"
    "extrap_baseline_global|Full attn, global RoPE|4|14|512|1024|100000|F"
    "extrap_interlaced_local|FSSFSS, local RoPE|4|14|512|1024|100000|FSSFSSFSSFSSFF"
    "extrap_interlaced_mixed|FSSFSS, mixed RoPE|4|14|512|1024|100000|FSSFSSFSSFSSFF|mixed"
    "extrap_pure_sparse_local|All sparse, local RoPE|4|14|512|1024|100000|S"
)

# Track started experiments
STARTED_FILE="/tmp/started_experiments.txt"
touch $STARTED_FILE

get_idle_gpus() {
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F',' '$2 < 10 && $3 < 500 {print $1}'
}

is_started() {
    grep -q "^$1$" $STARTED_FILE
}

mark_started() {
    echo "$1" >> $STARTED_FILE
}

start_experiment() {
    local gpu_id=$1
    local name=$2
    local desc=$3
    local k=$4
    local layers=$5
    local dim=$6
    local seq_len=$7
    local steps=$8
    local pattern=${9:-""}
    local mixed=${10:-""}
    
    local output_dir="results/ablation_auto"
    mkdir -p $output_dir
    
    echo "[$(date +%H:%M:%S)] Starting $name on GPU $gpu_id"
    
    # Build the Python command
    local extra_args=""
    if [ -n "$pattern" ]; then
        if [ "$pattern" == "F" ]; then
            extra_args="block_pattern='${"F" * $layers}',"
        elif [ "$pattern" == "S" ]; then
            extra_args="block_pattern='${"S" * $layers}',"
        else
            extra_args="block_pattern='$pattern',"
        fi
    fi
    
    if [ "$mixed" == "mixed" ]; then
        extra_args="$extra_args use_mixed_rope=True,"
    fi
    
    nohup python -c "
import sys
sys.path.insert(0, '/home/lopedg/project/HyperGraph-Sparse-Attention')
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from scripts.run_ablation import train_single_ablation, AblationConfig

config = AblationConfig(
    name='$name',
    description='$desc',
    num_hyper_nodes=$k,
    num_layers=$layers,
    dim=$dim,
    seq_len=$seq_len,
    num_steps=$steps,
    patience=5,
    dataset='pg19',
    $extra_args
)

result_queue = mp.Queue()
train_single_ablation(
    gpu_id=$gpu_id,
    config=config,
    output_dir='$output_dir',
    result_queue=result_queue,
    resume=False,
    num_workers=0,
)
" > $output_dir/${name}_main.log 2>&1 &
    
    mark_started "$name"
    echo "[$(date +%H:%M:%S)] $name started on GPU $gpu_id (PID: $!)"
}

echo "=========================================="
echo "GPU MONITOR - Auto Experiment Scheduler"
echo "=========================================="
echo "Experiments in queue: ${#EXPERIMENTS[@]}"
echo "Check interval: 5 minutes"
echo "=========================================="

while true; do
    # Get idle GPUs
    idle_gpus=$(get_idle_gpus)
    
    if [ -n "$idle_gpus" ]; then
        echo ""
        echo "[$(date +%H:%M:%S)] Found idle GPUs: $idle_gpus"
        
        for gpu in $idle_gpus; do
            # Find next experiment to start
            for exp in "${EXPERIMENTS[@]}"; do
                IFS='|' read -r name desc k layers dim seq_len steps pattern mixed <<< "$exp"
                
                if ! is_started "$name"; then
                    start_experiment $gpu "$name" "$desc" $k $layers $dim $seq_len $steps "$pattern" "$mixed"
                    sleep 30  # Wait between starts
                    break
                fi
            done
        done
    else
        echo "[$(date +%H:%M:%S)] No idle GPUs. Waiting..."
    fi
    
    # Check if all experiments started
    all_started=true
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name rest <<< "$exp"
        if ! is_started "$name"; then
            all_started=false
            break
        fi
    done
    
    if $all_started; then
        echo ""
        echo "[$(date +%H:%M:%S)] All experiments have been started!"
        echo "Monitor will continue checking for completion..."
    fi
    
    sleep 300  # Check every 5 minutes
done


