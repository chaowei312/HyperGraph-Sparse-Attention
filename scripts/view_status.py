#!/usr/bin/env python3
"""
Ablation Experiment Status Viewer

Compares experiments at equal steps, grouped by training setting.
Shows validation loss, training loss, and estimated time remaining.
"""

import re
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

@dataclass
class ExperimentStatus:
    name: str
    current_step: int
    target_steps: int
    train_loss: float
    best_val_loss: float
    last_val_loss: float
    time_elapsed: float  # seconds
    params: int
    pattern: str
    seq_len: int
    group: str  # "isoflop" or "hybrid_extrap"
    
    # History for comparison at equal steps
    train_loss_history: Dict[int, float] = None  # step -> loss
    val_loss_history: Dict[int, float] = None  # step -> loss
    
    def __post_init__(self):
        if self.train_loss_history is None:
            self.train_loss_history = {}
        if self.val_loss_history is None:
            self.val_loss_history = {}
    
    @property
    def progress(self) -> float:
        return self.current_step / self.target_steps * 100
    
    @property
    def eta_hours(self) -> float:
        if self.current_step == 0:
            return float('inf')
        time_per_step = self.time_elapsed / self.current_step
        remaining_steps = self.target_steps - self.current_step
        return remaining_steps * time_per_step / 3600


def parse_log_file(log_path: Path, target_steps: int, group: str) -> Optional[ExperimentStatus]:
    """Parse a single experiment log file."""
    if not log_path.exists():
        return None
    
    name = log_path.stem.rsplit('_gpu', 1)[0]
    
    current_step = 0
    train_loss = 0.0
    best_val_loss = float('inf')
    last_val_loss = float('inf')
    time_elapsed = 0.0
    params = 0
    pattern = ""
    seq_len = 1024
    
    train_loss_history = {}
    val_loss_history = {}
    
    # Track the most recent training step to associate with validation
    last_train_step = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse params
            if 'params' in line and 'warmup' in line:
                m = re.search(r'([\d,]+) params', line)
                if m:
                    params = int(m.group(1).replace(',', ''))
            
            # Parse pattern
            if 'pattern=' in line:
                m = re.search(r'pattern=(\w+)', line)
                if m:
                    pattern = m.group(1)
            
            # Parse seq_len
            if 'seq_len=' in line:
                m = re.search(r'seq_len=(\d+)', line)
                if m:
                    seq_len = int(m.group(1))
            
            # Parse training step
            m = re.search(r'step (\d+) \| loss: ([\d.]+) .* time: ([\d.]+)s', line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                time_s = float(m.group(3))
                
                # Track the most recent step for val association
                last_train_step = step
                
                if step > current_step:
                    current_step = step
                    train_loss = loss
                    time_elapsed = time_s
                
                # Record history at 500-step intervals
                if step % 500 == 0:
                    train_loss_history[step] = loss
            
            # Parse validation loss - associate with most recent training step
            m = re.search(r'val_loss: ([\d.]+)', line)
            if m:
                val = float(m.group(1))
                last_val_loss = val
                if val < best_val_loss:
                    best_val_loss = val
                
                # Associate with the most recent training step
                # Validation happens every 2000 steps, so round to nearest 2000
                val_step = (last_train_step // 2000) * 2000
                if val_step == 0:
                    val_step = 2000  # First validation at step 2000
                # Only record FIRST val_loss for each step (avoid orphan entries)
                if val_step not in val_loss_history:
                    val_loss_history[val_step] = val
    
    if current_step == 0:
        return None
    
    return ExperimentStatus(
        name=name,
        current_step=current_step,
        target_steps=target_steps,
        train_loss=train_loss,
        best_val_loss=best_val_loss,
        last_val_loss=last_val_loss,
        time_elapsed=time_elapsed,
        params=params,
        pattern=pattern,
        seq_len=seq_len,
        group=group,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
    )


def find_logs(base_dir: Path) -> List[Tuple[Path, int, str]]:
    """Find all experiment logs with their target steps and group."""
    logs = []
    
    # IsoFLOP experiments (124K or 83K steps, 2.03B tokens)
    isoflop_dir = base_dir / "results" / "ablation_isoflop_isotokens"
    if isoflop_dir.exists():
        for log in isoflop_dir.glob("*_gpu*.log"):
            if "longer_ctx" in log.name:
                target = 83000
            else:
                target = 124000
            logs.append((log, target, "isoflop"))
    
    # Hybrid/Extrap experiments (85K steps)
    hybrid_dir = base_dir / "results" / "ablation_phase2_cosine"
    if hybrid_dir.exists():
        for log in hybrid_dir.glob("*_gpu*.log"):
            # Skip old isoflop logs in this dir
            if "isoflop" in log.name:
                continue
            logs.append((log, 85000, "hybrid_extrap"))
    
    return logs


def print_separator(char='=', width=100):
    print(char * width)


def print_header(title: str, width=100):
    print()
    print_separator('=', width)
    print(f" {title}".center(width))
    print_separator('=', width)


def format_time(hours: float) -> str:
    if hours == float('inf'):
        return "N/A"
    if hours < 1:
        return f"{hours * 60:.0f}m"
    return f"{hours:.1f}h"


def main():
    base_dir = Path("/home/lopedg/project/HyperGraph-Sparse-Attention")
    
    # Find and parse all logs
    logs = find_logs(base_dir)
    experiments = []
    
    for log_path, target_steps, group in logs:
        exp = parse_log_file(log_path, target_steps, group)
        if exp:
            experiments.append(exp)
    
    # Deduplicate (keep most recent per experiment name)
    exp_dict = {}
    for exp in experiments:
        if exp.name not in exp_dict or exp.current_step > exp_dict[exp.name].current_step:
            exp_dict[exp.name] = exp
    
    experiments = list(exp_dict.values())
    
    # Group experiments
    isoflop_exps = [e for e in experiments if e.group == "isoflop"]
    hybrid_exps = [e for e in experiments if e.group == "hybrid_extrap"]
    
    # === ISOFLOP EXPERIMENTS ===
    if isoflop_exps:
        print_header("ISOFLOP EXPERIMENTS (Iso-Tokens: 2.03B)")
        print(f"{'Experiment':<25} {'Params':>10} {'Step':>8} {'Target':>8} {'Progress':>8} {'Train':>8} {'Val':>8} {'Best Val':>10} {'ETA':>8}")
        print_separator('-')
        
        for exp in sorted(isoflop_exps, key=lambda x: x.best_val_loss):
            print(f"{exp.name:<25} {exp.params/1e6:>9.1f}M {exp.current_step:>8,} {exp.target_steps:>8,} "
                  f"{exp.progress:>7.1f}% {exp.train_loss:>8.4f} {exp.last_val_loss:>8.4f} {exp.best_val_loss:>10.4f} {format_time(exp.eta_hours):>8}")
        
        # Find slowest model step
        min_step = min(exp.current_step for exp in isoflop_exps)
        # Round down to nearest 500
        compare_step = (min_step // 500) * 500
        
        print()
        print(f"  >>> COMPARISON AT {compare_step:,} STEPS (slowest model checkpoint)")
        print(f"  {'Experiment':<25} {'Train Loss':>12} {'Val Loss':>12}")
        print(f"  {'-'*50}")
        
        comparisons = []
        for exp in isoflop_exps:
            train_at_step = exp.train_loss_history.get(compare_step, None)
            # Find closest val step
            val_steps = sorted(exp.val_loss_history.keys())
            val_at_step = None
            for vs in val_steps:
                if vs <= compare_step + 1000:
                    val_at_step = exp.val_loss_history[vs]
            comparisons.append((exp.name, train_at_step, val_at_step))
        
        # Sort by val loss (if available) or train loss
        comparisons.sort(key=lambda x: x[2] if x[2] else x[1] if x[1] else 999)
        
        for name, train, val in comparisons:
            train_str = f"{train:.4f}" if train else "N/A"
            val_str = f"{val:.4f}" if val else "N/A"
            marker = " ★" if comparisons[0][0] == name else ""
            print(f"  {name:<25} {train_str:>12} {val_str:>12}{marker}")
    
    # === HYBRID/EXTRAP EXPERIMENTS ===
    if hybrid_exps:
        print_header("HYBRID & EXTRAPOLATION EXPERIMENTS (85K steps)")
        print(f"{'Experiment':<30} {'Pattern':>16} {'Step':>8} {'Progress':>8} {'Train':>8} {'Val':>8} {'Best Val':>10} {'ETA':>8}")
        print_separator('-')
        
        for exp in sorted(hybrid_exps, key=lambda x: x.best_val_loss):
            pattern_short = exp.pattern[:14] + ".." if len(exp.pattern) > 16 else exp.pattern
            print(f"{exp.name:<30} {pattern_short:>16} {exp.current_step:>8,} "
                  f"{exp.progress:>7.1f}% {exp.train_loss:>8.4f} {exp.last_val_loss:>8.4f} {exp.best_val_loss:>10.4f} {format_time(exp.eta_hours):>8}")
        
        # Find slowest model step
        min_step = min(exp.current_step for exp in hybrid_exps)
        compare_step = (min_step // 500) * 500
        
        print()
        print(f"  >>> COMPARISON AT {compare_step:,} STEPS (slowest model checkpoint)")
        print(f"  {'Experiment':<30} {'Train Loss':>12} {'Val Loss':>12}")
        print(f"  {'-'*55}")
        
        comparisons = []
        for exp in hybrid_exps:
            train_at_step = exp.train_loss_history.get(compare_step, None)
            # Find closest val step
            val_steps = sorted(exp.val_loss_history.keys())
            val_at_step = None
            for vs in val_steps:
                if vs <= compare_step + 1000:
                    val_at_step = exp.val_loss_history[vs]
            comparisons.append((exp.name, train_at_step, val_at_step))
        
        comparisons.sort(key=lambda x: x[2] if x[2] else x[1] if x[1] else 999)
        
        for name, train, val in comparisons:
            train_str = f"{train:.4f}" if train else "N/A"
            val_str = f"{val:.4f}" if val else "N/A"
            marker = " ★" if comparisons[0][0] == name else ""
            print(f"  {name:<30} {train_str:>12} {val_str:>12}{marker}")
    
    # === SUMMARY ===
    print_header("SUMMARY")
    
    all_exps = isoflop_exps + hybrid_exps
    if all_exps:
        total_running = len(all_exps)
        avg_progress = sum(e.progress for e in all_exps) / len(all_exps)
        max_eta = max(e.eta_hours for e in all_exps if e.eta_hours != float('inf'))
        
        print(f"  Total experiments running: {total_running}")
        print(f"  Average progress: {avg_progress:.1f}%")
        print(f"  Longest remaining: {format_time(max_eta)}")
        
        print()
        print("  Best performers by group:")
        if isoflop_exps:
            best_iso = min(isoflop_exps, key=lambda x: x.best_val_loss)
            print(f"    IsoFLOP:      {best_iso.name} (val: {best_iso.best_val_loss:.4f})")
        if hybrid_exps:
            best_hyb = min(hybrid_exps, key=lambda x: x.best_val_loss)
            print(f"    Hybrid/Extrap: {best_hyb.name} (val: {best_hyb.best_val_loss:.4f})")
    
    print()
    print(f"  Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    main()

