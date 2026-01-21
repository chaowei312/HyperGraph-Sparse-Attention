"""
Main Trainer class for Sparse Attention models.
"""

import os
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .optimizer import create_optimizer, create_scheduler, clip_grad_norm_
from .metrics import MetricsTracker, compute_perplexity, EarlyStopping
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
)


class Trainer:
    """
    Trainer for causal language models.
    
    Handles:
        - Training loop with gradient accumulation
        - Mixed precision training (AMP)
        - Learning rate scheduling
        - Checkpointing
        - Logging (console + optional wandb)
        - Evaluation
    
    Example:
        ```python
        from model import CausalLM, ModelConfig
        from model.train import Trainer, TrainingConfig
        
        model_config = ModelConfig.small()
        model = CausalLM(model_config)
        
        train_config = TrainingConfig(
            learning_rate=3e-4,
            batch_size=8,
            num_epochs=10,
        )
        
        trainer = Trainer(model, train_config)
        trainer.train(train_dataloader, eval_dataloader)
        ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            train_dataloader: Training data (can be passed to train() instead)
            eval_dataloader: Evaluation data (optional)
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Setup device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Optional torch.compile
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        
        # Optimizer and scheduler (created when training starts)
        self.optimizer = None
        self.scheduler = None
        
        # Mixed precision (only on CUDA)
        use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=use_amp)
        self.autocast_dtype = config.dtype
        self.use_amp = use_amp
        
        # Metrics tracking
        self.metrics = MetricsTracker()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float("inf")
        
        # Callbacks
        self._on_step_end_callbacks = []
        self._on_epoch_end_callbacks = []
        
        # Wandb
        self.wandb_run = None
        if config.wandb_project:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "training": {
                        k: v for k, v in self.config.__dict__.items()
                        if not k.startswith("_")
                    }
                },
            )
        except ImportError:
            print("wandb not installed, skipping wandb logging")
    
    def train(
        self,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        num_training_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run training loop.
        
        Args:
            train_dataloader: Training data (uses instance dataloader if not provided)
            eval_dataloader: Evaluation data (optional)
            num_training_steps: Override total steps (else calculated from epochs)
            
        Returns:
            Dictionary with final metrics
        """
        train_dataloader = train_dataloader or self.train_dataloader
        eval_dataloader = eval_dataloader or self.eval_dataloader
        
        if train_dataloader is None:
            raise ValueError("train_dataloader must be provided")
        
        # Calculate total steps
        steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        if num_training_steps is None:
            num_training_steps = steps_per_epoch * self.config.num_epochs
        
        # Create optimizer and scheduler
        self.optimizer = create_optimizer(self.model, self.config)
        self.scheduler = create_scheduler(
            self.optimizer, self.config, num_training_steps
        )
        
        # Resume from checkpoint if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)
        elif self.config.checkpoint_dir:
            latest = get_latest_checkpoint(self.config.checkpoint_dir)
            if latest:
                print(f"Resuming from {latest}")
                self._load_checkpoint(latest)
        
        # Training loop
        self.model.train()
        self.metrics.reset()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(train_dataloader, eval_dataloader)
            
            # Epoch-end evaluation
            if eval_dataloader is not None:
                eval_loss = self._evaluate(eval_dataloader)
                is_best = eval_loss < self.best_eval_loss
                if is_best:
                    self.best_eval_loss = eval_loss
            else:
                eval_loss = None
                is_best = False
            
            # Save epoch checkpoint
            if self.config.save_every_n_epochs > 0:
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self._save_checkpoint(is_best=is_best)
            
            # Callbacks
            for callback in self._on_epoch_end_callbacks:
                callback(self, epoch, epoch_loss, eval_loss)
            
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} complete - "
                  f"train_loss: {epoch_loss:.4f}" +
                  (f", eval_loss: {eval_loss:.4f}" if eval_loss else ""))
        
        # Final save
        self._save_checkpoint(is_best=False)
        
        if self.wandb_run:
            self.wandb_run.finish()
        
        return self.metrics.get_metrics()
    
    def _train_epoch(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        accumulation_steps = self.config.gradient_accumulation_steps
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Forward pass
            loss = self._training_step(batch)
            loss = loss / accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step (after accumulation)
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = clip_grad_norm_(self.model, self.config.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # Update metrics
                current_lr = self.scheduler.get_last_lr()[0]
                num_tokens = batch["input_ids"].numel()
                
                self.metrics.update(
                    loss=loss.item() * accumulation_steps,
                    grad_norm=grad_norm,
                    lr=current_lr,
                    num_tokens=num_tokens,
                )
                
                epoch_loss += loss.item() * accumulation_steps
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    log_str = self.metrics.get_log_string(
                        self.global_step, self.current_epoch
                    )
                    print(log_str)
                    self.metrics.mark_log()
                    
                    if self.wandb_run:
                        self.wandb_run.log(
                            self.metrics.get_metrics(),
                            step=self.global_step,
                        )
                
                # Evaluation
                if (eval_dataloader is not None and 
                    self.config.eval_every_n_steps > 0 and
                    self.global_step % self.config.eval_every_n_steps == 0):
                    eval_loss = self._evaluate(eval_dataloader)
                    eval_ppl = min(float("inf"), 2.0 ** eval_loss)
                    print(f">>> EVAL | step={self.global_step} | val_loss={eval_loss:.4f} | val_ppl={eval_ppl:.2f}")
                    
                    if self.wandb_run:
                        self.wandb_run.log(
                            {"eval_loss": eval_loss},
                            step=self.global_step,
                        )
                    
                    self.model.train()
                
                # Checkpoint
                if (self.config.save_every_n_steps > 0 and
                    self.global_step % self.config.save_every_n_steps == 0):
                    self._save_checkpoint()
                
                # Callbacks
                for callback in self._on_step_end_callbacks:
                    callback(self, self.global_step, loss.item() * accumulation_steps)
        
        return epoch_loss / max(1, num_batches)
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step with optional mixed precision."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids.clone()).to(self.device)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
        
        with torch.amp.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp):
            # Forward pass
            logits = self.model(input_ids, attn_mask=attn_mask)
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return loss
    
    @torch.no_grad()
    def _evaluate(self, eval_dataloader: DataLoader) -> float:
        """Run evaluation and return loss."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids.clone()).to(self.device)
            attn_mask = batch.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.device)
            
            with torch.amp.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp):
                logits = self.model(input_ids, attn_mask=attn_mask)
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            
            num_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
        
        # Clear GPU memory after evaluation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return total_loss / max(1, total_tokens)
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            loss=self.metrics.current_loss,
            checkpoint_dir=self.config.checkpoint_dir,
            config=self.config,
            metrics=self.metrics.get_metrics(),
            keep_last_n=self.config.keep_last_n_checkpoints,
            is_best=is_best,
        )
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training state from checkpoint."""
        info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device),
        )
        self.current_epoch = info["epoch"]
        self.global_step = info["global_step"]
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def add_callback(
        self,
        on_step_end: Optional[Callable] = None,
        on_epoch_end: Optional[Callable] = None,
    ):
        """
        Add training callbacks.
        
        Args:
            on_step_end: Called after each optimization step.
                         Signature: (trainer, step, loss)
            on_epoch_end: Called after each epoch.
                          Signature: (trainer, epoch, train_loss, eval_loss)
        """
        if on_step_end is not None:
            self._on_step_end_callbacks.append(on_step_end)
        if on_epoch_end is not None:
            self._on_epoch_end_callbacks.append(on_epoch_end)

