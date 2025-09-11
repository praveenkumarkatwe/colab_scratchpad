"""Reinforcement Learning trainer for EHI-based fine-tuning."""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm.auto import tqdm

from ehi.decoding import HindiSummarizationDecoder
from ehi.metrics import EntityHallucinationIndex, compute_all_metrics
from ehi.ner import HindiNERPipeline
from ehi.utils import MetricsTracker, format_time, setup_logging

logger = logging.getLogger(__name__)


class EHIRLTrainer:
    """
    REINFORCE-style RL trainer for EHI-based fine-tuning.
    
    Features:
    - REINFORCE with moving baseline
    - Per-batch reward z-score normalization  
    - Gradient clipping
    - EHI-based reward computation
    - Training monitoring and logging
    - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ner_pipeline: Optional[HindiNERPipeline] = None,
        ehi_weights: Optional[Dict[str, float]] = None,
        lr: float = 5e-5,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        baseline_momentum: float = 0.9,
        reward_normalization: bool = True,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize RL trainer.
        
        Args:
            model: Pre-trained model for fine-tuning
            tokenizer: Tokenizer for text processing
            ner_pipeline: NER pipeline for entity extraction
            ehi_weights: EHI component weights
            lr: Learning rate
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            baseline_momentum: Momentum for moving baseline
            reward_normalization: Whether to normalize rewards per batch
            generation_config: Configuration for text generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.ner_pipeline = ner_pipeline or HindiNERPipeline()
        
        # Initialize EHI metric
        self.ehi_metric = EntityHallucinationIndex(
            ner_pipeline=self.ner_pipeline,
            weights=ehi_weights,
        )
        
        # Initialize decoder
        self.decoder = HindiSummarizationDecoder(
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=generation_config,
        )
        
        # Training hyperparameters
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.baseline_momentum = baseline_momentum
        self.reward_normalization = reward_normalization
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.moving_baseline = 0.0
        self.best_score = -float('inf')
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Device
        self.device = next(model.parameters()).device
        
        logger.info(f"RL trainer initialized with device: {self.device}")
    
    def compute_rewards(
        self,
        source_texts: List[str],
        generated_summaries: List[str],
        reference_summaries: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute EHI-based rewards for generated summaries.
        
        Args:
            source_texts: List of source documents
            generated_summaries: List of generated summaries
            reference_summaries: Optional reference summaries
        
        Returns:
            Tuple of (reward_tensor, metrics_dict)
        """
        rewards = []
        ehi_scores = []
        
        for source, summary in zip(source_texts, generated_summaries):
            try:
                # Compute EHI for this example
                ehi_result = self.ehi_metric.compute(source, summary)
                ehi_score = ehi_result["EHI"]
                
                # EHI is the primary reward signal
                reward = ehi_score
                
                rewards.append(reward)
                ehi_scores.append(ehi_score)
                
            except Exception as e:
                logger.warning(f"Failed to compute reward: {e}")
                rewards.append(0.0)
                ehi_scores.append(0.0)
        
        # Convert to tensor
        reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        
        # Compute additional metrics for monitoring
        metrics = {
            "reward_mean": float(reward_tensor.mean()),
            "reward_std": float(reward_tensor.std()),
            "ehi_mean": np.mean(ehi_scores),
            "ehi_std": np.std(ehi_scores),
        }
        
        return reward_tensor, metrics
    
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards using z-score normalization.
        
        Args:
            rewards: Raw reward tensor
        
        Returns:
            Normalized reward tensor
        """
        if not self.reward_normalization or len(rewards) <= 1:
            return rewards
        
        mean = rewards.mean()
        std = rewards.std()
        
        if std > 1e-8:  # Avoid division by zero
            normalized = (rewards - mean) / std
        else:
            normalized = rewards - mean
        
        return normalized
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy loss.
        
        Args:
            log_probs: Log probabilities of generated sequences
            rewards: Reward signals
            baseline: Optional baseline values
        
        Returns:
            Policy loss tensor
        """
        # Subtract baseline if provided
        if baseline is not None:
            advantages = rewards - baseline
        else:
            advantages = rewards - self.moving_baseline
        
        # Normalize advantages
        if self.reward_normalization:
            advantages = self.normalize_rewards(advantages)
        
        # REINFORCE loss: -log_prob * advantage
        loss = -log_probs * advantages
        
        return loss.mean()
    
    def generate_and_score(
        self,
        batch: Dict[str, torch.Tensor],
        source_texts: List[str],
        reference_texts: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate summaries and compute rewards.
        
        Args:
            batch: Tokenized batch data
            source_texts: Original source texts
            reference_texts: Optional reference texts
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = batch["input_ids"].size(0)
        
        # Generate summaries with sampling
        generation_config = {
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 1.0,
            "max_new_tokens": 256,
            "min_length": 10,
            "pad_token_id": self.tokenizer.pad_token_id,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        
        with torch.no_grad():
            generated_outputs = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **generation_config,
            )
        
        # Extract generated sequences and scores
        generated_sequences = generated_outputs.sequences
        scores = generated_outputs.scores
        
        # Decode generated summaries
        generated_summaries = self.tokenizer.batch_decode(
            generated_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Clean generated texts
        generated_summaries = [
            self.decoder._clean_generated_text(summary, source)
            for summary, source in zip(generated_summaries, source_texts)
        ]
        
        # Compute log probabilities
        log_probs = self._compute_log_probs(generated_sequences, scores)
        
        # Compute rewards
        rewards, reward_metrics = self.compute_rewards(
            source_texts, generated_summaries, reference_texts
        )
        
        # Update moving baseline
        current_reward_mean = rewards.mean().item()
        self.moving_baseline = (
            self.baseline_momentum * self.moving_baseline +
            (1 - self.baseline_momentum) * current_reward_mean
        )
        
        # Compute policy loss
        policy_loss = self.compute_policy_loss(log_probs, rewards)
        
        # Prepare metrics
        metrics = {
            "loss": float(policy_loss),
            "moving_baseline": self.moving_baseline,
            "generated_summaries": generated_summaries[:3],  # First 3 for logging
            "source_texts": source_texts[:3],
            **reward_metrics,
        }
        
        return policy_loss, metrics
    
    def _compute_log_probs(
        self,
        sequences: torch.Tensor,
        scores: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log probabilities of generated sequences.
        
        Args:
            sequences: Generated token sequences
            scores: Generation scores
        
        Returns:
            Log probability tensor
        """
        if not scores:
            # Fallback: compute uniform log prob
            return torch.zeros(sequences.size(0), device=self.device)
        
        batch_size = sequences.size(0)
        log_probs = torch.zeros(batch_size, device=self.device)
        
        # Sum log probabilities across generation steps
        for step_idx, step_scores in enumerate(scores):
            if step_idx + 1 >= sequences.size(1):
                break
            
            # Get tokens for this step
            tokens = sequences[:, step_idx + 1]  # +1 because scores are for next tokens
            
            # Convert scores to log probabilities
            step_log_probs = F.log_softmax(step_scores, dim=-1)
            
            # Gather log probs for selected tokens
            selected_log_probs = step_log_probs.gather(1, tokens.unsqueeze(-1)).squeeze(-1)
            
            # Add to total (only for non-padding tokens)
            mask = tokens != self.tokenizer.pad_token_id
            log_probs += selected_log_probs * mask.float()
        
        return log_probs
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        source_texts: List[str],
        reference_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a single training step.
        
        Args:
            batch: Tokenized batch data
            source_texts: Original source texts
            reference_texts: Optional reference texts
        
        Returns:
            Step metrics dictionary
        """
        self.model.train()
        
        # Forward pass: generate and score
        loss, metrics = self.generate_and_score(batch, source_texts, reference_texts)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
        
        # Optimizer step with learning rate scheduling
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.optimizer.step()
        self.global_step += 1
        
        # Add step info to metrics
        metrics.update({
            "lr": lr,
            "global_step": self.global_step,
            "epoch": self.current_epoch,
        })
        
        return metrics
    
    def _get_lr(self) -> float:
        """Get learning rate with warmup."""
        if self.global_step < self.warmup_steps:
            # Linear warmup
            return self.lr * self.global_step / self.warmup_steps
        else:
            # Constant learning rate (could add decay here)
            return self.lr
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Train the model using REINFORCE.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            num_epochs: Number of training epochs
            eval_steps: Steps between evaluations
            save_steps: Steps between checkpoints
            logging_steps: Steps between logging
            output_dir: Directory to save checkpoints
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting RL training for {num_epochs} epochs...")
        
        total_steps = len(train_dataloader) * num_epochs
        progress_bar = tqdm(total=total_steps, desc="Training")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = []
            
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get source texts (needed for reward computation)
                source_texts = self._decode_batch_inputs(batch["input_ids"])
                
                # Training step
                step_metrics = self.train_step(batch, source_texts)
                epoch_metrics.append(step_metrics)
                
                # Update metrics tracker
                self.metrics_tracker.update({
                    "loss": step_metrics["loss"],
                    "reward_mean": step_metrics["reward_mean"],
                    "ehi_mean": step_metrics["ehi_mean"],
                })
                
                # Logging
                if self.global_step % logging_steps == 0:
                    self._log_metrics(step_metrics)
                
                # Evaluation
                if eval_dataloader and self.global_step % eval_steps == 0:
                    eval_results = self.evaluate(eval_dataloader)
                    self._log_evaluation(eval_results)
                    
                    # Save best model
                    if eval_results.get("EHI", 0) > self.best_score:
                        self.best_score = eval_results["EHI"]
                        if output_dir:
                            self.save_checkpoint(output_dir / "best_model")
                
                # Save checkpoint
                if output_dir and self.global_step % save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"
                    self.save_checkpoint(checkpoint_dir)
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{step_metrics['loss']:.4f}",
                    "reward": f"{step_metrics['reward_mean']:.4f}",
                    "lr": f"{step_metrics['lr']:.2e}",
                })
        
        progress_bar.close()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {format_time(total_time)}")
        
        # Final save
        if output_dir:
            self.save_checkpoint(output_dir / "final_model")
    
    def _decode_batch_inputs(self, input_ids: torch.Tensor) -> List[str]:
        """Decode batch input IDs to text."""
        return self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log training metrics."""
        log_str = (
            f"Step {metrics['global_step']:6d} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Reward: {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f} | "
            f"EHI: {metrics['ehi_mean']:.4f} | "
            f"Baseline: {metrics['moving_baseline']:.4f} | "
            f"LR: {metrics['lr']:.2e}"
        )
        logger.info(log_str)
    
    def _log_evaluation(self, eval_results: Dict[str, float]) -> None:
        """Log evaluation results."""
        log_str = "Evaluation Results: " + " | ".join([
            f"{k}: {v:.4f}" for k, v in eval_results.items()
        ])
        logger.info(log_str)
    
    def evaluate(
        self,
        eval_dataloader: DataLoader,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            eval_dataloader: Evaluation data loader
            num_samples: Optional limit on number of samples
        
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("Running evaluation...")
        
        self.model.eval()
        
        source_texts = []
        generated_summaries = []
        reference_summaries = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                if num_samples and i * eval_dataloader.batch_size >= num_samples:
                    break
                
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Generate summaries
                batch_sources = self._decode_batch_inputs(batch["input_ids"])
                batch_summaries = self.decoder.generate(batch_sources)
                
                # Decode reference summaries if available
                if "labels" in batch:
                    # Replace -100 with pad token for decoding
                    labels = batch["labels"].clone()
                    labels[labels == -100] = self.tokenizer.pad_token_id
                    batch_references = self.tokenizer.batch_decode(
                        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    reference_summaries.extend(batch_references)
                
                source_texts.extend(batch_sources)
                generated_summaries.extend(batch_summaries)
        
        # Compute all metrics
        eval_results = compute_all_metrics(
            source_texts=source_texts,
            summary_texts=generated_summaries,
            reference_texts=reference_summaries if reference_summaries else None,
            ner_pipeline=self.ner_pipeline,
            ehi_weights=self.ehi_metric.weights,
        )
        
        self.model.train()  # Return to training mode
        
        return eval_results
    
    def save_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "moving_baseline": self.moving_baseline,
            "best_score": self.best_score,
            "optimizer_state": self.optimizer.state_dict(),
        }
        
        torch.save(state, checkpoint_dir / "training_state.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            
            self.global_step = state["global_step"]
            self.current_epoch = state["current_epoch"]
            self.moving_baseline = state["moving_baseline"]
            self.best_score = state["best_score"]
            self.optimizer.load_state_dict(state["optimizer_state"])
            
            logger.info(f"Training state loaded from {checkpoint_dir}")
        else:
            logger.warning(f"No training state found in {checkpoint_dir}")