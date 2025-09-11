#!/usr/bin/env python3
"""Training CLI for EHI-based fine-tuning."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ehi.datasets import HindiSummarizationDataset, create_data_collator
from ehi.models import HindiSummarizationModel
from ehi.ner import HindiNERPipeline
from ehi.rl_trainer import EHIRLTrainer
from ehi.utils import setup_logging, load_config, set_seed


def load_dataset_for_training(
    config: Dict[str, Any],
    tokenizer,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Load and prepare datasets for training."""
    logger = logging.getLogger(__name__)
    
    # Initialize dataset handler
    dataset_handler = HindiSummarizationDataset(
        dataset_name=config.get("dataset_name", "xlsum"),
        language=config.get("language", "hi"),
        cache_dir=config.get("cache_dir", "data/cache"),
        max_source_length=config.get("max_source_length", 1024),
        max_target_length=config.get("max_target_length", 256),
    )
    
    # Load processed datasets
    processed_dir = config.get("processed_data_dir", "data/processed")
    if Path(processed_dir).exists():
        logger.info(f"Loading processed datasets from {processed_dir}")
        datasets = dataset_handler.load_processed_dataset(processed_dir)
    else:
        logger.info("Processed datasets not found, creating from scratch...")
        # Load and preprocess
        dataset_handler.load_raw_dataset()
        datasets = dataset_handler.preprocess_dataset(
            tokenizer=tokenizer,
            num_proc=config.get("num_proc", 4),
        )
        # Save for future use
        dataset_handler.save_processed_dataset(processed_dir)
    
    # Create data collator
    collate_fn = create_data_collator(tokenizer)
    
    # Create data loaders
    train_loader = None
    eval_loader = None
    
    if "train" in datasets:
        train_loader = DataLoader(
            datasets["train"],
            batch_size=config.get("train_batch_size", 4),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.get("num_workers", 0),
        )
        logger.info(f"Train loader: {len(train_loader)} batches")
    
    if "validation" in datasets:
        eval_loader = DataLoader(
            datasets["validation"],
            batch_size=config.get("eval_batch_size", 8),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get("num_workers", 0),
        )
        logger.info(f"Eval loader: {len(eval_loader)} batches")
    
    return train_loader, eval_loader


def main() -> None:
    """Main entry point for EHI training."""
    parser = argparse.ArgumentParser(
        description="Train Hindi summarization model with EHI-based RL fine-tuning"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for checkpoints (overrides config)",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, no training",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Override with CLI arguments
        if args.output_dir:
            config["output_dir"] = args.output_dir
        
        # Set random seed
        seed = config.get("seed", 42)
        set_seed(seed)
        logger.info(f"Set random seed to {seed}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("GPU not available, using CPU")
        
        # Initialize model
        logger.info("Initializing model...")
        model_config = config.get("model", {})
        model_wrapper = HindiSummarizationModel(
            model_name=model_config.get("name", "google/mt5-small"),
            use_lora=model_config.get("use_lora", True),
            lora_config=model_config.get("lora", {}),
            load_in_8bit=model_config.get("load_in_8bit", False),
        )
        
        # Load model and tokenizer
        model_wrapper.load_model_and_tokenizer()
        
        # Setup LoRA if enabled
        if model_config.get("use_lora", True):
            model_wrapper.setup_lora()
        
        model = model_wrapper.get_model()
        tokenizer = model_wrapper.get_tokenizer()
        
        # Print model info
        model_wrapper.print_model_info()
        
        # Initialize NER pipeline
        logger.info("Initializing NER pipeline...")
        ner_pipeline = HindiNERPipeline()
        if not ner_pipeline.is_available():
            logger.warning("No NER models available - EHI computation may be limited")
        else:
            logger.info(f"Available NER models: {ner_pipeline.get_available_models()}")
        
        # Load datasets
        logger.info("Loading datasets...")
        train_loader, eval_loader = load_dataset_for_training(config, tokenizer)
        
        if train_loader is None and not args.eval_only:
            raise ValueError("No training data available")
        
        # Initialize RL trainer
        logger.info("Initializing RL trainer...")
        training_config = config.get("training", {})
        
        trainer = EHIRLTrainer(
            model=model,
            tokenizer=tokenizer,
            ner_pipeline=ner_pipeline,
            ehi_weights=config.get("ehi_weights"),
            lr=training_config.get("lr", 5e-5),
            warmup_steps=training_config.get("warmup_steps", 100),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),
            baseline_momentum=training_config.get("baseline_momentum", 0.9),
            reward_normalization=training_config.get("reward_normalization", True),
            generation_config=config.get("generation", {}),
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Run evaluation if requested
        if args.eval_only or eval_loader is not None:
            logger.info("Running evaluation...")
            eval_results = trainer.evaluate(
                eval_loader,
                num_samples=config.get("eval_samples"),
            )
            
            logger.info("Evaluation Results:")
            for metric, value in eval_results.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            if args.eval_only:
                return
        
        # Training
        if train_loader is not None:
            logger.info("Starting training...")
            
            trainer.train(
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                num_epochs=training_config.get("num_epochs", 3),
                eval_steps=training_config.get("eval_steps", 500),
                save_steps=training_config.get("save_steps", 1000),
                logging_steps=training_config.get("logging_steps", 100),
                output_dir=config.get("output_dir", "outputs"),
            )
            
            logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()