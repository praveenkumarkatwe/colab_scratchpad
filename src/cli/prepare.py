#!/usr/bin/env python3
"""Data preparation CLI for Hindi summarization datasets."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ehi.datasets import HindiSummarizationDataset, create_data_collator
from ehi.models import HindiSummarizationModel
from ehi.utils import setup_logging, load_config


def main() -> None:
    """Main entry point for data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare and preprocess Hindi summarization datasets"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["xlsum", "ilsum", "custom"],
        required=True,
        help="Dataset to prepare",
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        help="Language code (default: hi for Hindi)",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Dataset split to prepare",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/mt5-small",
        help="Model name for tokenization",
    )
    
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=1024,
        help="Maximum source text length in tokens",
    )
    
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=256,
        help="Maximum target text length in tokens",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed datasets",
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory for raw datasets",
    )
    
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of processes for parallel processing",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
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
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        
        # Override config with CLI arguments
        config.update({
            "dataset_name": args.dataset,
            "language": args.lang,
            "model_name": args.model_name,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
            "cache_dir": args.cache_dir,
            "num_proc": args.num_proc,
        })
        
        # Determine splits to process
        if args.split == "all":
            splits = ["train", "validation", "test"]
        else:
            splits = [args.split]
        
        logger.info(f"Preparing dataset: {args.dataset}")
        logger.info(f"Splits: {splits}")
        logger.info(f"Model: {args.model_name}")
        
        # Initialize dataset handler
        dataset_handler = HindiSummarizationDataset(
            dataset_name=args.dataset,
            language=args.lang,
            cache_dir=args.cache_dir,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        
        # Load model and tokenizer for preprocessing
        logger.info("Loading tokenizer...")
        model_wrapper = HindiSummarizationModel(
            model_name=args.model_name,
            use_lora=False,  # Just need tokenizer
        )
        model_wrapper.load_model_and_tokenizer()
        tokenizer = model_wrapper.get_tokenizer()
        
        # Load and preprocess dataset
        logger.info("Loading raw dataset...")
        dataset_handler.load_raw_dataset(splits)
        
        logger.info("Preprocessing and tokenizing dataset...")
        processed_datasets = dataset_handler.preprocess_dataset(
            tokenizer=tokenizer,
            splits=splits,
            num_proc=args.num_proc,
        )
        
        # Save processed datasets
        output_dir = Path(args.output_dir)
        logger.info(f"Saving processed datasets to {output_dir}")
        dataset_handler.save_processed_dataset(output_dir)
        
        # Print dataset statistics
        stats = dataset_handler.get_dataset_stats()
        logger.info("Dataset Statistics:")
        for split_name, split_stats in stats.items():
            logger.info(f"  {split_name}:")
            logger.info(f"    Examples: {split_stats['num_examples']:,}")
            logger.info(f"    Avg source length: {split_stats['avg_source_length']:.1f} tokens")
            logger.info(f"    Avg target length: {split_stats['avg_target_length']:.1f} tokens")
        
        # Create and test data collator
        logger.info("Testing data collator...")
        collate_fn = create_data_collator(tokenizer)
        
        for split_name in splits:
            if split_name in processed_datasets and len(processed_datasets[split_name]) > 0:
                # Test with a small batch
                test_batch = [processed_datasets[split_name][0]]
                try:
                    collated = collate_fn(test_batch)
                    logger.info(f"  {split_name} collation test: ✓")
                    logger.info(f"    Input shape: {collated['input_ids'].shape}")
                    logger.info(f"    Labels shape: {collated['labels'].shape}")
                except Exception as e:
                    logger.error(f"  {split_name} collation test failed: {e}")
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()