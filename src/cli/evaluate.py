#!/usr/bin/env python3
"""Evaluation CLI for Hindi summarization models."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ehi.eval import HindiSummarizationEvaluator, load_and_evaluate
from ehi.ner import HindiNERPipeline
from ehi.utils import setup_logging, load_config


def load_evaluation_data(data_path: str) -> tuple[list[str], Optional[list[str]]]:
    """Load evaluation data from file."""
    logger = logging.getLogger(__name__)
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading evaluation data from {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    source_texts = []
    reference_texts = []
    
    if isinstance(data, list):
        # List of examples with text/summary fields
        for item in data:
            if isinstance(item, dict):
                source_texts.append(item.get("text", ""))
                if "summary" in item:
                    reference_texts.append(item["summary"])
    elif isinstance(data, dict):
        # Dict with source/reference arrays
        source_texts = data.get("source", data.get("text", []))
        reference_texts = data.get("reference", data.get("summary", []))
    else:
        raise ValueError(f"Unsupported data format in {data_path}")
    
    # Ensure we have source texts
    if not source_texts:
        raise ValueError("No source texts found in data file")
    
    # Use references only if we have them for all examples
    if reference_texts and len(reference_texts) != len(source_texts):
        logger.warning(
            f"Reference count ({len(reference_texts)}) doesn't match "
            f"source count ({len(source_texts)}), ignoring references"
        )
        reference_texts = None
    
    logger.info(f"Loaded {len(source_texts)} source texts")
    if reference_texts:
        logger.info(f"Loaded {len(reference_texts)} reference summaries")
    
    return source_texts, reference_texts


def main() -> None:
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Hindi summarization models with EHI metrics"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to evaluation configuration file",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="Path to evaluation data (JSON file)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to evaluate",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation",
    )
    
    parser.add_argument(
        "--plot-cdfs",
        action="store_true",
        help="Generate CDF plots",
    )
    
    parser.add_argument(
        "--generation-strategy",
        type=str,
        choices=["beam_search", "nucleus", "diverse_beam"],
        default="beam_search",
        help="Generation strategy",
    )
    
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search",
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p value for nucleus sampling",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum generation length",
    )
    
    parser.add_argument(
        "--compare-models",
        type=str,
        nargs="+",
        help="Paths to multiple models for comparison",
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
        
        # Build generation config from CLI args and config
        generation_config = config.get("generation", {})
        
        if args.generation_strategy == "beam_search":
            generation_config.update({
                "do_sample": False,
                "num_beams": args.num_beams,
            })
        elif args.generation_strategy == "nucleus":
            generation_config.update({
                "do_sample": True,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "num_beams": 1,
            })
        elif args.generation_strategy == "diverse_beam":
            generation_config.update({
                "do_sample": False,
                "num_beams": args.num_beams,
                "num_beam_groups": 2,
                "diversity_penalty": 1.0,
            })
        
        generation_config.update({
            "max_new_tokens": args.max_new_tokens,
            "min_length": args.min_length,
            "length_penalty": generation_config.get("length_penalty", 1.0),
            "no_repeat_ngram_size": generation_config.get("no_repeat_ngram_size", 3),
            "early_stopping": True,
        })
        
        logger.info(f"Generation strategy: {args.generation_strategy}")
        logger.info(f"Generation config: {generation_config}")
        
        # Prepare output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NER pipeline
        logger.info("Initializing NER pipeline...")
        ner_pipeline = HindiNERPipeline()
        if not ner_pipeline.is_available():
            logger.warning("No NER models available - EHI computation may be limited")
        
        # Get EHI weights from config
        ehi_weights = config.get("ehi_weights")
        
        # Model comparison mode
        if args.compare_models:
            logger.info(f"Comparing {len(args.compare_models)} models")
            
            if not args.data:
                raise ValueError("Data file required for model comparison")
            
            # Load data
            source_texts, reference_texts = load_evaluation_data(args.data)
            
            # Limit examples
            if args.max_examples:
                source_texts = source_texts[:args.max_examples]
                if reference_texts:
                    reference_texts = reference_texts[:args.max_examples]
            
            # Evaluate each model
            model_results = {}
            
            for i, model_path in enumerate(args.compare_models):
                model_name = f"model_{i+1}"
                logger.info(f"Evaluating {model_name}: {model_path}")
                
                try:
                    evaluator = HindiSummarizationEvaluator(
                        model_path=model_path,
                        ner_pipeline=ner_pipeline,
                        ehi_weights=ehi_weights,
                    )
                    
                    results = evaluator.evaluate_dataset(
                        source_texts=source_texts,
                        reference_texts=reference_texts,
                        batch_size=args.batch_size,
                        generation_config=generation_config,
                    )
                    
                    model_results[model_name] = results
                    
                    # Save individual results
                    model_output_dir = output_dir / model_name
                    evaluator.save_results(results, model_output_dir)
                    
                    if args.plot_cdfs:
                        evaluator.plot_metric_distributions(results, model_output_dir)
                    
                    logger.info(f"Results for {model_name}:")
                    for metric, value in results["aggregate_metrics"].items():
                        logger.info(f"  {metric}: {value:.4f}")
                
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name}: {e}")
                    continue
            
            # Generate comparison
            if len(model_results) > 1:
                evaluator = HindiSummarizationEvaluator(ner_pipeline=ner_pipeline)
                evaluator.compare_models(model_results, output_dir)
                logger.info(f"Model comparison saved to {output_dir}")
        
        # Single model evaluation
        else:
            logger.info(f"Evaluating model: {args.checkpoint}")
            
            # Use convenience function if data file provided
            if args.data:
                results = load_and_evaluate(
                    model_path=args.checkpoint,
                    data_path=args.data,
                    output_dir=str(output_dir),
                    max_examples=args.max_examples,
                    generation_config=generation_config,
                )
            else:
                # Initialize evaluator without data
                evaluator = HindiSummarizationEvaluator(
                    model_path=args.checkpoint,
                    ner_pipeline=ner_pipeline,
                    ehi_weights=ehi_weights,
                )
                
                logger.warning("No data file provided - only model loaded")
                results = {"aggregate_metrics": {}, "per_example_metrics": []}
            
            # Generate plots if requested
            if args.plot_cdfs and results["per_example_metrics"]:
                evaluator = HindiSummarizationEvaluator(
                    model_path=args.checkpoint,
                    ner_pipeline=ner_pipeline,
                    ehi_weights=ehi_weights,
                )
                evaluator.plot_metric_distributions(results, output_dir)
            
            # Print aggregate results
            if results["aggregate_metrics"]:
                logger.info("Evaluation Results:")
                for metric, value in results["aggregate_metrics"].items():
                    logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"Evaluation completed! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()