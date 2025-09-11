#!/usr/bin/env python3
"""Inference CLI for Hindi text summarization."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ehi.decoding import SummarizationPipeline
from ehi.utils import setup_logging, normalize_unicode


def load_input_texts(input_source: str) -> List[str]:
    """Load input texts from file or stdin."""
    logger = logging.getLogger(__name__)
    
    texts = []
    
    if input_source == "-":
        # Read from stdin
        logger.info("Reading from stdin...")
        try:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    texts.append(normalize_unicode(line))
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
    else:
        # Read from file
        input_path = Path(input_source)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Reading from file: {input_path}")
        
        if input_path.suffix.lower() == ".json":
            # JSON file
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # List of texts or objects
                for item in data:
                    if isinstance(item, str):
                        texts.append(normalize_unicode(item))
                    elif isinstance(item, dict):
                        # Extract text field
                        text = item.get("text", item.get("source", ""))
                        if text:
                            texts.append(normalize_unicode(text))
            elif isinstance(data, dict):
                # Single object or dict with text array
                if "text" in data or "source" in data:
                    text = data.get("text", data.get("source", ""))
                    if isinstance(text, list):
                        texts.extend([normalize_unicode(t) for t in text])
                    else:
                        texts.append(normalize_unicode(text))
        
        else:
            # Plain text file - one text per line
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(normalize_unicode(line))
    
    logger.info(f"Loaded {len(texts)} input texts")
    return texts


def save_summaries(
    summaries: List[str],
    source_texts: List[str],
    output_path: Optional[str] = None,
    format: str = "jsonl",
) -> None:
    """Save generated summaries to file or stdout."""
    logger = logging.getLogger(__name__)
    
    if output_path is None:
        # Output to stdout
        if format == "jsonl":
            for i, (source, summary) in enumerate(zip(source_texts, summaries)):
                result = {
                    "id": i,
                    "source": source,
                    "summary": summary,
                }
                print(json.dumps(result, ensure_ascii=False))
        else:
            for summary in summaries:
                print(summary)
    
    else:
        # Output to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for i, (source, summary) in enumerate(zip(source_texts, summaries)):
                    result = {
                        "id": i,
                        "source": source,
                        "summary": summary,
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        elif format == "json":
            results = []
            for i, (source, summary) in enumerate(zip(source_texts, summaries)):
                results.append({
                    "id": i,
                    "source": source,
                    "summary": summary,
                })
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for summary in summaries:
                    f.write(summary + "\n")
        
        logger.info(f"Summaries saved to {output_path}")


def main() -> None:
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Generate summaries for Hindi texts using trained models"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        default="-",
        help="Input file path (use '-' for stdin)",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file path (default: stdout)",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "json", "txt"],
        default="jsonl",
        help="Output format",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation",
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
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty for generation",
    )
    
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="No repeat n-gram size",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (cuda/cpu)",
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
        # Build generation config
        generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "min_length": args.min_length,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "early_stopping": True,
        }
        
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
        
        logger.info(f"Generation strategy: {args.generation_strategy}")
        logger.info(f"Loading model from: {args.checkpoint}")
        
        # Initialize summarization pipeline
        pipeline = SummarizationPipeline(
            model_path=args.checkpoint,
            generation_config=generation_config,
            device=args.device,
        )
        
        logger.info("Model loaded successfully")
        
        # Load input texts
        logger.info("Loading input texts...")
        input_texts = load_input_texts(args.input_file)
        
        if not input_texts:
            logger.warning("No input texts found")
            return
        
        # Generate summaries
        logger.info(f"Generating summaries for {len(input_texts)} texts...")
        
        summaries = pipeline.summarize_batch(
            input_texts,
            batch_size=args.batch_size,
        )
        
        if len(summaries) != len(input_texts):
            logger.warning(
                f"Generated {len(summaries)} summaries for {len(input_texts)} inputs"
            )
        
        # Save results
        save_summaries(
            summaries=summaries,
            source_texts=input_texts,
            output_path=args.output_file,
            format=args.format,
        )
        
        logger.info("Inference completed successfully!")
        
        # Print some statistics
        if summaries:
            avg_source_len = sum(len(text.split()) for text in input_texts) / len(input_texts)
            avg_summary_len = sum(len(summary.split()) for summary in summaries) / len(summaries)
            
            logger.info(f"Average source length: {avg_source_len:.1f} words")
            logger.info(f"Average summary length: {avg_summary_len:.1f} words")
            logger.info(f"Compression ratio: {avg_source_len / avg_summary_len:.1f}x")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()