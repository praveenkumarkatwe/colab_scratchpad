"""Evaluation utilities for Hindi summarization with EHI metrics."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ehi.decoding import HindiSummarizationDecoder, SummarizationPipeline
from ehi.metrics import compute_all_metrics
from ehi.ner import HindiNERPipeline
from ehi.utils import ensure_dir

logger = logging.getLogger(__name__)


class HindiSummarizationEvaluator:
    """
    Comprehensive evaluator for Hindi summarization models.
    
    Features:
    - EHI metric computation
    - Entity F1 evaluation
    - ROUGE score computation
    - Statistical analysis
    - CDF plot generation
    - Result export (JSON/CSV)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        ner_pipeline: Optional[HindiNERPipeline] = None,
        ehi_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model for generating summaries
            ner_pipeline: NER pipeline for entity extraction
            ehi_weights: EHI component weights
            device: Device for model inference
        """
        self.model_path = model_path
        self.ner_pipeline = ner_pipeline or HindiNERPipeline()
        self.ehi_weights = ehi_weights
        self.device = device
        
        # Initialize summarization pipeline if model provided
        self.pipeline = None
        if model_path:
            try:
                self.pipeline = SummarizationPipeline(
                    model_path=model_path,
                    device=device,
                )
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.pipeline = None
    
    def evaluate_dataset(
        self,
        source_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        generated_summaries: Optional[List[str]] = None,
        generate_summaries: bool = True,
        batch_size: int = 8,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a dataset of texts.
        
        Args:
            source_texts: List of source documents
            reference_texts: Optional list of reference summaries
            generated_summaries: Optional pre-generated summaries
            generate_summaries: Whether to generate summaries using model
            batch_size: Batch size for generation
            generation_config: Configuration for text generation
        
        Returns:
            Comprehensive evaluation results
        """
        if not source_texts:
            raise ValueError("Source texts cannot be empty")
        
        # Generate summaries if needed
        if generated_summaries is None and generate_summaries:
            if self.pipeline is None:
                raise ValueError("No model loaded for summary generation")
            
            logger.info(f"Generating summaries for {len(source_texts)} texts...")
            generated_summaries = self.pipeline.summarize_batch(
                source_texts,
                batch_size=batch_size,
                **(generation_config or {}),
            )
        elif generated_summaries is None:
            raise ValueError("No summaries provided and generation disabled")
        
        # Ensure lengths match
        if len(generated_summaries) != len(source_texts):
            raise ValueError("Number of summaries must match number of source texts")
        
        # Compute all metrics
        logger.info("Computing evaluation metrics...")
        metrics = compute_all_metrics(
            source_texts=source_texts,
            summary_texts=generated_summaries,
            reference_texts=reference_texts,
            ner_pipeline=self.ner_pipeline,
            ehi_weights=self.ehi_weights,
        )
        
        # Compute per-example metrics for detailed analysis
        per_example_metrics = self._compute_per_example_metrics(
            source_texts, generated_summaries, reference_texts
        )
        
        # Compile results
        results = {
            "dataset_size": len(source_texts),
            "has_references": reference_texts is not None,
            "aggregate_metrics": metrics,
            "per_example_metrics": per_example_metrics,
            "summary_statistics": self._compute_summary_statistics(per_example_metrics),
        }
        
        return results
    
    def _compute_per_example_metrics(
        self,
        source_texts: List[str],
        generated_summaries: List[str],
        reference_texts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute metrics for each example individually."""
        from ehi.metrics import EntityHallucinationIndex, EntityF1Score, compute_rouge_metrics
        
        ehi_metric = EntityHallucinationIndex(
            ner_pipeline=self.ner_pipeline,
            weights=self.ehi_weights,
        )
        entity_f1_metric = EntityF1Score(ner_pipeline=self.ner_pipeline)
        
        per_example_results = []
        
        for i, (source, summary) in enumerate(zip(source_texts, generated_summaries)):
            # Compute EHI
            ehi_result = ehi_metric.compute(source, summary)
            
            # Compute Entity F1
            ef1_result = entity_f1_metric.compute(source, summary)
            
            # Compute ROUGE if reference available
            rouge_result = {}
            if reference_texts and i < len(reference_texts):
                rouge_result = compute_rouge_metrics([summary], [reference_texts[i]])
            
            # Combine results
            example_metrics = {
                "example_id": i,
                "source_length": len(source.split()),
                "summary_length": len(summary.split()),
                "EHI": ehi_result["EHI"],
                "EF": ehi_result["EF"],
                "NH": ehi_result["NH"],
                "LF": ehi_result["LF"],
                "OF": ehi_result["OF"],
                "PH": ehi_result["PH"],
                "Entity_F1": ef1_result["f1"],
                "Entity_Precision": ef1_result["precision"],
                "Entity_Recall": ef1_result["recall"],
                **rouge_result,
            }
            
            # Add reference length if available
            if reference_texts and i < len(reference_texts):
                example_metrics["reference_length"] = len(reference_texts[i].split())
            
            per_example_results.append(example_metrics)
        
        return per_example_results
    
    def _compute_summary_statistics(
        self, per_example_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for metrics."""
        if not per_example_metrics:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for key in per_example_metrics[0].keys():
            if key in ["example_id"]:
                continue
            
            values = [m[key] for m in per_example_metrics if isinstance(m.get(key), (int, float))]
            if values:
                numeric_metrics[key] = values
        
        # Compute statistics
        statistics = {}
        for metric_name, values in numeric_metrics.items():
            values = np.array(values)
            statistics[metric_name] = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q50": float(np.percentile(values, 50)),
                "q75": float(np.percentile(values, 75)),
            }
        
        return statistics
    
    def evaluate_dataloader(
        self,
        dataloader: DataLoader,
        max_examples: Optional[int] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a PyTorch DataLoader.
        
        Args:
            dataloader: DataLoader with tokenized data
            max_examples: Maximum number of examples to evaluate
            generation_config: Configuration for text generation
        
        Returns:
            Evaluation results
        """
        if self.pipeline is None:
            raise ValueError("No model loaded for evaluation")
        
        source_texts = []
        reference_texts = []
        
        logger.info("Extracting texts from dataloader...")
        
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if max_examples and len(source_texts) >= max_examples:
                break
            
            # Decode source texts
            batch_sources = self.pipeline.tokenizer.batch_decode(
                batch["input_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            source_texts.extend(batch_sources)
            
            # Decode reference texts if available
            if "labels" in batch:
                labels = batch["labels"].clone()
                labels[labels == -100] = self.pipeline.tokenizer.pad_token_id
                batch_references = self.pipeline.tokenizer.batch_decode(
                    labels,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                reference_texts.extend(batch_references)
        
        # Limit to max_examples
        if max_examples:
            source_texts = source_texts[:max_examples]
            reference_texts = reference_texts[:max_examples] if reference_texts else None
        
        # Evaluate
        return self.evaluate_dataset(
            source_texts=source_texts,
            reference_texts=reference_texts,
            generation_config=generation_config,
        )
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        prefix: str = "evaluation",
    ) -> None:
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results dictionary
            output_dir: Output directory
            prefix: File prefix
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        # Save aggregate metrics as JSON
        aggregate_path = output_dir / f"{prefix}_aggregate.json"
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(results["aggregate_metrics"], f, indent=2, ensure_ascii=False)
        
        # Save per-example metrics as CSV
        if results["per_example_metrics"]:
            df = pd.DataFrame(results["per_example_metrics"])
            csv_path = output_dir / f"{prefix}_per_example.csv"
            df.to_csv(csv_path, index=False)
        
        # Save summary statistics as JSON
        stats_path = output_dir / f"{prefix}_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(results["summary_statistics"], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    def plot_metric_distributions(
        self,
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        metrics: Optional[List[str]] = None,
        prefix: str = "evaluation",
    ) -> None:
        """
        Plot metric distributions and CDFs.
        
        Args:
            results: Evaluation results
            output_dir: Output directory for plots
            metrics: List of metrics to plot (defaults to main metrics)
            prefix: File prefix for plot names
        """
        if not results["per_example_metrics"]:
            logger.warning("No per-example metrics to plot")
            return
        
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        df = pd.DataFrame(results["per_example_metrics"])
        
        # Default metrics to plot
        if metrics is None:
            metrics = ["EHI", "EF", "NH", "LF", "OF", "Entity_F1"]
            if "rouge1" in df.columns:
                metrics.extend(["rouge1", "rouge2", "rougeL"])
        
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            logger.warning("No available metrics to plot")
            return
        
        # Set up plot style
        plt.style.use("default")
        sns.set_palette("husl")
        
        # 1. Distribution plots (histograms)
        fig, axes = plt.subplots(
            len(available_metrics), 2,
            figsize=(12, 4 * len(available_metrics)),
        )
        
        if len(available_metrics) == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(available_metrics):
            values = df[metric].dropna()
            
            if len(values) == 0:
                continue
            
            # Histogram
            axes[i, 0].hist(values, bins=30, alpha=0.7, edgecolor="black")
            axes[i, 0].set_title(f"{metric} Distribution")
            axes[i, 0].set_xlabel(metric)
            axes[i, 0].set_ylabel("Frequency")
            axes[i, 0].grid(True, alpha=0.3)
            
            # CDF
            sorted_values = np.sort(values)
            y_cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            axes[i, 1].plot(sorted_values, y_cdf, linewidth=2)
            axes[i, 1].set_title(f"{metric} CDF")
            axes[i, 1].set_xlabel(metric)
            axes[i, 1].set_ylabel("Cumulative Probability")
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # 2. CDF comparison plot (main metrics only)
        main_metrics = ["EHI", "Entity_F1"]
        if "rouge1" in df.columns:
            main_metrics.append("rouge1")
        
        available_main = [m for m in main_metrics if m in df.columns]
        
        if available_main:
            plt.figure(figsize=(10, 6))
            
            for metric in available_main:
                values = df[metric].dropna()
                if len(values) > 0:
                    sorted_values = np.sort(values)
                    y_cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                    plt.plot(sorted_values, y_cdf, linewidth=2, label=metric)
            
            plt.xlabel("Metric Value")
            plt.ylabel("Cumulative Probability")
            plt.title("Cumulative Distribution Functions")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"{prefix}_cdf_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()
        
        # 3. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix,
                mask=mask,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
            )
            plt.title("Metric Correlations")
            plt.tight_layout()
            plt.savefig(output_dir / f"{prefix}_correlations.png", dpi=300, bbox_inches="tight")
            plt.close()
        
        logger.info(f"Plots saved to {output_dir}")
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Any]],
        output_dir: Union[str, Path],
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Compare results from multiple models.
        
        Args:
            model_results: Dict mapping model names to their results
            output_dir: Output directory
            metrics: Metrics to compare
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        if not model_results:
            logger.warning("No model results to compare")
            return
        
        # Extract aggregate metrics for comparison
        comparison_data = []
        
        for model_name, results in model_results.items():
            aggregate = results.get("aggregate_metrics", {})
            row = {"model": model_name, **aggregate}
            comparison_data.append(row)
        
        if not comparison_data:
            logger.warning("No aggregate metrics found for comparison")
            return
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        csv_path = output_dir / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        
        # Plot comparison
        metrics = metrics or ["EHI", "Entity_F1", "rouge1", "rouge2", "rougeL"]
        available_metrics = [m for m in metrics if m in df.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(
                len(available_metrics), 1,
                figsize=(10, 4 * len(available_metrics)),
            )
            
            if len(available_metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(available_metrics):
                df.plot(x="model", y=metric, kind="bar", ax=axes[i])
                axes[i].set_title(f"{metric} Comparison")
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis="x", rotation=45)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()
        
        logger.info(f"Model comparison saved to {output_dir}")


def load_and_evaluate(
    model_path: str,
    data_path: str,
    output_dir: str,
    max_examples: Optional[int] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to load data and evaluate a model.
    
    Args:
        model_path: Path to model
        data_path: Path to evaluation data (JSON file)
        output_dir: Output directory for results
        max_examples: Maximum number of examples to evaluate
        generation_config: Generation configuration
    
    Returns:
        Evaluation results
    """
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        source_texts = [item.get("text", "") for item in data]
        reference_texts = [item.get("summary", "") for item in data]
    else:
        source_texts = data.get("source", [])
        reference_texts = data.get("reference", [])
    
    # Limit examples
    if max_examples:
        source_texts = source_texts[:max_examples]
        reference_texts = reference_texts[:max_examples]
    
    # Initialize evaluator
    evaluator = HindiSummarizationEvaluator(model_path=model_path)
    
    # Evaluate
    results = evaluator.evaluate_dataset(
        source_texts=source_texts,
        reference_texts=reference_texts,
        generation_config=generation_config,
    )
    
    # Save results
    evaluator.save_results(results, output_dir)
    evaluator.plot_metric_distributions(results, output_dir)
    
    return results