"""Dataset loading and preprocessing for Hindi summarization."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

from ehi.utils import ensure_dir, normalize_unicode

logger = logging.getLogger(__name__)


class HindiSummarizationDataset:
    """
    Dataset handler for Hindi summarization tasks.
    
    Supports:
    - XL-Sum Hindi dataset
    - ILSUM Hindi dataset
    - Custom JSON datasets
    """
    
    def __init__(
        self,
        dataset_name: str = "xlsum",
        language: str = "hindi",
        cache_dir: Optional[str] = None,
        max_source_length: int = 1024,
        max_target_length: int = 256,
    ) -> None:
        """
        Initialize dataset handler.
        
        Args:
            dataset_name: Name of dataset (xlsum, ilsum, or custom)
            language: Language code (hi for Hindi)
            cache_dir: Directory to cache processed datasets
            max_source_length: Maximum source text length in tokens
            max_target_length: Maximum target text length in tokens
        """
        self.dataset_name = dataset_name.lower()
        self.language = language.lower()
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        ensure_dir(self.cache_dir)
        
        # Dataset configurations
        self.dataset_configs = {
            "xlsum": {
                "hf_name": "csebuetnlp/xlsum",
                "text_column": "text",
                "summary_column": "summary",
                "lang_code": "hindi",
            },
            "ilsum": {
                "hf_name": None,  # Custom loading required
                "text_column": "article",
                "summary_column": "highlights",
                "lang_code": "hi",
            },
        }
        
        self.raw_datasets: Optional[DatasetDict] = None
        self.processed_datasets: Optional[Dict[str, Dataset]] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
    
    def load_raw_dataset(self, splits: Optional[List[str]] = None) -> DatasetDict:
        """
        Load raw dataset from Hugging Face or local files.
        
        Args:
            splits: List of splits to load (train, validation, test)
        
        Returns:
            DatasetDict with raw text data
        """
        splits = splits or ["train", "validation", "test"]
        
        if self.dataset_name == "xlsum":
            return self._load_xlsum(splits)
        elif self.dataset_name == "ilsum":
            return self._load_ilsum(splits)
        else:
            return self._load_custom_dataset(splits)
    
    def _load_xlsum(self, splits: List[str]) -> DatasetDict:
        """Load XL-Sum dataset."""
        logger.info(f"Loading XL-Sum dataset for language: {self.language}")
        
        try:
            # Load XL-Sum dataset
            dataset = load_dataset(
                "csebuetnlp/xlsum",
                self.dataset_configs["xlsum"]["lang_code"],
                cache_dir=self.cache_dir,
            )
            
            # Filter for requested splits
            filtered_dataset = {}
            for split in splits:
                if split in dataset:
                    filtered_dataset[split] = dataset[split]
                else:
                    logger.warning(f"Split '{split}' not found in XL-Sum dataset")
            
            # Normalize text
            for split_name, split_data in filtered_dataset.items():
                filtered_dataset[split_name] = split_data.map(
                    self._normalize_text_example,
                    desc=f"Normalizing {split_name} split",
                )
            
            return DatasetDict(filtered_dataset)
            
        except Exception as e:
            logger.error(f"Failed to load XL-Sum dataset: {e}")
            raise
    
    def _load_ilsum(self, splits: List[str]) -> DatasetDict:
        """Load ILSUM dataset (placeholder - requires custom implementation)."""
        logger.warning("ILSUM dataset loading not fully implemented")
        
        # This is a placeholder for ILSUM dataset loading
        # In practice, you would implement the specific loading logic for ILSUM
        
        # For now, return empty datasets
        empty_data = {"text": [], "summary": []}
        datasets = {}
        
        for split in splits:
            datasets[split] = Dataset.from_dict(empty_data)
        
        return DatasetDict(datasets)
    
    def _load_custom_dataset(self, splits: List[str]) -> DatasetDict:
        """Load custom dataset from local JSON files."""
        logger.info(f"Loading custom dataset: {self.dataset_name}")
        
        datasets = {}
        
        for split in splits:
            file_path = self.cache_dir / f"{self.dataset_name}_{split}.json"
            
            if not file_path.exists():
                logger.warning(f"Custom dataset file not found: {file_path}")
                # Create empty dataset
                datasets[split] = Dataset.from_dict({"text": [], "summary": []})
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Convert to Hugging Face dataset format
                if isinstance(data, list):
                    # List of examples
                    texts = [item.get("text", "") for item in data]
                    summaries = [item.get("summary", "") for item in data]
                elif isinstance(data, dict):
                    # Dict with text/summary keys
                    texts = data.get("text", [])
                    summaries = data.get("summary", [])
                else:
                    raise ValueError(f"Unsupported data format in {file_path}")
                
                # Create dataset
                dataset_dict = {"text": texts, "summary": summaries}
                datasets[split] = Dataset.from_dict(dataset_dict)
                
                # Normalize text
                datasets[split] = datasets[split].map(
                    self._normalize_text_example,
                    desc=f"Normalizing {split} split",
                )
                
            except Exception as e:
                logger.error(f"Failed to load custom dataset file {file_path}: {e}")
                datasets[split] = Dataset.from_dict({"text": [], "summary": []})
        
        return DatasetDict(datasets)
    
    def _normalize_text_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize text in a dataset example."""
        # Get column names based on dataset config
        config = self.dataset_configs.get(self.dataset_name, {})
        text_col = config.get("text_column", "text")
        summary_col = config.get("summary_column", "summary")
        
        # Normalize Unicode and clean text
        if text_col in example and example[text_col]:
            example[text_col] = normalize_unicode(str(example[text_col]).strip())
        
        if summary_col in example and example[summary_col]:
            example[summary_col] = normalize_unicode(str(example[summary_col]).strip())
        
        return example
    
    def preprocess_dataset(
        self,
        tokenizer: PreTrainedTokenizer,
        splits: Optional[List[str]] = None,
        num_proc: int = 4,
    ) -> Dict[str, Dataset]:
        """
        Preprocess and tokenize dataset.
        
        Args:
            tokenizer: Hugging Face tokenizer
            splits: List of splits to preprocess
            num_proc: Number of processes for parallel processing
        
        Returns:
            Dictionary of preprocessed datasets
        """
        if self.raw_datasets is None:
            self.raw_datasets = self.load_raw_dataset(splits)
        
        self.tokenizer = tokenizer
        splits = splits or list(self.raw_datasets.keys())
        
        logger.info("Preprocessing and tokenizing datasets...")
        
        processed = {}
        
        for split in splits:
            if split not in self.raw_datasets:
                logger.warning(f"Split '{split}' not found in raw datasets")
                continue
            
            # Tokenize dataset
            processed_split = self.raw_datasets[split].map(
                self._tokenize_example,
                batched=True,
                num_proc=num_proc,
                remove_columns=self.raw_datasets[split].column_names,
                desc=f"Tokenizing {split} split",
            )
            
            # Filter out examples that are too long
            processed_split = processed_split.filter(
                lambda x: (
                    len(x["input_ids"]) <= self.max_source_length
                    and len(x["labels"]) <= self.max_target_length
                ),
                desc=f"Filtering {split} split by length",
            )
            
            processed[split] = processed_split
            
            logger.info(
                f"Processed {split} split: {len(processed_split)} examples "
                f"(filtered from {len(self.raw_datasets[split])})"
            )
        
        self.processed_datasets = processed
        return processed
    
    def _tokenize_example(self, examples: Dict[str, List[str]]) -> Dict[str, List]:
        """Tokenize a batch of examples."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not set. Call preprocess_dataset first.")
        
        # Get column names
        config = self.dataset_configs.get(self.dataset_name, {})
        text_col = config.get("text_column", "text")
        summary_col = config.get("summary_column", "summary")
        
        # Extract texts and summaries
        texts = examples.get(text_col, examples.get("text", []))
        summaries = examples.get(summary_col, examples.get("summary", []))
        
        # Tokenize source texts
        model_inputs = self.tokenizer(
            texts,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,  # We'll pad dynamically in data loader
        )
        
        # Tokenize target summaries
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                summaries,
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
            )
        
        # Replace padding token id with -100 so it's ignored in loss computation
        if "input_ids" in labels:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def save_processed_dataset(self, output_dir: Union[str, Path]) -> None:
        """Save processed datasets to disk."""
        if not self.processed_datasets:
            raise ValueError("No processed datasets to save. Call preprocess_dataset first.")
        
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        for split_name, dataset in self.processed_datasets.items():
            split_path = output_dir / f"{self.dataset_name}_{split_name}_processed"
            dataset.save_to_disk(str(split_path))
            logger.info(f"Saved {split_name} split to {split_path}")
    
    def load_processed_dataset(
        self, input_dir: Union[str, Path], splits: Optional[List[str]] = None
    ) -> Dict[str, Dataset]:
        """Load processed datasets from disk."""
        input_dir = Path(input_dir)
        splits = splits or ["train", "validation", "test"]
        
        processed = {}
        
        for split in splits:
            split_path = input_dir / f"{self.dataset_name}_{split}_processed"
            
            if split_path.exists():
                try:
                    dataset = Dataset.load_from_disk(str(split_path))
                    processed[split] = dataset
                    logger.info(f"Loaded {split} split from {split_path}")
                except Exception as e:
                    logger.error(f"Failed to load {split} split from {split_path}: {e}")
            else:
                logger.warning(f"Processed dataset not found: {split_path}")
        
        self.processed_datasets = processed
        return processed
    
    def get_dataloader(
        self,
        split: str,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: Optional[callable] = None,
    ) -> DataLoader:
        """
        Get PyTorch DataLoader for a dataset split.
        
        Args:
            split: Dataset split name
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            collate_fn: Custom collate function
        
        Returns:
            PyTorch DataLoader
        """
        if not self.processed_datasets or split not in self.processed_datasets:
            raise ValueError(f"Processed dataset for split '{split}' not available")
        
        dataset = self.processed_datasets[split]
        
        # Set format for PyTorch
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        if not self.processed_datasets:
            return {}
        
        stats = {}
        
        for split_name, dataset in self.processed_datasets.items():
            stats[split_name] = {
                "num_examples": len(dataset),
                "avg_source_length": 0,
                "avg_target_length": 0,
            }
            
            if len(dataset) > 0:
                # Calculate average lengths
                source_lengths = [len(ex["input_ids"]) for ex in dataset]
                target_lengths = [
                    len([l for l in ex["labels"] if l != -100]) for ex in dataset
                ]
                
                stats[split_name]["avg_source_length"] = sum(source_lengths) / len(
                    source_lengths
                )
                stats[split_name]["avg_target_length"] = sum(target_lengths) / len(
                    target_lengths
                )
        
        return stats


def create_data_collator(tokenizer: PreTrainedTokenizer):
    """
    Create a data collator for dynamic padding.
    
    Args:
        tokenizer: Hugging Face tokenizer
    
    Returns:
        Data collator function
    """
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for dynamic padding.
        
        Args:
            batch: List of examples
        
        Returns:
            Batch dictionary with padded tensors
        """
        # Extract features
        input_ids = [torch.tensor(ex["input_ids"]) for ex in batch]
        labels = [torch.tensor(ex["labels"]) for ex in batch]
        
        # Pad input_ids and create attention_mask
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(ids) for ids in input_ids],
            batch_first=True,
            padding_value=0,
        )
        
        # Pad labels
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }
    
    return collate_fn