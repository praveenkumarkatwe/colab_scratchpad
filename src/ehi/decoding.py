"""Text generation and decoding utilities for Hindi summarization."""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerationConfig

logger = logging.getLogger(__name__)


class HindiSummarizationDecoder:
    """
    Decoder for Hindi text summarization with various generation strategies.
    
    Supports:
    - Beam search decoding
    - Nucleus (top-p) sampling
    - Temperature control
    - Repetition penalties
    - Length control
    - Batch decoding
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the decoder.
        
        Args:
            model: Pre-trained model for generation
            tokenizer: Tokenizer for encoding/decoding
            generation_config: Generation configuration parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Default generation configuration
        self.default_config = {
            "max_new_tokens": 256,
            "min_length": 10,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
            "early_stopping": True,
            "do_sample": False,
            "num_beams": 4,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
        }
        
        # Update with provided config
        if generation_config:
            self.default_config.update(generation_config)
        
        # Set pad token id if not set
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def generate(
        self,
        input_text: Union[str, List[str]],
        generation_config: Optional[Dict[str, Any]] = None,
        return_scores: bool = False,
        **kwargs,
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Generate summaries for input text(s).
        
        Args:
            input_text: Input text or list of texts
            generation_config: Generation configuration (overrides defaults)
            return_scores: Whether to return generation scores
            **kwargs: Additional generation parameters
        
        Returns:
            Generated summaries or dict with summaries and scores
        """
        # Handle single input
        if isinstance(input_text, str):
            input_text = [input_text]
        
        # Merge generation configs
        config = self.default_config.copy()
        if generation_config:
            config.update(generation_config)
        config.update(kwargs)
        
        try:
            # Tokenize inputs
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,  # Source length limit
            )
            
            # Move to model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    **config,
                    output_scores=return_scores,
                    return_dict_in_generate=return_scores,
                )
            
            # Decode outputs
            if return_scores:
                sequences = generated.sequences
                scores = generated.scores if hasattr(generated, 'scores') else None
            else:
                sequences = generated
                scores = None
            
            # Decode sequences
            summaries = self.tokenizer.batch_decode(
                sequences,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Remove input prefix for seq2seq models
            if len(summaries) == len(input_text):
                summaries = [
                    self._clean_generated_text(summary, original)
                    for summary, original in zip(summaries, input_text)
                ]
            
            if return_scores:
                return {
                    "summaries": summaries,
                    "scores": scores,
                    "sequences": sequences,
                }
            else:
                return summaries
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return empty summaries as fallback
            fallback = [""] * len(input_text)
            if return_scores:
                return {"summaries": fallback, "scores": None, "sequences": None}
            else:
                return fallback
    
    def _clean_generated_text(self, generated: str, original: str) -> str:
        """
        Clean generated text by removing input prefix and artifacts.
        
        Args:
            generated: Generated text
            original: Original input text
        
        Returns:
            Cleaned generated text
        """
        # Remove leading/trailing whitespace
        generated = generated.strip()
        
        # For seq2seq models, the output should not contain the input
        # But sometimes it does due to tokenization issues
        
        # Simple heuristic: if generated text starts with original text,
        # remove the original part
        if generated.startswith(original.strip()):
            generated = generated[len(original.strip()):].strip()
        
        # Remove any remaining special tokens that might have leaked
        special_tokens = [
            self.tokenizer.pad_token,
            self.tokenizer.eos_token,
            self.tokenizer.bos_token,
            self.tokenizer.unk_token,
        ]
        
        for token in special_tokens:
            if token:
                generated = generated.replace(token, "").strip()
        
        return generated
    
    def beam_search_decode(
        self,
        input_text: Union[str, List[str]],
        num_beams: int = 4,
        length_penalty: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate using beam search decoding.
        
        Args:
            input_text: Input text or list of texts
            num_beams: Number of beams for beam search
            length_penalty: Length penalty factor
            **kwargs: Additional generation parameters
        
        Returns:
            Generated summaries
        """
        config = {
            "do_sample": False,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "early_stopping": True,
        }
        config.update(kwargs)
        
        return self.generate(input_text, generation_config=config)
    
    def nucleus_sampling_decode(
        self,
        input_text: Union[str, List[str]],
        top_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate using nucleus (top-p) sampling.
        
        Args:
            input_text: Input text or list of texts
            top_p: Cumulative probability threshold for nucleus sampling
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
        
        Returns:
            Generated summaries
        """
        config = {
            "do_sample": True,
            "num_beams": 1,
            "top_p": top_p,
            "temperature": temperature,
        }
        config.update(kwargs)
        
        return self.generate(input_text, generation_config=config)
    
    def generate_with_scores(
        self,
        input_text: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate summaries with generation scores.
        
        Args:
            input_text: Input text or list of texts
            **kwargs: Generation parameters
        
        Returns:
            Dictionary with summaries and scores
        """
        return self.generate(input_text, return_scores=True, **kwargs)
    
    def batch_generate(
        self,
        input_texts: List[str],
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
        """
        Generate summaries in batches for efficiency.
        
        Args:
            input_texts: List of input texts
            batch_size: Batch size for generation
            **kwargs: Generation parameters
        
        Returns:
            List of generated summaries
        """
        if not input_texts:
            return []
        
        summaries = []
        
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i + batch_size]
            batch_summaries = self.generate(batch, **kwargs)
            summaries.extend(batch_summaries)
        
        return summaries
    
    def update_generation_config(self, config: Dict[str, Any]) -> None:
        """
        Update default generation configuration.
        
        Args:
            config: New configuration parameters
        """
        self.default_config.update(config)
        logger.info(f"Updated generation config: {config}")
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get current generation configuration."""
        return self.default_config.copy()


def create_generation_config(
    strategy: str = "beam_search",
    max_new_tokens: int = 256,
    min_length: int = 10,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create generation configuration for different strategies.
    
    Args:
        strategy: Generation strategy (beam_search, nucleus, diverse_beam)
        max_new_tokens: Maximum new tokens to generate
        min_length: Minimum generation length
        **kwargs: Additional parameters
    
    Returns:
        Generation configuration dictionary
    """
    base_config = {
        "max_new_tokens": max_new_tokens,
        "min_length": min_length,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        "early_stopping": True,
        "pad_token_id": None,  # Will be set by decoder
    }
    
    if strategy == "beam_search":
        strategy_config = {
            "do_sample": False,
            "num_beams": 4,
        }
    elif strategy == "nucleus":
        strategy_config = {
            "do_sample": True,
            "num_beams": 1,
            "top_p": 0.9,
            "temperature": 1.0,
        }
    elif strategy == "diverse_beam":
        strategy_config = {
            "do_sample": False,
            "num_beams": 4,
            "num_beam_groups": 2,
            "diversity_penalty": 1.0,
        }
    elif strategy == "contrastive":
        strategy_config = {
            "do_sample": False,
            "num_beams": 4,
            "penalty_alpha": 0.6,
            "top_k": 4,
        }
    else:
        raise ValueError(f"Unknown generation strategy: {strategy}")
    
    base_config.update(strategy_config)
    base_config.update(kwargs)
    
    return base_config


def estimate_generation_time(
    num_examples: int,
    avg_input_length: int,
    max_new_tokens: int,
    num_beams: int = 1,
    tokens_per_second: float = 10.0,
) -> float:
    """
    Estimate generation time for a dataset.
    
    Args:
        num_examples: Number of examples to generate
        avg_input_length: Average input length in tokens
        max_new_tokens: Maximum new tokens per example
        num_beams: Number of beams (affects computation)
        tokens_per_second: Estimated generation speed
    
    Returns:
        Estimated time in seconds
    """
    # Rough estimation based on input processing + generation
    total_tokens = num_examples * (avg_input_length + max_new_tokens) * num_beams
    return total_tokens / tokens_per_second


class SummarizationPipeline:
    """
    High-level pipeline for Hindi text summarization.
    
    Combines model loading, decoding, and post-processing.
    """
    
    def __init__(
        self,
        model_path: str,
        generation_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize summarization pipeline.
        
        Args:
            model_path: Path to model or Hugging Face model name
            generation_config: Generation configuration
            device: Device to run model on
        """
        from ehi.models import load_model_for_inference
        
        self.model_path = model_path
        self.device = device
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_model_for_inference(
            model_path, device=device
        )
        
        # Initialize decoder
        self.decoder = HindiSummarizationDecoder(
            self.model, self.tokenizer, generation_config
        )
    
    def summarize(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Summarize text(s).
        
        Args:
            text: Input text or list of texts
            **kwargs: Generation parameters
        
        Returns:
            Summary or list of summaries
        """
        is_single = isinstance(text, str)
        
        summaries = self.decoder.generate(text, **kwargs)
        
        if is_single and summaries:
            return summaries[0]
        else:
            return summaries
    
    def summarize_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
        """
        Summarize a batch of texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            **kwargs: Generation parameters
        
        Returns:
            List of summaries
        """
        return self.decoder.batch_generate(
            texts, batch_size=batch_size, **kwargs
        )