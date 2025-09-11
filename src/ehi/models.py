"""Model management with LoRA fine-tuning support for Hindi summarization."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ehi.utils import count_parameters, get_device

logger = logging.getLogger(__name__)


class HindiSummarizationModel:
    """
    Model wrapper for Hindi summarization with LoRA fine-tuning support.
    
    Supports:
    - mT5 models (google/mt5-small, google/mt5-base)
    - mBART models (facebook/mbart-large-50-many-to-many-mmt)
    - LoRA (Low-Rank Adaptation) fine-tuning
    - Gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        model_name: str = "google/mt5-small",
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize the summarization model.
        
        Args:
            model_name: Hugging Face model name
            use_lora: Whether to use LoRA fine-tuning
            lora_config: LoRA configuration parameters
            load_in_8bit: Whether to load model in 8-bit precision
            device_map: Device mapping for model loading
            torch_dtype: PyTorch data type for model
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.torch_dtype = torch_dtype or torch.float32
        
        # Default LoRA configuration
        self.lora_config = lora_config or {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": self._get_default_target_modules(),
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": TaskType.SEQ_2_SEQ_LM,
        }
        
        self.device = get_device()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.peft_model: Optional[PeftModel] = None
        self.is_peft_model = False
    
    def _get_default_target_modules(self) -> list:
        """Get default target modules for LoRA based on model architecture."""
        if "mt5" in self.model_name.lower():
            # mT5 specific modules
            return ["q", "v", "k", "o", "wi", "wo"]
        elif "mbart" in self.model_name.lower():
            # mBART specific modules
            return ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        else:
            # Generic T5/BART-like modules
            return ["q", "v", "k", "o", "wi", "wo"]
    
    def load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer from Hugging Face."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            
            # Configure model loading arguments
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": self.device_map,
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            
            # Resize token embeddings if tokenizer was modified
            if len(self.tokenizer) > self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            
            # Prepare model for 8-bit training if needed
            if self.load_in_8bit and self.use_lora:
                self.model = prepare_model_for_int8_training(self.model)
            
            # Move to device if not using device_map
            if self.device_map is None:
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully. Parameters: {count_parameters(self.model):,}")
            
        except Exception as e:
            logger.error(f"Failed to load model and tokenizer: {e}")
            raise
    
    def setup_lora(self) -> None:
        """Set up LoRA fine-tuning."""
        if not self.use_lora or self.model is None:
            return
        
        logger.info("Setting up LoRA fine-tuning...")
        
        try:
            # Create LoRA configuration
            lora_config = LoraConfig(**self.lora_config)
            
            # Apply LoRA to model
            self.peft_model = get_peft_model(self.model, lora_config)
            self.is_peft_model = True
            
            # Print trainable parameters info
            trainable_params = count_parameters(self.peft_model, trainable_only=True)
            total_params = count_parameters(self.peft_model, trainable_only=False)
            
            logger.info(
                f"LoRA setup complete. Trainable parameters: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}% of total)"
            )
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            raise
    
    def get_model(self) -> PreTrainedModel:
        """Get the model (LoRA or base model)."""
        if self.is_peft_model and self.peft_model is not None:
            return self.peft_model
        elif self.model is not None:
            return self.model
        else:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer first.")
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer first.")
        return self.tokenizer
    
    def save_model(self, output_dir: Union[str, Path]) -> None:
        """
        Save model and tokenizer.
        
        Args:
            output_dir: Directory to save model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Tokenizer saved to {output_dir}")
        
        # Save model
        if self.is_peft_model and self.peft_model is not None:
            # Save LoRA adapters
            self.peft_model.save_pretrained(output_dir)
            logger.info(f"LoRA model saved to {output_dir}")
        elif self.model is not None:
            # Save full model
            self.model.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
        else:
            raise ValueError("No model to save")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load saved model.
        
        Args:
            model_path: Path to saved model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("Tokenizer loaded successfully")
            
            # Check if it's a LoRA model
            if (model_path / "adapter_config.json").exists():
                # Load base model first
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                )
                
                # Load LoRA adapters
                self.peft_model = PeftModel.from_pretrained(base_model, model_path)
                self.model = base_model
                self.is_peft_model = True
                
                logger.info("LoRA model loaded successfully")
            else:
                # Load full model
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                )
                self.is_peft_model = False
                
                logger.info("Full model loaded successfully")
            
            # Move to device if needed
            if self.device_map is None:
                model = self.get_model()
                model = model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def merge_and_unload_lora(self) -> PreTrainedModel:
        """
        Merge LoRA weights into base model and return unloaded model.
        
        Returns:
            Base model with merged LoRA weights
        """
        if not self.is_peft_model or self.peft_model is None:
            raise ValueError("No LoRA model to merge")
        
        logger.info("Merging LoRA weights into base model...")
        
        try:
            # Merge and unload
            merged_model = self.peft_model.merge_and_unload()
            logger.info("LoRA weights merged successfully")
            return merged_model
            
        except Exception as e:
            logger.error(f"Failed to merge LoRA weights: {e}")
            raise
    
    def enable_training_mode(self) -> None:
        """Enable training mode."""
        model = self.get_model()
        model.train()
        
        if self.is_peft_model:
            # Only LoRA parameters should require gradients
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
    def enable_eval_mode(self) -> None:
        """Enable evaluation mode."""
        model = self.get_model()
        model.eval()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage information."""
        if not torch.cuda.is_available():
            return {"gpu_memory_used": 0.0, "gpu_memory_total": 0.0}
        
        memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            "gpu_memory_used": memory_used,
            "gpu_memory_total": memory_total,
            "gpu_memory_percent": 100 * memory_used / memory_total,
        }
    
    def print_model_info(self) -> None:
        """Print model information."""
        if self.model is None:
            logger.warning("Model not loaded")
            return
        
        model = self.get_model()
        
        total_params = count_parameters(model, trainable_only=False)
        trainable_params = count_parameters(model, trainable_only=True)
        
        info = [
            f"Model: {self.model_name}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Trainable %: {100 * trainable_params / total_params:.2f}%",
            f"Model type: {'LoRA' if self.is_peft_model else 'Full'}",
            f"Device: {next(model.parameters()).device}",
            f"Data type: {next(model.parameters()).dtype}",
        ]
        
        if torch.cuda.is_available():
            memory_info = self.get_memory_usage()
            info.append(
                f"GPU memory: {memory_info['gpu_memory_used']:.2f}GB / "
                f"{memory_info['gpu_memory_total']:.2f}GB "
                f"({memory_info['gpu_memory_percent']:.1f}%)"
            )
        
        logger.info("Model Information:\n" + "\n".join(f"  {line}" for line in info))


def load_model_for_inference(
    model_path: Union[str, Path],
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Convenience function to load model and tokenizer for inference.
    
    Args:
        model_path: Path to saved model
        model_name: Base model name (for LoRA models)
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = Path(model_path)
    device = device or str(get_device())
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if it's a LoRA model
    if (model_path / "adapter_config.json").exists():
        if model_name is None:
            raise ValueError("Base model name required for LoRA models")
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load full model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
    
    model.eval()
    return model, tokenizer


def create_model_config(
    model_name: str = "google/mt5-small",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_8bit: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create model configuration dictionary.
    
    Args:
        model_name: Hugging Face model name
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        load_in_8bit: Whether to load in 8-bit
        **kwargs: Additional configuration parameters
    
    Returns:
        Model configuration dictionary
    """
    config = {
        "model_name": model_name,
        "use_lora": use_lora,
        "load_in_8bit": load_in_8bit,
        "lora_config": {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": "none",
            "task_type": TaskType.SEQ_2_SEQ_LM,
        },
    }
    
    config.update(kwargs)
    return config