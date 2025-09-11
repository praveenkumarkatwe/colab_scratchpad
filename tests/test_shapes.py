"""Tests for model shapes and basic functionality."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from ehi.models import HindiSummarizationModel, create_model_config
from tests.fixtures import sample_config


class TestHindiSummarizationModel:
    """Test model wrapper functionality."""
    
    def test_init_default(self):
        """Test model initialization with defaults."""
        model = HindiSummarizationModel()
        
        assert model.model_name == "google/mt5-small"
        assert model.use_lora is True
        assert model.load_in_8bit is False
        assert model.model is None
        assert model.tokenizer is None
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        lora_config = {"r": 16, "lora_alpha": 64}
        
        model = HindiSummarizationModel(
            model_name="google/mt5-base",
            use_lora=False,
            lora_config=lora_config,
            load_in_8bit=True,
        )
        
        assert model.model_name == "google/mt5-base"
        assert model.use_lora is False
        assert model.load_in_8bit is True
        assert model.lora_config["r"] == 16
        assert model.lora_config["lora_alpha"] == 64
    
    def test_get_default_target_modules(self):
        """Test default target modules for different model types."""
        # Test mT5
        model = HindiSummarizationModel(model_name="google/mt5-small")
        modules = model._get_default_target_modules()
        assert "q" in modules
        assert "v" in modules
        assert "wi" in modules
        
        # Test mBART
        model = HindiSummarizationModel(model_name="facebook/mbart-large-50")
        modules = model._get_default_target_modules()
        assert "q_proj" in modules
        assert "v_proj" in modules
        assert "fc1" in modules
    
    @patch('ehi.models.AutoTokenizer.from_pretrained')
    @patch('ehi.models.AutoModelForSeq2SeqLM.from_pretrained')
    def test_load_model_and_tokenizer(self, mock_model, mock_tokenizer):
        """Test model and tokenizer loading."""
        # Mock tokenizer
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "</s>"
        mock_tokenizer.return_value = mock_tok
        
        # Mock model
        mock_mdl = Mock()
        mock_mdl.config.vocab_size = 1000
        mock_model.return_value = mock_mdl
        
        model_wrapper = HindiSummarizationModel()
        model_wrapper.load_model_and_tokenizer()
        
        # Check calls
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
        
        # Check tokenizer pad token handling
        assert mock_tok.pad_token == "</s>"
        
        # Check model assignment
        assert model_wrapper.model == mock_mdl
        assert model_wrapper.tokenizer == mock_tok
    
    @patch('ehi.models.get_peft_model')
    def test_setup_lora(self, mock_get_peft_model):
        """Test LoRA setup."""
        model_wrapper = HindiSummarizationModel(use_lora=True)
        
        # Mock base model
        mock_base_model = Mock()
        model_wrapper.model = mock_base_model
        
        # Mock LoRA model
        mock_lora_model = Mock()
        mock_get_peft_model.return_value = mock_lora_model
        
        model_wrapper.setup_lora()
        
        # Check LoRA setup
        mock_get_peft_model.assert_called_once()
        assert model_wrapper.peft_model == mock_lora_model
        assert model_wrapper.is_peft_model is True
    
    def test_setup_lora_disabled(self):
        """Test LoRA setup when disabled."""
        model_wrapper = HindiSummarizationModel(use_lora=False)
        model_wrapper.model = Mock()
        
        model_wrapper.setup_lora()
        
        # Should not set up LoRA
        assert model_wrapper.peft_model is None
        assert model_wrapper.is_peft_model is False
    
    def test_get_model_with_lora(self):
        """Test getting model with LoRA."""
        model_wrapper = HindiSummarizationModel()
        
        mock_lora_model = Mock()
        model_wrapper.peft_model = mock_lora_model
        model_wrapper.is_peft_model = True
        
        result = model_wrapper.get_model()
        assert result == mock_lora_model
    
    def test_get_model_without_lora(self):
        """Test getting model without LoRA."""
        model_wrapper = HindiSummarizationModel()
        
        mock_base_model = Mock()
        model_wrapper.model = mock_base_model
        model_wrapper.is_peft_model = False
        
        result = model_wrapper.get_model()
        assert result == mock_base_model
    
    def test_get_model_not_loaded(self):
        """Test error when model not loaded."""
        model_wrapper = HindiSummarizationModel()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            model_wrapper.get_model()
    
    def test_get_tokenizer(self):
        """Test getting tokenizer."""
        model_wrapper = HindiSummarizationModel()
        
        mock_tokenizer = Mock()
        model_wrapper.tokenizer = mock_tokenizer
        
        result = model_wrapper.get_tokenizer()
        assert result == mock_tokenizer
    
    def test_get_tokenizer_not_loaded(self):
        """Test error when tokenizer not loaded."""
        model_wrapper = HindiSummarizationModel()
        
        with pytest.raises(ValueError, match="Tokenizer not loaded"):
            model_wrapper.get_tokenizer()
    
    @patch('pathlib.Path.mkdir')
    def test_save_model_lora(self, mock_mkdir):
        """Test saving LoRA model."""
        model_wrapper = HindiSummarizationModel()
        
        # Mock tokenizer and LoRA model
        mock_tokenizer = Mock()
        mock_lora_model = Mock()
        
        model_wrapper.tokenizer = mock_tokenizer
        model_wrapper.peft_model = mock_lora_model
        model_wrapper.is_peft_model = True
        
        output_dir = "test_output"
        model_wrapper.save_model(output_dir)
        
        # Check saves
        mock_tokenizer.save_pretrained.assert_called_once()
        mock_lora_model.save_pretrained.assert_called_once()
    
    @patch('pathlib.Path.mkdir')
    def test_save_model_full(self, mock_mkdir):
        """Test saving full model."""
        model_wrapper = HindiSummarizationModel()
        
        # Mock tokenizer and base model
        mock_tokenizer = Mock()
        mock_base_model = Mock()
        
        model_wrapper.tokenizer = mock_tokenizer
        model_wrapper.model = mock_base_model
        model_wrapper.is_peft_model = False
        
        output_dir = "test_output"
        model_wrapper.save_model(output_dir)
        
        # Check saves
        mock_tokenizer.save_pretrained.assert_called_once()
        mock_base_model.save_pretrained.assert_called_once()
    
    def test_enable_training_mode(self):
        """Test enabling training mode."""
        model_wrapper = HindiSummarizationModel()
        
        # Mock model with named parameters
        mock_model = Mock()
        mock_model.named_parameters.return_value = [
            ("layer.lora_A", Mock(requires_grad=False)),
            ("layer.lora_B", Mock(requires_grad=False)),
            ("layer.weight", Mock(requires_grad=True)),
        ]
        
        model_wrapper.peft_model = mock_model
        model_wrapper.is_peft_model = True
        
        model_wrapper.enable_training_mode()
        
        mock_model.train.assert_called_once()
    
    def test_enable_eval_mode(self):
        """Test enabling evaluation mode."""
        model_wrapper = HindiSummarizationModel()
        
        mock_model = Mock()
        model_wrapper.model = mock_model
        model_wrapper.is_peft_model = False
        
        model_wrapper.enable_eval_mode()
        
        mock_model.eval.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=2**30)  # 1GB
    @patch('torch.cuda.get_device_properties')
    def test_get_memory_usage_gpu(self, mock_props, mock_allocated, mock_available):
        """Test GPU memory usage reporting."""
        mock_props.return_value.total_memory = 8 * (2**30)  # 8GB
        
        model_wrapper = HindiSummarizationModel()
        usage = model_wrapper.get_memory_usage()
        
        assert "gpu_memory_used" in usage
        assert "gpu_memory_total" in usage
        assert "gpu_memory_percent" in usage
        assert usage["gpu_memory_used"] == 1.0  # 1GB
        assert usage["gpu_memory_total"] == 8.0  # 8GB
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_memory_usage_cpu(self, mock_available):
        """Test memory usage reporting on CPU."""
        model_wrapper = HindiSummarizationModel()
        usage = model_wrapper.get_memory_usage()
        
        assert usage["gpu_memory_used"] == 0.0
        assert usage["gpu_memory_total"] == 0.0
    
    @patch('ehi.utils.count_parameters')
    def test_print_model_info(self, mock_count_params):
        """Test printing model information."""
        model_wrapper = HindiSummarizationModel()
        
        # Mock model
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_param.dtype = torch.float32
        mock_model.parameters.return_value = [mock_param]
        
        model_wrapper.model = mock_model
        model_wrapper.is_peft_model = False
        
        # Mock parameter counts
        mock_count_params.side_effect = [1000000, 100000]  # total, trainable
        
        # Should not raise errors
        model_wrapper.print_model_info()
        
        # Check parameter counting was called
        assert mock_count_params.call_count == 2


class TestModelUtilities:
    """Test model utility functions."""
    
    def test_create_model_config_default(self):
        """Test creating default model configuration."""
        config = create_model_config()
        
        assert config["model_name"] == "google/mt5-small"
        assert config["use_lora"] is True
        assert config["lora_config"]["r"] == 8
        assert config["lora_config"]["lora_alpha"] == 32
        assert config["load_in_8bit"] is False
    
    def test_create_model_config_custom(self):
        """Test creating custom model configuration."""
        config = create_model_config(
            model_name="google/mt5-base",
            use_lora=False,
            lora_r=16,
            lora_alpha=64,
            load_in_8bit=True,
            custom_param="value",
        )
        
        assert config["model_name"] == "google/mt5-base"
        assert config["use_lora"] is False
        assert config["lora_config"]["r"] == 16
        assert config["lora_config"]["lora_alpha"] == 64
        assert config["load_in_8bit"] is True
        assert config["custom_param"] == "value"


class TestModelShapes:
    """Test model shapes and tensor operations."""
    
    def test_mock_model_forward_pass(self):
        """Test mock model forward pass shapes."""
        # This is a shape test that doesn't require actual model loading
        
        batch_size = 2
        seq_length = 10
        vocab_size = 1000
        hidden_size = 512
        
        # Mock input tensors
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # Check tensor shapes
        assert input_ids.shape == (batch_size, seq_length)
        assert attention_mask.shape == (batch_size, seq_length)
        assert labels.shape == (batch_size, seq_length)
        
        # Mock model output
        mock_output = {
            "loss": torch.tensor(0.5),
            "logits": torch.randn(batch_size, seq_length, vocab_size),
        }
        
        assert mock_output["logits"].shape == (batch_size, seq_length, vocab_size)
        assert mock_output["loss"].dim() == 0  # Scalar loss
    
    def test_generation_output_shapes(self):
        """Test generation output tensor shapes."""
        batch_size = 2
        max_new_tokens = 20
        vocab_size = 1000
        
        # Mock generation outputs
        generated_sequences = torch.randint(0, vocab_size, (batch_size, max_new_tokens))
        
        assert generated_sequences.shape == (batch_size, max_new_tokens)
        assert generated_sequences.dtype == torch.long
    
    def test_lora_parameter_shapes(self):
        """Test LoRA parameter shapes."""
        # Mock LoRA parameters
        hidden_size = 512
        lora_r = 8
        
        # LoRA A and B matrices
        lora_A = torch.randn(hidden_size, lora_r)
        lora_B = torch.randn(lora_r, hidden_size)
        
        assert lora_A.shape == (hidden_size, lora_r)
        assert lora_B.shape == (lora_r, hidden_size)
        
        # Check LoRA forward pass shape
        x = torch.randn(2, 10, hidden_size)  # (batch, seq, hidden)
        lora_output = x @ lora_A @ lora_B.T
        
        assert lora_output.shape == x.shape
    
    def test_attention_mask_shapes(self):
        """Test attention mask handling."""
        batch_size = 3
        seq_lengths = [5, 8, 10]
        max_length = max(seq_lengths)
        
        # Create padded attention masks
        attention_masks = []
        for length in seq_lengths:
            mask = torch.cat([
                torch.ones(length),
                torch.zeros(max_length - length)
            ])
            attention_masks.append(mask)
        
        attention_mask = torch.stack(attention_masks)
        
        assert attention_mask.shape == (batch_size, max_length)
        assert attention_mask.sum(dim=1).tolist() == seq_lengths
    
    def test_label_mask_for_loss(self):
        """Test label masking for loss computation."""
        batch_size = 2
        seq_length = 8
        vocab_size = 1000
        pad_token_id = 0
        ignore_index = -100
        
        # Create labels with padding
        labels = torch.randint(1, vocab_size, (batch_size, seq_length))
        labels[:, -2:] = pad_token_id  # Last 2 tokens are padding
        
        # Mask padded positions
        masked_labels = labels.clone()
        masked_labels[labels == pad_token_id] = ignore_index
        
        assert masked_labels.shape == labels.shape
        assert (masked_labels[:, -2:] == ignore_index).all()
        assert (masked_labels[:, :-2] != ignore_index).all()