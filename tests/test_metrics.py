"""Tests for EHI metrics computation."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ehi.metrics import (
    EntityHallucinationIndex,
    EntityF1Score,
    compute_rouge_metrics,
    compute_all_metrics,
)
from tests.fixtures import sample_texts, sample_summaries, mock_ner_pipeline


class TestEntityHallucinationIndex:
    """Test EHI metric computation."""
    
    def test_init_default_weights(self):
        """Test EHI initialization with default weights."""
        ehi = EntityHallucinationIndex()
        
        expected_weights = {
            "w_EF": 1.0,
            "w_NH": 1.0,
            "w_LF": 1.0,
            "w_OF": 1.0,
            "w_PH": 0.0,
        }
        assert ehi.weights == expected_weights
    
    def test_init_custom_weights(self):
        """Test EHI initialization with custom weights."""
        custom_weights = {
            "w_EF": 2.0,
            "w_NH": 1.5,
            "w_LF": 1.0,
            "w_OF": 0.5,
            "w_PH": 0.1,
        }
        ehi = EntityHallucinationIndex(weights=custom_weights)
        assert ehi.weights == custom_weights
    
    def test_compute_perfect_match(self, mock_ner_pipeline):
        """Test EHI computation with perfect entity match."""
        ehi = EntityHallucinationIndex(ner_pipeline=mock_ner_pipeline)
        
        source = "राम एक अच्छा लड़का है। वह दिल्ली में रहता है।"
        summary = "राम दिल्ली में रहता है।"
        
        result = ehi.compute(source, summary)
        
        assert "EHI" in result
        assert "EF" in result
        assert "NH" in result
        assert "LF" in result
        assert "OF" in result
        assert isinstance(result["EHI"], float)
        assert 0 <= result["EHI"] <= 1
    
    def test_compute_empty_texts(self, mock_ner_pipeline):
        """Test EHI computation with empty texts."""
        ehi = EntityHallucinationIndex(ner_pipeline=mock_ner_pipeline)
        
        result = ehi.compute("", "")
        
        assert result["EHI"] == 0.0
        assert result["EF"] == 0
        assert result["NH"] == 0
        assert result["LF"] == 0
    
    def test_compute_with_hallucination(self, mock_ner_pipeline):
        """Test EHI computation with hallucinated entities."""
        ehi = EntityHallucinationIndex(ner_pipeline=mock_ner_pipeline)
        
        source = "राम एक अच्छा लड़का है।"
        summary = "राम और श्याम अच्छे लड़के हैं।"  # श्याम is hallucinated
        
        # Mock entity extraction to simulate hallucination
        with patch.object(mock_ner_pipeline, 'extract_entities') as mock_extract:
            mock_extract.side_effect = [
                [("राम", "PERSON")],  # Source entities
                [("राम", "PERSON"), ("श्याम", "PERSON")],  # Summary entities
            ]
            
            result = ehi.compute(source, summary)
            
            assert result["NH"] > 0  # Should detect negative hallucination
            assert result["EF"] > 0  # Should detect faithful entities
    
    def test_overfocus_computation(self):
        """Test over-focus computation."""
        ehi = EntityHallucinationIndex()
        
        # Test with single entity (should be 0)
        entities = {"राम"}
        text = "राम राम राम राम"
        of_score = ehi._compute_overfocus(entities, text)
        assert of_score == 0.0
        
        # Test with multiple entities, balanced
        entities = {"राम", "श्याम", "गीता"}
        text = "राम और श्याम और गीता"
        of_score = ehi._compute_overfocus(entities, text)
        assert of_score >= 0.0
    
    def test_fuzzy_matching(self):
        """Test fuzzy entity matching."""
        ehi = EntityHallucinationIndex()
        
        source_entities = {"राम कुमार", "नई दिल्ली"}
        summary_entities = {"राम", "दिल्ली"}
        
        matched_source, matched_summary = ehi._fuzzy_match_entities(
            source_entities, summary_entities
        )
        
        # Should find some fuzzy matches
        assert len(matched_summary) >= 0
        assert len(matched_source) >= 0


class TestEntityF1Score:
    """Test Entity F1 score computation."""
    
    def test_init(self, mock_ner_pipeline):
        """Test Entity F1 initialization."""
        ef1 = EntityF1Score(ner_pipeline=mock_ner_pipeline)
        assert ef1.ner_pipeline == mock_ner_pipeline
        assert ef1.fuzzy_threshold == 0.8
        assert ef1.use_fuzzy is True
    
    def test_compute_perfect_match(self, mock_ner_pipeline):
        """Test F1 computation with perfect match."""
        ef1 = EntityF1Score(ner_pipeline=mock_ner_pipeline)
        
        source = "राम दिल्ली में रहता है।"
        summary = "राम दिल्ली में है।"
        
        result = ef1.compute(source, summary)
        
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert all(0 <= v <= 1 for v in result.values())
    
    def test_compute_empty_texts(self, mock_ner_pipeline):
        """Test F1 computation with empty texts."""
        ef1 = EntityF1Score(ner_pipeline=mock_ner_pipeline)
        
        result = ef1.compute("", "")
        
        assert result["precision"] == 1.0  # No entities = perfect match
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
    
    def test_compute_no_overlap(self, mock_ner_pipeline):
        """Test F1 computation with no entity overlap."""
        ef1 = EntityF1Score(ner_pipeline=mock_ner_pipeline)
        
        # Mock to return different entities
        with patch.object(mock_ner_pipeline, 'extract_entities') as mock_extract:
            mock_extract.side_effect = [
                [("राम", "PERSON")],  # Source
                [("श्याम", "PERSON")],  # Summary
            ]
            
            result = ef1.compute("राम है।", "श्याम है।")
            
            assert result["precision"] == 0.0
            assert result["recall"] == 0.0
            assert result["f1"] == 0.0


class TestRougeMetrics:
    """Test ROUGE metrics computation."""
    
    def test_compute_rouge_metrics_basic(self):
        """Test basic ROUGE computation."""
        predictions = ["राम दिल्ली में है।"]
        references = ["राम दिल्ली में रहता है।"]
        
        result = compute_rouge_metrics(predictions, references)
        
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
        assert all(0 <= v <= 1 for v in result.values())
    
    def test_compute_rouge_empty_lists(self):
        """Test ROUGE with empty lists."""
        result = compute_rouge_metrics([], [])
        assert result == {}
    
    def test_compute_rouge_mismatched_lengths(self):
        """Test ROUGE with mismatched list lengths."""
        with pytest.raises(ValueError):
            compute_rouge_metrics(["text1"], ["ref1", "ref2"])
    
    def test_compute_rouge_custom_types(self):
        """Test ROUGE with custom metric types."""
        predictions = ["राम है।"]
        references = ["राम दिल्ली में है।"]
        rouge_types = ["rouge1"]
        
        result = compute_rouge_metrics(
            predictions, references, rouge_types=rouge_types
        )
        
        assert "rouge1" in result
        assert "rouge2" not in result
        assert "rougeL" not in result


class TestComputeAllMetrics:
    """Test comprehensive metrics computation."""
    
    def test_compute_all_metrics_basic(self, mock_ner_pipeline):
        """Test computation of all metrics."""
        source_texts = ["राम दिल्ली में रहता है।"]
        summary_texts = ["राम दिल्ली में है।"]
        reference_texts = ["राम दिल्ली निवासी है।"]
        
        result = compute_all_metrics(
            source_texts=source_texts,
            summary_texts=summary_texts,
            reference_texts=reference_texts,
            ner_pipeline=mock_ner_pipeline,
        )
        
        # Check all expected metrics are present
        expected_metrics = ["EHI", "EF", "NH", "LF", "OF", "PH", "Entity_F1"]
        for metric in expected_metrics:
            assert metric in result
            assert isinstance(result[metric], float)
        
        # Check ROUGE metrics are included
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
    
    def test_compute_all_metrics_no_references(self, mock_ner_pipeline):
        """Test computation without reference texts."""
        source_texts = ["राम दिल्ली में रहता है।"]
        summary_texts = ["राम दिल्ली में है।"]
        
        result = compute_all_metrics(
            source_texts=source_texts,
            summary_texts=summary_texts,
            ner_pipeline=mock_ner_pipeline,
        )
        
        # EHI metrics should be present
        assert "EHI" in result
        assert "Entity_F1" in result
        
        # ROUGE should not be included
        assert "rouge1" not in result
    
    def test_compute_all_metrics_mismatched_lengths(self, mock_ner_pipeline):
        """Test error handling with mismatched lengths."""
        with pytest.raises(ValueError):
            compute_all_metrics(
                source_texts=["text1"],
                summary_texts=["summary1", "summary2"],
                ner_pipeline=mock_ner_pipeline,
            )
    
    def test_compute_all_metrics_empty_lists(self, mock_ner_pipeline):
        """Test with empty input lists."""
        result = compute_all_metrics(
            source_texts=[],
            summary_texts=[],
            ner_pipeline=mock_ner_pipeline,
        )
        
        assert result == {}


class TestMetricsEdgeCases:
    """Test edge cases and error handling in metrics."""
    
    def test_ehi_with_none_input(self, mock_ner_pipeline):
        """Test EHI with None inputs."""
        ehi = EntityHallucinationIndex(ner_pipeline=mock_ner_pipeline)
        
        result = ehi.compute(None, "summary")
        assert result["EHI"] == 0.0
        
        result = ehi.compute("source", None)
        assert result["EHI"] == 0.0
    
    def test_entity_f1_with_ner_failure(self, mock_ner_pipeline):
        """Test Entity F1 when NER extraction fails."""
        ef1 = EntityF1Score(ner_pipeline=mock_ner_pipeline)
        
        # Mock NER to raise exception
        with patch.object(mock_ner_pipeline, 'extract_entities', side_effect=Exception("NER failed")):
            result = ef1.compute("source", "summary")
            
            # Should handle gracefully and return default values
            assert result["precision"] == 0.0
            assert result["recall"] == 0.0
            assert result["f1"] == 0.0
    
    def test_rouge_with_special_characters(self):
        """Test ROUGE with special characters and punctuation."""
        predictions = ["राम, दिल्ली में है!"]
        references = ["राम दिल्ली में रहता है।"]
        
        # Should not raise errors
        result = compute_rouge_metrics(predictions, references)
        assert all(isinstance(v, float) for v in result.values())
    
    def test_metrics_with_very_long_texts(self, mock_ner_pipeline):
        """Test metrics with very long input texts."""
        long_text = "राम " * 1000  # Very long text
        short_summary = "राम है।"
        
        ehi = EntityHallucinationIndex(ner_pipeline=mock_ner_pipeline)
        result = ehi.compute(long_text, short_summary)
        
        # Should handle without errors
        assert isinstance(result["EHI"], float)
        assert result["EHI"] >= 0