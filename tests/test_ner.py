"""Tests for Hindi NER pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from ehi.ner import HindiNERPipeline
from tests.fixtures import sample_texts, sample_entities


class TestHindiNERPipeline:
    """Test Hindi NER pipeline functionality."""
    
    def test_init_default(self):
        """Test NER pipeline initialization with defaults."""
        ner = HindiNERPipeline()
        
        assert ner.use_indicner is True
        assert ner.use_stanza is True
        assert ner.use_spacy is True
        assert ner.entity_types == {"PERSON", "ORG", "LOC", "MISC"}
    
    def test_init_custom_entity_types(self):
        """Test initialization with custom entity types."""
        custom_types = {"PERSON", "ORG"}
        ner = HindiNERPipeline(entity_types=custom_types)
        
        assert ner.entity_types == custom_types
    
    def test_init_selective_models(self):
        """Test initialization with selective model usage."""
        ner = HindiNERPipeline(
            use_indicner=False,
            use_stanza=True,
            use_spacy=False,
        )
        
        assert ner.use_indicner is False
        assert ner.use_stanza is True
        assert ner.use_spacy is False
    
    @patch('ehi.ner.logger')
    def test_init_models_graceful_failure(self, mock_logger):
        """Test graceful handling when models fail to load."""
        with patch('stanza.Pipeline', side_effect=Exception("Stanza failed")):
            with patch('spacy.load', side_effect=Exception("spaCy failed")):
                ner = HindiNERPipeline()
                
                # Should not crash, just log warnings
                assert mock_logger.warning.called
                assert ner.stanza_pipeline is None
                assert ner.spacy_pipeline is None
    
    def test_extract_entities_empty_text(self):
        """Test entity extraction with empty text."""
        ner = HindiNERPipeline()
        
        result = ner.extract_entities("")
        assert result == []
        
        result = ner.extract_entities("   ")
        assert result == []
    
    def test_extract_entities_with_mock_stanza(self):
        """Test entity extraction using mocked Stanza."""
        ner = HindiNERPipeline()
        
        # Mock Stanza pipeline
        mock_pipeline = Mock()
        mock_doc = Mock()
        mock_sentence = Mock()
        mock_entity = Mock()
        
        mock_entity.text = "राम"
        mock_entity.type = "PERSON"
        mock_sentence.ents = [mock_entity]
        mock_doc.sentences = [mock_sentence]
        mock_pipeline.return_value = mock_doc
        
        ner.stanza_pipeline = mock_pipeline
        ner.spacy_pipeline = None  # Disable spaCy
        ner.indicner_pipeline = None  # Disable IndicNER
        
        result = ner.extract_entities("राम एक व्यक्ति है।")
        
        assert len(result) == 1
        assert result[0] == ("राम", "PERSON")
    
    def test_extract_entities_with_mock_spacy(self):
        """Test entity extraction using mocked spaCy."""
        ner = HindiNERPipeline()
        
        # Mock spaCy pipeline
        mock_pipeline = Mock()
        mock_doc = Mock()
        mock_entity = Mock()
        
        mock_entity.text = "दिल्ली"
        mock_entity.label_ = "GPE"
        mock_doc.ents = [mock_entity]
        mock_pipeline.return_value = mock_doc
        
        ner.spacy_pipeline = mock_pipeline
        ner.stanza_pipeline = None  # Disable Stanza
        ner.indicner_pipeline = None  # Disable IndicNER
        
        result = ner.extract_entities("दिल्ली एक शहर है।")
        
        assert len(result) == 1
        assert result[0] == ("दिल्ली", "LOC")  # GPE mapped to LOC
    
    def test_map_entity_type(self):
        """Test entity type mapping."""
        ner = HindiNERPipeline()
        
        # Test Stanza/CoNLL mappings
        assert ner._map_entity_type("PER") == "PERSON"
        assert ner._map_entity_type("ORG") == "ORG"
        assert ner._map_entity_type("LOC") == "LOC"
        assert ner._map_entity_type("GPE") == "LOC"
        
        # Test spaCy mappings
        assert ner._map_entity_type("PERSON") == "PERSON"
        assert ner._map_entity_type("ORG") == "ORG"
        assert ner._map_entity_type("FAC") == "LOC"
        
        # Test unknown type
        assert ner._map_entity_type("UNKNOWN") == "MISC"
    
    def test_clean_entity_text(self):
        """Test entity text cleaning."""
        ner = HindiNERPipeline()
        
        # Test whitespace removal
        assert ner._clean_entity_text("  राम  ") == "राम"
        assert ner._clean_entity_text("राम\nकुमार") == "राम कुमार"
        
        # Test punctuation removal
        assert ner._clean_entity_text("राम,") == "राम"
        assert ner._clean_entity_text(",राम.") == "राम"
        assert ner._clean_entity_text("श्री राम जी") == "श्री राम जी"
        
        # Test empty result
        assert ner._clean_entity_text(".,!") == ""
        assert ner._clean_entity_text("") == ""
    
    def test_postprocess_entities(self):
        """Test entity post-processing."""
        ner = HindiNERPipeline()
        
        # Test deduplication (case-insensitive)
        entities = [
            ("राम", "PERSON"),
            ("RAM", "PERSON"),  # Different case
            ("राम", "PERSON"),  # Duplicate
            ("दिल्ली", "LOC"),
        ]
        
        result = ner._postprocess_entities(entities)
        
        # Should remove duplicates
        assert len(result) == 3  # राम (2 variations), दिल्ली
        
        # Check that entities are cleaned
        entity_texts = [ent[0] for ent in result]
        assert all(text.strip() == text for text in entity_texts)
    
    def test_extract_entity_sets(self):
        """Test entity set extraction."""
        ner = HindiNERPipeline()
        
        # Mock entity extraction
        with patch.object(ner, 'extract_entities') as mock_extract:
            mock_extract.side_effect = [
                [("राम", "PERSON"), ("दिल्ली", "LOC")],  # Source
                [("राम", "PERSON")],  # Summary
            ]
            
            source_set, summary_set = ner.extract_entity_sets(
                "राम दिल्ली में रहता है।",
                "राम एक व्यक्ति है।"
            )
            
            assert source_set == {"राम", "दिल्ली"}
            assert summary_set == {"राम"}
    
    def test_is_available(self):
        """Test availability checking."""
        ner = HindiNERPipeline()
        
        # Mock all pipelines as None
        ner.indicner_pipeline = None
        ner.stanza_pipeline = None
        ner.spacy_pipeline = None
        
        assert ner.is_available() is False
        
        # Mock one pipeline as available
        ner.stanza_pipeline = Mock()
        assert ner.is_available() is True
    
    def test_get_available_models(self):
        """Test getting available model list."""
        ner = HindiNERPipeline()
        
        # Mock different availability scenarios
        ner.indicner_pipeline = None
        ner.stanza_pipeline = Mock()
        ner.spacy_pipeline = Mock()
        
        models = ner.get_available_models()
        
        assert "Stanza" in models
        assert "spaCy" in models
        assert "IndicNER" not in models
    
    def test_extract_entities_fallback_order(self):
        """Test that NER extraction follows correct fallback order."""
        ner = HindiNERPipeline()
        
        # Mock all pipelines but with different behaviors
        ner.indicner_pipeline = None  # Not available
        
        mock_stanza = Mock()
        mock_stanza_doc = Mock()
        mock_stanza_sentence = Mock()
        mock_stanza_entity = Mock()
        mock_stanza_entity.text = "राम"
        mock_stanza_entity.type = "PERSON"
        mock_stanza_sentence.ents = [mock_stanza_entity]
        mock_stanza_doc.sentences = [mock_stanza_sentence]
        mock_stanza.return_value = mock_stanza_doc
        ner.stanza_pipeline = mock_stanza
        
        mock_spacy = Mock()
        ner.spacy_pipeline = mock_spacy
        
        # Call extraction
        result = ner.extract_entities("राम एक व्यक्ति है।")
        
        # Should use Stanza (higher priority than spaCy)
        assert mock_stanza.called
        assert not mock_spacy.called
        assert len(result) == 1
        assert result[0] == ("राम", "PERSON")
    
    def test_extract_entities_spacy_fallback(self):
        """Test spaCy fallback when Stanza fails."""
        ner = HindiNERPipeline()
        
        # Mock Stanza to fail
        ner.indicner_pipeline = None
        ner.stanza_pipeline = Mock(side_effect=Exception("Stanza failed"))
        
        # Mock spaCy to succeed
        mock_spacy = Mock()
        mock_doc = Mock()
        mock_entity = Mock()
        mock_entity.text = "दिल्ली"
        mock_entity.label_ = "GPE"
        mock_doc.ents = [mock_entity]
        mock_spacy.return_value = mock_doc
        ner.spacy_pipeline = mock_spacy
        
        result = ner.extract_entities("दिल्ली एक शहर है।")
        
        # Should fallback to spaCy
        assert mock_spacy.called
        assert len(result) == 1
        assert result[0] == ("दिल्ली", "LOC")
    
    def test_unicode_normalization(self):
        """Test Unicode normalization in entity extraction."""
        ner = HindiNERPipeline()
        
        # Mock extraction to test normalization
        with patch('ehi.utils.normalize_unicode') as mock_normalize:
            mock_normalize.return_value = "normalized_text"
            
            # Mock pipeline to avoid actual NER
            ner.stanza_pipeline = None
            ner.spacy_pipeline = None
            ner.indicner_pipeline = None
            
            ner.extract_entities("test_text")
            
            # Should call normalization
            mock_normalize.assert_called_with("test_text", form="NFC")
    
    def test_entity_type_filtering(self):
        """Test filtering by entity types."""
        ner = HindiNERPipeline(entity_types={"PERSON"})
        
        # Mock extraction with mixed entity types
        mock_pipeline = Mock()
        mock_doc = Mock()
        mock_sentence = Mock()
        
        # Create entities of different types
        person_entity = Mock()
        person_entity.text = "राम"
        person_entity.type = "PERSON"
        
        loc_entity = Mock()
        loc_entity.text = "दिल्ली"
        loc_entity.type = "LOC"
        
        mock_sentence.ents = [person_entity, loc_entity]
        mock_doc.sentences = [mock_sentence]
        mock_pipeline.return_value = mock_doc
        
        ner.stanza_pipeline = mock_pipeline
        ner.spacy_pipeline = None
        ner.indicner_pipeline = None
        
        result = ner.extract_entities("राम दिल्ली में रहता है।")
        
        # Should only return PERSON entities
        assert len(result) == 1
        assert result[0] == ("राम", "PERSON")
    
    def test_no_models_available(self):
        """Test behavior when no NER models are available."""
        ner = HindiNERPipeline()
        
        # Set all pipelines to None
        ner.indicner_pipeline = None
        ner.stanza_pipeline = None
        ner.spacy_pipeline = None
        
        result = ner.extract_entities("राम दिल्ली में रहता है।")
        
        # Should return empty list gracefully
        assert result == []