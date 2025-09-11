"""Hindi Named Entity Recognition pipeline with multiple fallback models."""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from ehi.utils import normalize_unicode

logger = logging.getLogger(__name__)


class HindiNERPipeline:
    """
    Hindi NER pipeline with fallback models.
    
    Pipeline order:
    1. IndicNER/HiNER (if available)
    2. Stanza Hindi model
    3. spaCy multilingual model (xx_ent_wiki_sm)
    
    Features:
    - Unicode NFC normalization
    - Entity deduplication  
    - Case-insensitive comparison
    - Support for PERSON/ORG/LOC entity types
    """
    
    def __init__(
        self,
        use_indicner: bool = True,
        use_stanza: bool = True,
        use_spacy: bool = True,
        entity_types: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize the Hindi NER pipeline.
        
        Args:
            use_indicner: Whether to try IndicNER/HiNER models
            use_stanza: Whether to use Stanza Hindi model
            use_spacy: Whether to use spaCy multilingual model
            entity_types: Set of entity types to extract (None for all)
        """
        self.use_indicner = use_indicner
        self.use_stanza = use_stanza 
        self.use_spacy = use_spacy
        
        # Default to standard NER entity types
        self.entity_types = entity_types or {"PERSON", "ORG", "LOC", "MISC"}
        
        # Initialize models
        self.indicner_pipeline = None
        self.stanza_pipeline = None
        self.spacy_pipeline = None
        
        self._init_models()
    
    def _init_models(self) -> None:
        """Initialize available NER models with graceful fallback."""
        
        # Try IndicNER/HiNER first
        if self.use_indicner:
            try:
                # Note: This is a placeholder for IndicNER integration
                # In practice, you would integrate with actual IndicNER models
                logger.info("IndicNER/HiNER not implemented - using fallback models")
                self.indicner_pipeline = None
            except Exception as e:
                logger.warning(f"Failed to load IndicNER: {e}")
                self.indicner_pipeline = None
        
        # Try Stanza Hindi
        if self.use_stanza:
            try:
                import stanza
                
                # Download Hindi model if not present
                try:
                    self.stanza_pipeline = stanza.Pipeline(
                        lang="hi",
                        processors="tokenize,ner",
                        verbose=False,
                        download_method=None,
                    )
                    logger.info("Stanza Hindi NER model loaded successfully")
                except Exception:
                    # Try to download the model
                    logger.info("Downloading Stanza Hindi model...")
                    stanza.download("hi")
                    self.stanza_pipeline = stanza.Pipeline(
                        lang="hi", 
                        processors="tokenize,ner",
                        verbose=False,
                    )
                    logger.info("Stanza Hindi NER model downloaded and loaded")
                    
            except Exception as e:
                logger.warning(f"Failed to load Stanza Hindi model: {e}")
                self.stanza_pipeline = None
        
        # Try spaCy multilingual
        if self.use_spacy:
            try:
                import spacy
                
                # Try to load the multilingual model
                try:
                    self.spacy_pipeline = spacy.load("xx_ent_wiki_sm")
                    logger.info("spaCy multilingual NER model loaded successfully")
                except OSError:
                    logger.warning(
                        "spaCy xx_ent_wiki_sm model not found. "
                        "Install with: python -m spacy download xx_ent_wiki_sm"
                    )
                    self.spacy_pipeline = None
                    
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.spacy_pipeline = None
        
        # Check if any model is available
        available_models = []
        if self.indicner_pipeline:
            available_models.append("IndicNER")
        if self.stanza_pipeline:
            available_models.append("Stanza")
        if self.spacy_pipeline:
            available_models.append("spaCy")
        
        if not available_models:
            logger.error("No NER models available! Please install at least one.")
        else:
            logger.info(f"Available NER models: {', '.join(available_models)}")
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from Hindi text.
        
        Args:
            text: Input Hindi text
        
        Returns:
            List of (entity_text, entity_type) tuples
        """
        if not text or not text.strip():
            return []
        
        # Normalize Unicode text
        text = normalize_unicode(text, form="NFC")
        
        # Try models in order of preference
        entities = []
        
        # 1. Try IndicNER first (placeholder - not implemented)
        if self.indicner_pipeline:
            try:
                entities = self._extract_with_indicner(text)
                if entities:
                    logger.debug("Used IndicNER for entity extraction")
                    return self._postprocess_entities(entities)
            except Exception as e:
                logger.warning(f"IndicNER extraction failed: {e}")
        
        # 2. Try Stanza
        if self.stanza_pipeline:
            try:
                entities = self._extract_with_stanza(text)
                if entities:
                    logger.debug("Used Stanza for entity extraction")
                    return self._postprocess_entities(entities)
            except Exception as e:
                logger.warning(f"Stanza extraction failed: {e}")
        
        # 3. Try spaCy as final fallback
        if self.spacy_pipeline:
            try:
                entities = self._extract_with_spacy(text)
                if entities:
                    logger.debug("Used spaCy for entity extraction")
                    return self._postprocess_entities(entities)
            except Exception as e:
                logger.warning(f"spaCy extraction failed: {e}")
        
        logger.warning("No NER model succeeded in extracting entities")
        return []
    
    def _extract_with_indicner(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using IndicNER (placeholder implementation)."""
        # This is a placeholder for actual IndicNER integration
        # In practice, you would call the IndicNER model here
        return []
    
    def _extract_with_stanza(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using Stanza Hindi model."""
        if not self.stanza_pipeline:
            return []
        
        doc = self.stanza_pipeline(text)
        entities = []
        
        for sentence in doc.sentences:
            for entity in sentence.ents:
                # Map Stanza entity types to our standard types
                entity_type = self._map_entity_type(entity.type)
                if entity_type in self.entity_types:
                    entities.append((entity.text, entity_type))
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using spaCy multilingual model."""
        if not self.spacy_pipeline:
            return []
        
        doc = self.spacy_pipeline(text)
        entities = []
        
        for ent in doc.ents:
            # Map spaCy entity types to our standard types
            entity_type = self._map_entity_type(ent.label_)
            if entity_type in self.entity_types:
                entities.append((ent.text, entity_type))
        
        return entities
    
    def _map_entity_type(self, original_type: str) -> str:
        """
        Map model-specific entity types to standard types.
        
        Args:
            original_type: Original entity type from the model
        
        Returns:
            Standardized entity type
        """
        # Mapping from various NER schemes to our standard types
        type_mapping = {
            # Stanza/CoNLL types
            "PER": "PERSON",
            "PERSON": "PERSON",
            "ORG": "ORG",
            "ORGANIZATION": "ORG",
            "LOC": "LOC",
            "LOCATION": "LOC",
            "GPE": "LOC",  # Geopolitical entity
            "MISC": "MISC",
            
            # spaCy types
            "PERSON": "PERSON",
            "ORG": "ORG",
            "GPE": "LOC",
            "LOC": "LOC",
            "FAC": "LOC",  # Facility
            "NORP": "MISC",  # Nationalities/groups
            "EVENT": "MISC",
            "PRODUCT": "MISC",
            "WORK_OF_ART": "MISC",
            "LANGUAGE": "MISC",
        }
        
        return type_mapping.get(original_type.upper(), "MISC")
    
    def _postprocess_entities(
        self, entities: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """
        Post-process extracted entities.
        
        Args:
            entities: Raw list of (entity_text, entity_type) tuples
        
        Returns:
            Cleaned and deduplicated list of entities
        """
        if not entities:
            return []
        
        # Clean and normalize entities
        cleaned_entities = []
        seen_entities = set()
        
        for entity_text, entity_type in entities:
            # Clean entity text
            cleaned_text = self._clean_entity_text(entity_text)
            
            if not cleaned_text:
                continue
            
            # Create case-insensitive key for deduplication
            dedup_key = (cleaned_text.lower(), entity_type)
            
            if dedup_key not in seen_entities:
                seen_entities.add(dedup_key)
                cleaned_entities.append((cleaned_text, entity_type))
        
        return cleaned_entities
    
    def _clean_entity_text(self, text: str) -> str:
        """
        Clean entity text by removing extra whitespace and punctuation.
        
        Args:
            text: Raw entity text
        
        Returns:
            Cleaned entity text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        
        # Remove leading/trailing punctuation but keep internal punctuation
        text = re.sub(r"^[^\w\u0900-\u097F]+|[^\w\u0900-\u097F]+$", "", text)
        
        return text
    
    def extract_entity_sets(
        self, source_text: str, summary_text: str
    ) -> Tuple[Set[str], Set[str]]:
        """
        Extract entity sets from source and summary texts.
        
        Args:
            source_text: Source document text
            summary_text: Summary text
        
        Returns:
            Tuple of (source_entities, summary_entities) as sets of entity strings
        """
        source_entities = self.extract_entities(source_text)
        summary_entities = self.extract_entities(summary_text)
        
        # Convert to sets of entity strings (case-insensitive)
        source_set = {entity.lower() for entity, _ in source_entities}
        summary_set = {entity.lower() for entity, _ in summary_entities}
        
        return source_set, summary_set
    
    def is_available(self) -> bool:
        """Check if at least one NER model is available."""
        return any([
            self.indicner_pipeline is not None,
            self.stanza_pipeline is not None,
            self.spacy_pipeline is not None,
        ])
    
    def get_available_models(self) -> List[str]:
        """Get list of available NER models."""
        models = []
        if self.indicner_pipeline:
            models.append("IndicNER")
        if self.stanza_pipeline:
            models.append("Stanza")
        if self.spacy_pipeline:
            models.append("spaCy")
        return models