"""Test fixtures and utilities for the EHI test suite."""

import pytest
from typing import Dict, List, Tuple

# Sample Hindi texts for testing
SAMPLE_HINDI_TEXTS = [
    "राम एक अच्छा लड़का है। वह दिल्ली में रहता है।",
    "सीता माता एक महान व्यक्तित्व थीं। वे अयोध्या की रानी थीं।",
    "भारत सरकार ने नई नीति की घोषणा की है।",
    "मुंबई में एक नया व्यापारिक केंद्र खोला गया है।",
]

SAMPLE_SUMMARIES = [
    "राम दिल्ली में रहता है।",
    "सीता अयोध्या की रानी थीं।", 
    "सरकार ने नई नीति घोषित की।",
    "मुंबई में व्यापारिक केंद्र खुला।",
]

# Sample entities for testing
SAMPLE_ENTITIES = [
    [("राम", "PERSON"), ("दिल्ली", "LOC")],
    [("सीता", "PERSON"), ("अयोध्या", "LOC")],
    [("भारत सरकार", "ORG")],
    [("मुंबई", "LOC")],
]


@pytest.fixture
def sample_texts() -> List[str]:
    """Fixture providing sample Hindi texts."""
    return SAMPLE_HINDI_TEXTS.copy()


@pytest.fixture
def sample_summaries() -> List[str]:
    """Fixture providing sample Hindi summaries."""
    return SAMPLE_SUMMARIES.copy()


@pytest.fixture
def sample_entities() -> List[List[Tuple[str, str]]]:
    """Fixture providing sample entity annotations."""
    return [entities.copy() for entities in SAMPLE_ENTITIES]


@pytest.fixture
def sample_text_pairs() -> List[Tuple[str, str]]:
    """Fixture providing (source, summary) pairs."""
    return list(zip(SAMPLE_HINDI_TEXTS, SAMPLE_SUMMARIES))


@pytest.fixture
def mock_ner_pipeline():
    """Mock NER pipeline for testing."""
    class MockNERPipeline:
        def __init__(self):
            self.entity_map = dict(zip(SAMPLE_HINDI_TEXTS, SAMPLE_ENTITIES))
        
        def extract_entities(self, text: str) -> List[Tuple[str, str]]:
            return self.entity_map.get(text, [])
        
        def extract_entity_sets(self, source: str, summary: str) -> Tuple[set, set]:
            source_entities = {ent for ent, _ in self.extract_entities(source)}
            summary_entities = {ent for ent, _ in self.extract_entities(summary)}
            return source_entities, summary_entities
        
        def is_available(self) -> bool:
            return True
        
        def get_available_models(self) -> List[str]:
            return ["mock"]
    
    return MockNERPipeline()


@pytest.fixture
def sample_config() -> Dict:
    """Sample configuration for testing."""
    return {
        "model": {
            "name": "google/mt5-small",
            "use_lora": True,
            "lora": {
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
            },
        },
        "training": {
            "lr": 5e-5,
            "num_epochs": 1,
            "train_batch_size": 2,
            "eval_batch_size": 2,
        },
        "ehi_weights": {
            "w_EF": 1.0,
            "w_NH": 1.0,
            "w_LF": 1.0,
            "w_OF": 1.0,
            "w_PH": 0.0,
        },
        "generation": {
            "max_new_tokens": 50,
            "min_length": 5,
            "do_sample": False,
            "num_beams": 2,
        },
    }