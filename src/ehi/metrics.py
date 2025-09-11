"""Entity Hallucination Index (EHI) and related metrics for summarization evaluation."""

import logging
import math
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from rapidfuzz import fuzz
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score

from ehi.ner import HindiNERPipeline

logger = logging.getLogger(__name__)


class EntityHallucinationIndex:
    """
    Entity Hallucination Index (EHI) metric for evaluating factual consistency.
    
    EHI components:
    - EF (Entity Faithful): Entities present in both source and summary
    - NH (Negative Hallucination): Entities in summary but not in source
    - LF (Lost Faithful): Entities in source but not in summary  
    - OF (Over-Focus): Penalty for narrow entity concentration
    - PH (Positive Hallucination): Verified entities not in source (optional)
    
    EHI = (w_EF*|EF| + w_PH*|PH|) / (w_EF*|EF| + w_PH*|PH| + w_NH*|NH| + w_OF*|OF| + w_LF*|LF|)
    """
    
    def __init__(
        self,
        ner_pipeline: Optional[HindiNERPipeline] = None,
        weights: Optional[Dict[str, float]] = None,
        enable_positive_hallucination: bool = False,
        overfocus_threshold: float = 0.7,
        entity_similarity_threshold: float = 0.8,
    ) -> None:
        """
        Initialize EHI metric.
        
        Args:
            ner_pipeline: Hindi NER pipeline for entity extraction
            weights: Component weights dict with keys: w_EF, w_NH, w_LF, w_OF, w_PH
            enable_positive_hallucination: Whether to compute PH component
            overfocus_threshold: Threshold for over-focus detection
            entity_similarity_threshold: Threshold for fuzzy entity matching
        """
        self.ner_pipeline = ner_pipeline or HindiNERPipeline()
        
        # Default weights as specified in the problem statement
        self.weights = weights or {
            "w_EF": 1.0,  # Entity Faithful
            "w_NH": 1.0,  # Negative Hallucination  
            "w_LF": 1.0,  # Lost Faithful
            "w_OF": 1.0,  # Over-Focus
            "w_PH": 0.0,  # Positive Hallucination (default off)
        }
        
        self.enable_positive_hallucination = enable_positive_hallucination
        self.overfocus_threshold = overfocus_threshold
        self.entity_similarity_threshold = entity_similarity_threshold
        
        if not self.ner_pipeline.is_available():
            logger.warning("No NER models available - EHI computation may fail")
    
    def compute(
        self,
        source_text: str,
        summary_text: str,
        verified_entities: Optional[Set[str]] = None,
    ) -> Dict[str, Union[float, int]]:
        """
        Compute EHI and component metrics.
        
        Args:
            source_text: Source document text
            summary_text: Generated summary text
            verified_entities: Optional set of verified entities for PH computation
        
        Returns:
            Dictionary with EHI score and component values
        """
        if not source_text or not summary_text:
            return self._empty_result()
        
        try:
            # Extract entities from source and summary
            source_entities = self._extract_entity_set(source_text)
            summary_entities = self._extract_entity_set(summary_text)
            
            # Compute fuzzy matches to handle minor variations
            source_entities, summary_entities = self._fuzzy_match_entities(
                source_entities, summary_entities
            )
            
            # Compute EHI components
            ef_entities = source_entities & summary_entities  # Entity Faithful
            nh_entities = summary_entities - source_entities  # Negative Hallucination
            lf_entities = source_entities - summary_entities  # Lost Faithful
            
            # Over-focus computation
            of_score = self._compute_overfocus(summary_entities, summary_text)
            
            # Positive hallucination (if enabled)
            ph_entities = set()
            if self.enable_positive_hallucination and verified_entities:
                ph_entities = (summary_entities - source_entities) & verified_entities
                # Remove PH entities from NH count
                nh_entities = nh_entities - ph_entities
            
            # Component counts
            ef_count = len(ef_entities)
            nh_count = len(nh_entities)
            lf_count = len(lf_entities)
            ph_count = len(ph_entities)
            
            # Compute weighted EHI score
            numerator = (
                self.weights["w_EF"] * ef_count + 
                self.weights["w_PH"] * ph_count
            )
            
            denominator = (
                self.weights["w_EF"] * ef_count +
                self.weights["w_PH"] * ph_count +
                self.weights["w_NH"] * nh_count +
                self.weights["w_OF"] * of_score +
                self.weights["w_LF"] * lf_count
            )
            
            # Handle edge case where denominator is zero
            ehi_score = numerator / denominator if denominator > 0 else 0.0
            
            return {
                "EHI": ehi_score,
                "EF": ef_count,
                "NH": nh_count,
                "LF": lf_count,
                "OF": of_score,
                "PH": ph_count,
                "total_source_entities": len(source_entities),
                "total_summary_entities": len(summary_entities),
                "entity_faithful_entities": list(ef_entities),
                "negative_hallucination_entities": list(nh_entities),
                "lost_faithful_entities": list(lf_entities),
                "positive_hallucination_entities": list(ph_entities),
            }
            
        except Exception as e:
            logger.error(f"Error computing EHI: {e}")
            return self._empty_result()
    
    def _extract_entity_set(self, text: str) -> Set[str]:
        """Extract entity set from text using NER pipeline."""
        entities = self.ner_pipeline.extract_entities(text)
        # Return case-insensitive entity set
        return {entity.lower().strip() for entity, _ in entities if entity.strip()}
    
    def _fuzzy_match_entities(
        self, source_entities: Set[str], summary_entities: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Perform fuzzy matching between source and summary entities.
        
        Args:
            source_entities: Set of source entities
            summary_entities: Set of summary entities
        
        Returns:
            Tuple of (matched_source_entities, matched_summary_entities)
        """
        if not source_entities or not summary_entities:
            return source_entities, summary_entities
        
        matched_source = set(source_entities)
        matched_summary = set()
        
        # For each summary entity, find best match in source
        for summary_entity in summary_entities:
            best_match = None
            best_score = 0.0
            
            for source_entity in source_entities:
                # Compute fuzzy similarity
                similarity = fuzz.ratio(summary_entity, source_entity) / 100.0
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = source_entity
            
            # If similarity is above threshold, treat as same entity
            if best_score >= self.entity_similarity_threshold and best_match:
                matched_summary.add(best_match)  # Use source entity form
            else:
                matched_summary.add(summary_entity)  # Keep original
        
        return matched_source, matched_summary
    
    def _compute_overfocus(self, entities: Set[str], text: str) -> float:
        """
        Compute over-focus score using entity frequency concentration.
        
        Args:
            entities: Set of extracted entities
            text: Original text for frequency analysis
        
        Returns:
            Over-focus score (higher = more concentrated)
        """
        if not entities or len(entities) <= 1:
            return 0.0
        
        # Count entity frequencies in text (case-insensitive)
        text_lower = text.lower()
        entity_counts = {}
        
        for entity in entities:
            # Count occurrences of entity in text
            count = text_lower.count(entity.lower())
            entity_counts[entity] = max(count, 1)  # At least 1 occurrence
        
        # Compute Herfindahl index for concentration
        total_mentions = sum(entity_counts.values())
        if total_mentions == 0:
            return 0.0
        
        # Normalized frequencies
        frequencies = [count / total_mentions for count in entity_counts.values()]
        
        # Herfindahl index (sum of squared frequencies)
        herfindahl = sum(freq ** 2 for freq in frequencies)
        
        # Normalize to [0, 1] and apply threshold
        max_herfindahl = 1.0  # When all mentions are of one entity
        min_herfindahl = 1.0 / len(entities)  # When equally distributed
        
        if max_herfindahl > min_herfindahl:
            normalized_herfindahl = (herfindahl - min_herfindahl) / (
                max_herfindahl - min_herfindahl
            )
        else:
            normalized_herfindahl = 0.0
        
        # Apply over-focus penalty if above threshold
        if normalized_herfindahl > self.overfocus_threshold:
            return normalized_herfindahl * len(entities)  # Scale by entity count
        
        return 0.0
    
    def _empty_result(self) -> Dict[str, Union[float, int]]:
        """Return empty/default result when computation fails."""
        return {
            "EHI": 0.0,
            "EF": 0,
            "NH": 0, 
            "LF": 0,
            "OF": 0.0,
            "PH": 0,
            "total_source_entities": 0,
            "total_summary_entities": 0,
            "entity_faithful_entities": [],
            "negative_hallucination_entities": [],
            "lost_faithful_entities": [],
            "positive_hallucination_entities": [],
        }


class EntityF1Score:
    """
    Entity F1 score for measuring entity overlap between source and summary.
    Supports both exact and fuzzy matching.
    """
    
    def __init__(
        self,
        ner_pipeline: Optional[HindiNERPipeline] = None,
        fuzzy_threshold: float = 0.8,
        use_fuzzy: bool = True,
    ) -> None:
        """
        Initialize Entity F1 metric.
        
        Args:
            ner_pipeline: Hindi NER pipeline for entity extraction
            fuzzy_threshold: Threshold for fuzzy string matching
            use_fuzzy: Whether to use fuzzy matching
        """
        self.ner_pipeline = ner_pipeline or HindiNERPipeline()
        self.fuzzy_threshold = fuzzy_threshold
        self.use_fuzzy = use_fuzzy
    
    def compute(self, source_text: str, summary_text: str) -> Dict[str, float]:
        """
        Compute Entity F1 score.
        
        Args:
            source_text: Source document text
            summary_text: Generated summary text
        
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not source_text or not summary_text:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            # Extract entities
            source_entities = self._extract_entity_set(source_text)
            summary_entities = self._extract_entity_set(summary_text)
            
            if not source_entities and not summary_entities:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            
            if not source_entities:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
            if not summary_entities:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
            # Compute matches
            if self.use_fuzzy:
                matches = self._fuzzy_matches(source_entities, summary_entities)
            else:
                matches = len(source_entities & summary_entities)
            
            # Compute metrics
            precision = matches / len(summary_entities)
            recall = matches / len(source_entities)
            
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            
        except Exception as e:
            logger.error(f"Error computing Entity F1: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def _extract_entity_set(self, text: str) -> Set[str]:
        """Extract entity set from text."""
        entities = self.ner_pipeline.extract_entities(text)
        return {entity.lower().strip() for entity, _ in entities if entity.strip()}
    
    def _fuzzy_matches(self, source_entities: Set[str], summary_entities: Set[str]) -> int:
        """Count fuzzy matches between entity sets."""
        matches = 0
        used_source = set()
        
        for summary_entity in summary_entities:
            best_match = None
            best_score = 0.0
            
            for source_entity in source_entities:
                if source_entity in used_source:
                    continue
                
                similarity = fuzz.ratio(summary_entity, source_entity) / 100.0
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = source_entity
            
            if best_score >= self.fuzzy_threshold and best_match:
                matches += 1
                used_source.add(best_match)
        
        return matches


def compute_rouge_metrics(
    predictions: List[str],
    references: List[str],
    rouge_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute ROUGE metrics for summarization evaluation.
    
    Args:
        predictions: List of generated summaries
        references: List of reference summaries
        rouge_types: List of ROUGE types to compute (default: rouge1, rouge2, rougeL)
    
    Returns:
        Dictionary with ROUGE scores
    """
    if not predictions or not references:
        return {}
    
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=False)
    
    scores = {rouge_type: [] for rouge_type in rouge_types}
    
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            # Add zeros for empty predictions/references
            for rouge_type in rouge_types:
                scores[rouge_type].append(0.0)
            continue
        
        rouge_scores = scorer.score(ref, pred)
        
        for rouge_type in rouge_types:
            if rouge_type in rouge_scores:
                # Use F1 score for each ROUGE metric
                scores[rouge_type].append(rouge_scores[rouge_type].fmeasure)
            else:
                scores[rouge_type].append(0.0)
    
    # Compute averages
    avg_scores = {}
    for rouge_type, score_list in scores.items():
        avg_scores[rouge_type] = np.mean(score_list) if score_list else 0.0
    
    return avg_scores


def compute_all_metrics(
    source_texts: List[str],
    summary_texts: List[str],
    reference_texts: Optional[List[str]] = None,
    ner_pipeline: Optional[HindiNERPipeline] = None,
    ehi_weights: Optional[Dict[str, float]] = None,
    verified_entities_list: Optional[List[Set[str]]] = None,
) -> Dict[str, float]:
    """
    Compute all metrics (EHI, Entity F1, ROUGE) for a dataset.
    
    Args:
        source_texts: List of source documents
        summary_texts: List of generated summaries  
        reference_texts: Optional list of reference summaries for ROUGE
        ner_pipeline: Hindi NER pipeline
        ehi_weights: EHI component weights
        verified_entities_list: List of verified entity sets for each example
    
    Returns:
        Dictionary with averaged metric scores
    """
    if not source_texts or not summary_texts:
        return {}
    
    if len(source_texts) != len(summary_texts):
        raise ValueError("Number of source texts and summaries must match")
    
    # Initialize metrics
    ehi_metric = EntityHallucinationIndex(
        ner_pipeline=ner_pipeline,
        weights=ehi_weights,
    )
    entity_f1_metric = EntityF1Score(ner_pipeline=ner_pipeline)
    
    # Collect scores
    ehi_scores = []
    ef_scores = []
    nh_scores = []
    lf_scores = []
    of_scores = []
    ph_scores = []
    entity_f1_scores = []
    
    for i, (source, summary) in enumerate(zip(source_texts, summary_texts)):
        # Get verified entities for this example if available
        verified_entities = (
            verified_entities_list[i] if verified_entities_list else None
        )
        
        # Compute EHI
        ehi_result = ehi_metric.compute(source, summary, verified_entities)
        ehi_scores.append(ehi_result["EHI"])
        ef_scores.append(ehi_result["EF"])
        nh_scores.append(ehi_result["NH"])
        lf_scores.append(ehi_result["LF"])
        of_scores.append(ehi_result["OF"])
        ph_scores.append(ehi_result["PH"])
        
        # Compute Entity F1
        ef1_result = entity_f1_metric.compute(source, summary)
        entity_f1_scores.append(ef1_result["f1"])
    
    # Aggregate results
    results = {
        "EHI": np.mean(ehi_scores) if ehi_scores else 0.0,
        "EF": np.mean(ef_scores) if ef_scores else 0.0,
        "NH": np.mean(nh_scores) if nh_scores else 0.0,
        "LF": np.mean(lf_scores) if lf_scores else 0.0,
        "OF": np.mean(of_scores) if of_scores else 0.0,
        "PH": np.mean(ph_scores) if ph_scores else 0.0,
        "Entity_F1": np.mean(entity_f1_scores) if entity_f1_scores else 0.0,
    }
    
    # Add ROUGE metrics if reference texts are provided
    if reference_texts:
        if len(reference_texts) != len(summary_texts):
            logger.warning("Number of reference texts doesn't match summaries")
        else:
            rouge_scores = compute_rouge_metrics(summary_texts, reference_texts)
            results.update(rouge_scores)
    
    return results