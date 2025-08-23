#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_factcc_qags_scores.py

Parse JSON files in the repository and compute FactCC and QAGS scores.

This script reads JSON files from the repository and calculates:
1. FactCC scores - consistency probability via sequence classifier
2. QAGS scores - QA-based faithfulness via question generation and answering

Usage:
    python compute_factcc_qags_scores.py --input_path input.json --output_path output.json
    
    Or to process all JSON files in repository:
    python compute_factcc_qags_scores.py --process_all
"""

import argparse
import json
import math
import os
import sys
import glob
from typing import Any, Dict, List, Optional, Tuple
import re
from collections import Counter

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# ----------------------------
# Helper functions
# ----------------------------

def pick_first(*vals):
    """Return the first non-empty value from provided candidates."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def sanitize(s: Optional[str]) -> str:
    """Sanitize string input."""
    return s if (isinstance(s, str) and len(s) > 0) else ""


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply softmax to tensor."""
    return torch.nn.functional.softmax(x, dim=dim)


def tokenize_pair(tokenizer, a: str, b: str, max_a: int, max_b: int):
    """Tokenize a pair (a=source, b=summary) with independent truncation budgets."""
    a_ids = tokenizer.encode(a, add_special_tokens=False, truncation=True, max_length=max_a)
    b_ids = tokenizer.encode(b, add_special_tokens=False, truncation=True, max_length=max_b)

    pair = tokenizer.prepare_for_model(
        a_ids,
        b_ids,
        truncation=True,
        max_length=max_a + max_b + 3,  # [CLS], [SEP], [SEP]
        return_tensors="pt",
        add_special_tokens=True,
    )
    return pair


def simple_token_f1(a: str, b: str) -> float:
    """Token-level F1 (space-split; case-insensitive; strips punctuation lightly)."""
    def norm(t: str) -> List[str]:
        t = t.lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        toks = [x for x in t.split() if x]
        return toks

    A, B = norm(a), norm(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    
    cA, cB = Counter(A), Counter(B)
    overlap = sum((cA & cB).values())
    if overlap == 0:
        return 0.0
    
    precision = overlap / max(1, sum(cA.values()))
    recall = overlap / max(1, sum(cB.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ----------------------------
# FactCC Scorer
# ----------------------------

class FactCCScorer:
    """
    Wraps a sequence classifier to output P(consistent | source, summary).
    """

    def __init__(self, model_name: str = "tals/albert-base-v2-factcc", device: str = "auto"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            self.device = -1
        else:
            try:
                self.device = int(device)
            except Exception:
                self.device = -1

        if self.device >= 0:
            self.model.to(f"cuda:{self.device}")
        self.model.eval()

        # Try to infer label mapping
        self.consistent_label_idx = None
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) == self.model.config.num_labels:
            for k, v in id2label.items():
                name = str(v).lower()
                if any(t in name for t in ["entail", "support", "consistent", "true", "label_1"]):
                    self.consistent_label_idx = int(k)
                    break
            if self.consistent_label_idx is None and 1 in id2label:
                self.consistent_label_idx = 1  # assume positive=1
        if self.consistent_label_idx is None:
            self.consistent_label_idx = -1  # will take max logit as 'consistent'

    @torch.no_grad()
    def score(self, source: str, summary: str,
              max_source_tokens: int = 512, max_summary_tokens: int = 256) -> float:
        """Calculate FactCC score for source-summary pair."""
        source = sanitize(source)
        summary = sanitize(summary)
        if not source or not summary:
            return 0.0

        pair = tokenize_pair(self.tokenizer, source, summary, max_source_tokens, max_summary_tokens)
        if self.device >= 0:
            pair = {k: v.to(f"cuda:{self.device}") for k, v in pair.items()}
        
        logits = self.model(**pair).logits  # [1, num_labels]
        probs = softmax(logits, dim=-1).squeeze(0)

        if self.consistent_label_idx == -1:
            # take max as 'consistent'
            return float(torch.max(probs).item())
        else:
            return float(probs[self.consistent_label_idx].item())


# ----------------------------
# QAGS Scorer
# ----------------------------

class QAGSLite:
    """
    QAGS-like scorer: generate questions from SUMMARY, then answer the questions
    twice (from SOURCE and from SUMMARY). Compute token-level F1 between the
    two answers and average.
    """

    def __init__(self,
                 qg_model: str = "valhalla/t5-small-qg-hl",
                 qa_model: str = "deepset/roberta-base-squad2",
                 device: str = "auto"):
        
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            self.device = -1
        else:
            try:
                self.device = int(device)
            except Exception:
                self.device = -1

        # QG model
        self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model, use_fast=True)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model)
        if self.device >= 0:
            self.qg_model.to(f"cuda:{self.device}")
        self.qg_model.eval()

        # QA pipeline
        self.qa_pipe = pipeline(
            task="question-answering",
            model=qa_model,
            tokenizer=qa_model,
            device=self.device
        )

    @torch.no_grad()
    def _gen_questions(self, summary: str, max_questions: int = 5,
                       max_input_len: int = 384, max_out_len: int = 32) -> List[str]:
        """Generate questions from summary using highlight trick."""
        sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', summary) if s.strip()]
        questions = []
        
        for sent in sents:
            prompt = f"generate questions: <hl> {sent} <hl> {summary}"
            enc = self.qg_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len
            )
            if self.device >= 0:
                enc = {k: v.to(f"cuda:{self.device}") for k, v in enc.items()}
            
            out = self.qg_model.generate(
                **enc,
                num_beams=4,
                max_length=max_out_len,
                early_stopping=True
            )
            text = self.qg_tokenizer.decode(out[0], skip_special_tokens=True).strip()
            
            # Models often emit multiple questions separated by "?" or "<sep>"
            parts = re.split(r'\?|\<sep\>', text)
            for p in parts:
                q = p.strip()
                if q:
                    if not q.endswith("?"):
                        q += "?"
                    questions.append(q)
            if len(questions) >= max_questions:
                break
        
        # De-duplicate and trim
        dedup = []
        seen = set()
        for q in questions:
            qn = q.lower()
            if qn not in seen:
                seen.add(qn)
                dedup.append(q)
        return dedup[:max_questions]

    def _answer(self, question: str, context: str) -> str:
        """Answer a question given context."""
        context = sanitize(context)
        if not question or not context:
            return ""
        try:
            out = self.qa_pipe({"question": question, "context": context})
            if isinstance(out, list) and out:
                out = out[0]
            ans = out.get("answer", "") if isinstance(out, dict) else ""
            return ans.strip()
        except Exception:
            return ""

    def score(self, source: str, summary: str, max_questions: int = 5) -> float:
        """Calculate QAGS score for source-summary pair."""
        summary = sanitize(summary)
        source = sanitize(source)
        if not summary or not source:
            return 0.0

        questions = self._gen_questions(summary, max_questions=max_questions)
        if not questions:
            return 0.0

        f1s = []
        for q in questions:
            ans_src = self._answer(q, source)
            ans_sum = self._answer(q, summary)
            f1s.append(simple_token_f1(ans_src, ans_sum))
        
        if not f1s:
            return 0.0
        return float(sum(f1s) / len(f1s))


# ----------------------------
# JSON Processing
# ----------------------------

def process_json(records: List[Dict[str, Any]],
                 factcc: FactCCScorer,
                 qags: QAGSLite,
                 max_source_tokens: int = 512,
                 max_summary_tokens: int = 256,
                 qags_max_questions: int = 5) -> List[Dict[str, Any]]:
    """
    Augment each record with FactCC and QAGS scores.
    Handles multiple key variants and raises a clear error if a required field is missing.
    """
    for idx, item in enumerate(records):
        # Flexible key matching
        source = pick_first(
            item.get("input"),
            item.get("text"), 
            item.get("source")
        )
        
        reference = pick_first(
            item.get("reference"),
            item.get("reference_summary")
        )

        # For before/after style data
        before = pick_first(
            item.get("generatedsummary_before"),
            item.get("generated_before"),
            item.get("gen_before"),
            item.get("summary_before"),
            item.get("generated_summary")
        )

        after = pick_first(
            item.get("generatedsummary_after"),
            item.get("generated_after"),
            item.get("gen_after"),
            item.get("summary_after"),
            item.get("generated_finetuned")
        )

        # Check required fields
        if source is None:
            print(f"Warning: Record {idx} missing source text (checked: input, text, source)")
            continue
            
        # If we have before/after summaries, process both
        if before is not None and after is not None:
            try:
                item["factcc_before"] = factcc.score(source, before,
                                                   max_source_tokens=max_source_tokens,
                                                   max_summary_tokens=max_summary_tokens)
            except Exception as e:
                print(f"Warning: FactCC failed for record {idx} (before): {e}")
                item["factcc_before"] = 0.0

            try:
                item["factcc_after"] = factcc.score(source, after,
                                                  max_source_tokens=max_source_tokens,
                                                  max_summary_tokens=max_summary_tokens)
            except Exception as e:
                print(f"Warning: FactCC failed for record {idx} (after): {e}")
                item["factcc_after"] = 0.0

            try:
                item["qags_before"] = qags.score(source, before, max_questions=qags_max_questions)
            except Exception as e:
                print(f"Warning: QAGS failed for record {idx} (before): {e}")
                item["qags_before"] = 0.0

            try:
                item["qags_after"] = qags.score(source, after, max_questions=qags_max_questions)
            except Exception as e:
                print(f"Warning: QAGS failed for record {idx} (after): {e}")
                item["qags_after"] = 0.0
        
        # If we only have a single summary (check various possible fields)
        else:
            summary = pick_first(
                item.get("summary"),
                item.get("generated_summary"),
                item.get("output"),
                before,  # might be the only summary
                after    # might be the only summary
            )
            
            if summary is not None:
                try:
                    item["factcc_score"] = factcc.score(source, summary,
                                                      max_source_tokens=max_source_tokens,
                                                      max_summary_tokens=max_summary_tokens)
                except Exception as e:
                    print(f"Warning: FactCC failed for record {idx}: {e}")
                    item["factcc_score"] = 0.0

                try:
                    item["qags_score"] = qags.score(source, summary, max_questions=qags_max_questions)
                except Exception as e:
                    print(f"Warning: QAGS failed for record {idx}: {e}")
                    item["qags_score"] = 0.0

    return records


def parse_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a JSON file and return the data."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON file {file_path} must contain a list of objects")
        return data
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def find_json_files(directory: str = ".") -> List[str]:
    """Find all JSON files in the given directory."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    return sorted(json_files)


# ----------------------------
# Main CLI
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Compute FactCC and QAGS scores for JSON files")
    parser.add_argument("--input_path", type=str, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, help="Path to output JSON file")
    parser.add_argument("--process_all", action="store_true", 
                       help="Process all JSON files in the repository")
    parser.add_argument("--factcc_model", default="tals/albert-base-v2-factcc",
                       help="HuggingFace model for FactCC scoring")
    parser.add_argument("--qg_model", default="valhalla/t5-small-qg-hl",
                       help="HuggingFace model for question generation")
    parser.add_argument("--qa_model", default="deepset/roberta-base-squad2",
                       help="HuggingFace model for question answering")
    parser.add_argument("--device", default="auto",
                       help="Device: 'auto', 'cpu', or CUDA device index")
    parser.add_argument("--max_source_tokens", type=int, default=448,
                       help="Max tokens for source text")
    parser.add_argument("--max_summary_tokens", type=int, default=192,
                       help="Max tokens for summary text")
    parser.add_argument("--qags_max_questions", type=int, default=5,
                       help="Max questions for QAGS scoring")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🚀 Initializing FactCC and QAGS scorers...")
    
    # Initialize scorers
    factcc = FactCCScorer(model_name=args.factcc_model, device=args.device)
    qags = QAGSLite(qg_model=args.qg_model, qa_model=args.qa_model, device=args.device)
    
    print("✅ Models loaded successfully!")
    
    if args.process_all:
        # Process all JSON files in repository
        json_files = find_json_files()
        if not json_files:
            print("❌ No JSON files found in the current directory")
            return
            
        print(f"📁 Found {len(json_files)} JSON files:")
        for f in json_files:
            print(f"  - {f}")
        
        for file_path in json_files:
            print(f"\n📊 Processing {file_path}...")
            data = parse_json_file(file_path)
            if not data:
                continue
                
            # Process the data
            scored_data = process_json(
                records=data,
                factcc=factcc,
                qags=qags,
                max_source_tokens=args.max_source_tokens,
                max_summary_tokens=args.max_summary_tokens,
                qags_max_questions=args.qags_max_questions,
            )
            
            # Save scored data
            base_name = os.path.splitext(file_path)[0]
            output_path = f"{base_name}_scored.json"
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(scored_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved scored data to: {output_path}")
    
    else:
        # Process single file
        if not args.input_path:
            print("❌ Please provide --input_path or use --process_all")
            return
            
        if not os.path.exists(args.input_path):
            print(f"❌ File not found: {args.input_path}")
            return
            
        print(f"📊 Processing {args.input_path}...")
        data = parse_json_file(args.input_path)
        if not data:
            return
            
        # Process the data
        scored_data = process_json(
            records=data,
            factcc=factcc,
            qags=qags,
            max_source_tokens=args.max_source_tokens,
            max_summary_tokens=args.max_summary_tokens,
            qags_max_questions=args.qags_max_questions,
        )
        
        # Determine output path
        if args.output_path:
            output_path = args.output_path
        else:
            base_name = os.path.splitext(args.input_path)[0]
            output_path = f"{base_name}_scored.json"
        
        # Save scored data
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scored_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved scored data to: {output_path}")


if __name__ == "__main__":
    main()