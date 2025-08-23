#!/usr/bin/env python3
"""
Test script to demonstrate JSON parsing functionality without requiring model downloads.
"""

import json
import os
import sys
import glob
from typing import Any, Dict, List


def pick_first(*vals):
    """Return the first non-empty value from provided candidates."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def analyze_record_structure(item: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze what fields are available in a record for FactCC/QAGS processing."""
    
    # Check for source text
    source = pick_first(
        item.get("input"),
        item.get("text"), 
        item.get("source")
    )
    
    # Check for reference
    reference = pick_first(
        item.get("reference"),
        item.get("reference_summary")
    )

    # Check for before/after summaries
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
    
    # Check for single summary
    summary = pick_first(
        item.get("summary"),
        item.get("generated_summary"),
        item.get("output"),
        before,  # might be the only summary
        after    # might be the only summary
    )
    
    return {
        "has_source": source is not None,
        "has_reference": reference is not None,
        "has_before": before is not None,
        "has_after": after is not None,
        "has_summary": summary is not None,
        "source_length": len(source) if source else 0,
        "before_length": len(before) if before else 0,
        "after_length": len(after) if after else 0,
        "can_process_before_after": source is not None and before is not None and after is not None,
        "can_process_single": source is not None and summary is not None,
        "sample_source": source[:100] + "..." if source and len(source) > 100 else source,
        "sample_before": before[:100] + "..." if before and len(before) > 100 else before,
        "sample_after": after[:100] + "..." if after and len(after) > 100 else after,
    }


def test_json_file(file_path: str):
    """Test parsing of a single JSON file."""
    print(f"\n📊 Testing {os.path.basename(file_path)}")
    print("-" * 50)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ Error: JSON is not a list")
            return
        
        if not data:
            print("❌ Error: JSON list is empty")
            return
        
        print(f"✅ Total records: {len(data)}")
        
        # Analyze first few records
        processable_before_after = 0
        processable_single = 0
        
        for i, record in enumerate(data[:5]):  # Check first 5 records
            analysis = analyze_record_structure(record)
            
            print(f"\nRecord {i + 1}:")
            print(f"  Source: {'✅' if analysis['has_source'] else '❌'} ({analysis['source_length']} chars)")
            print(f"  Before: {'✅' if analysis['has_before'] else '❌'} ({analysis['before_length']} chars)")
            print(f"  After:  {'✅' if analysis['has_after'] else '❌'} ({analysis['after_length']} chars)")
            print(f"  Can process before/after: {'✅' if analysis['can_process_before_after'] else '❌'}")
            print(f"  Can process single: {'✅' if analysis['can_process_single'] else '❌'}")
            
            if analysis['sample_source']:
                print(f"  Source preview: {analysis['sample_source']}")
            if analysis['sample_before']:
                print(f"  Before preview: {analysis['sample_before']}")
            if analysis['sample_after']:
                print(f"  After preview: {analysis['sample_after']}")
        
        # Count how many records can be processed
        for record in data:
            analysis = analyze_record_structure(record)
            if analysis['can_process_before_after']:
                processable_before_after += 1
            if analysis['can_process_single']:
                processable_single += 1
        
        print(f"\n📈 Processing Summary:")
        print(f"  Records processable for before/after: {processable_before_after}/{len(data)} ({100*processable_before_after/len(data):.1f}%)")
        print(f"  Records processable for single summary: {processable_single}/{len(data)} ({100*processable_single/len(data):.1f}%)")
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")


def main():
    """Test JSON parsing functionality on all repository files."""
    print("🧪 JSON Parsing Test for FactCC/QAGS Processing")
    print("=" * 60)
    
    # Find JSON files
    json_files = glob.glob("*.json")
    json_files = [f for f in json_files if not f.endswith("_scored.json")]  # Exclude processed files
    
    print(f"Found {len(json_files)} JSON files to test:")
    for f in json_files:
        print(f"  - {f}")
    
    # Test each file
    for file_path in sorted(json_files):
        test_json_file(file_path)
    
    print(f"\n✅ Testing complete!")
    print("\n📝 Notes:")
    print("  - ✅ means the field is present and non-empty")
    print("  - ❌ means the field is missing or empty")
    print("  - 'Before/after' processing requires source + before + after summaries")
    print("  - 'Single' processing requires source + any summary")
    print("\n🚀 To run FactCC/QAGS scoring (requires internet for model download):")
    print("  python compute_factcc_qags_scores.py --process_all")


if __name__ == "__main__":
    main()