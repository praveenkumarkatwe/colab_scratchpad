#!/usr/bin/env python3
"""
Analyze JSON files in the repository to understand their structure.
"""

import json
import glob
import os
from collections import Counter
from typing import Dict, Any, List


def analyze_json_structure(file_path: str) -> Dict[str, Any]:
    """Analyze the structure of a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return {"error": "JSON is not a list"}
        
        if not data:
            return {"error": "JSON list is empty"}
        
        # Analyze first record to understand structure
        first_record = data[0]
        
        # Count different key patterns
        all_keys = set()
        for record in data[:10]:  # Sample first 10 records
            if isinstance(record, dict):
                all_keys.update(record.keys())
        
        # Analyze sample content lengths
        sample_record = data[0] if data else {}
        
        analysis = {
            "file": os.path.basename(file_path),
            "total_records": len(data),
            "all_keys": sorted(list(all_keys)),
            "sample_record_keys": list(sample_record.keys()) if isinstance(sample_record, dict) else [],
            "sample_content": {}
        }
        
        # Get sample content for common fields
        if isinstance(sample_record, dict):
            for key in ["id", "input", "text", "source", "reference", "generatedsummary_before", "generatedsummary_after"]:
                if key in sample_record:
                    content = str(sample_record[key])
                    analysis["sample_content"][key] = {
                        "length": len(content),
                        "preview": content[:100] + "..." if len(content) > 100 else content
                    }
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}


def main():
    """Analyze all JSON files in the repository."""
    json_files = glob.glob("*.json")
    json_files = [f for f in json_files if not f.endswith("_scored.json")]  # Exclude already processed files
    
    print("📁 JSON Files Analysis")
    print("=" * 50)
    
    for file_path in sorted(json_files):
        print(f"\n📊 {file_path}")
        print("-" * 30)
        
        analysis = analyze_json_structure(file_path)
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            continue
        
        print(f"Records: {analysis['total_records']}")
        print(f"Keys: {', '.join(analysis['all_keys'])}")
        
        if analysis['sample_content']:
            print("\nSample content:")
            for key, info in analysis['sample_content'].items():
                print(f"  {key}: {info['length']} chars - {info['preview']}")
    
    print(f"\n✅ Analysis complete! Found {len(json_files)} JSON files.")


if __name__ == "__main__":
    main()