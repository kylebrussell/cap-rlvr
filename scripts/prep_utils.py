from typing import Iterator, Dict, List
import json, pathlib, random, gzip
import re

# Adapted for the CAP dataset format from common-pile
CAP_PATH = pathlib.Path('../data_raw/cap_raw')

_SPLIT = {
    'train': lambda i: i % 10 not in (8, 9),
    'eval':  lambda i: i % 10 == 8,
    'test':  lambda i: i % 10 == 9,
}

def stream_cap() -> Iterator[Dict]:
    """Stream CAP records from compressed JSONL files"""
    cap_files = sorted(CAP_PATH.glob('cap_*.jsonl.gz'))
    print(f"Found {len(cap_files)} CAP files to process")
    
    for cap_file in cap_files:
        print(f"Processing {cap_file.name}...")
        with gzip.open(cap_file, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    # Convert to expected format
                    yield {
                        'case_id': record['id'],
                        'casebody': record['text'],
                        'name': extract_case_name(record['text']),
                        'decision_date': extract_date(record['text']),
                        'citation': extract_citation(record['text']),
                        'metadata': record.get('metadata', {})
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing line {line_num} in {cap_file.name}: {e}")
                    continue

def extract_case_name(text: str) -> str:
    """Extract case name from the text"""
    lines = text.split('\n')
    for line in lines[:5]:  # Check first few lines
        line = line.strip()
        if ' v. ' in line or ' v ' in line:
            return line
    return "Unknown Case"

def extract_date(text: str) -> str:
    """Extract decision date from text"""
    # Look for date patterns like "Feb. 12, 1973"
    date_pattern = r'(\w+\.?\s+\d{1,2},?\s+\d{4})'
    lines = text.split('\n')
    for line in lines[:10]:
        match = re.search(date_pattern, line)
        if match:
            return match.group(1)
    return "1900-01-01"  # Default fallback

def extract_citation(text: str) -> Dict:
    """Extract citation info from text"""
    # Look for U.S. citations like "123 U.S. 456 (1999)"
    us_pattern = r'(\d+)\s+U\.S\.\s+(\d+)\s+\((\d{4})\)'
    match = re.search(us_pattern, text)
    if match:
        return {
            'volume': match.group(1),
            'page': match.group(2), 
            'year': match.group(3),
            'reporter': 'U.S.'
        }
    return {}

def dump(task: str, rec: Dict, idx: int, out_root='../data_tasks'):
    """Dump record to appropriate split file"""
    for split, cond in _SPLIT.items():
        if cond(idx):
            p = pathlib.Path(out_root)/task
            p.mkdir(parents=True, exist_ok=True)
            with open(p/f'{split}.jsonl','a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")