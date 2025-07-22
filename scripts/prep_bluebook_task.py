import re, sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prep_utils import stream_cap, dump

# Enhanced citation patterns for various reporters
CITATION_PATTERNS = {
    'us': re.compile(r'(\d+)\s+U\.S\.\s+(\d+)\s+\((\d{4})\)'),
    'f2d': re.compile(r'(\d+)\s+F\.2d\s+(\d+)\s+\(([^)]+)\s+(\d{4})\)'),
    'f3d': re.compile(r'(\d+)\s+F\.3d\s+(\d+)\s+\(([^)]+)\s+(\d{4})\)'),
    'fed': re.compile(r'(\d+)\s+F\.\s+(\d+)\s+\(([^)]+)\s+(\d{4})\)'),
    'sct': re.compile(r'(\d+)\s+S\.\s?Ct\.\s+(\d+)\s+\((\d{4})\)')
}

print("Starting Bluebook citation task preparation...")

citation_count = 0
for idx, rec in enumerate(stream_cap()):
    if idx % 1000 == 0:
        print(f"Processed {idx} records, found {citation_count} citations...")
    
    casebody = rec['casebody']
    
    # Check each citation pattern
    for reporter_type, pattern in CITATION_PATTERNS.items():
        matches = pattern.findall(casebody)
        
        for match in matches:
            if reporter_type == 'us':
                volume, page, year = match
                citation_text = f"{volume} U.S. {page} ({year})"
                masked_text = "___ U.S. ___ (___)"
                metadata = {
                    'volume': volume,
                    'page': page, 
                    'year': year,
                    'reporter': 'U.S.'
                }
            elif reporter_type in ['f2d', 'f3d', 'fed']:
                volume, page, court, year = match
                reporter = 'F.2d' if reporter_type == 'f2d' else ('F.3d' if reporter_type == 'f3d' else 'F.')
                citation_text = f"{volume} {reporter} {page} ({court} {year})"
                masked_text = f"___ {reporter} ___ ({court} ___)"
                metadata = {
                    'volume': volume,
                    'page': page,
                    'court': court,
                    'year': year,
                    'reporter': reporter
                }
            elif reporter_type == 'sct':
                volume, page, year = match
                citation_text = f"{volume} S. Ct. {page} ({year})"
                masked_text = "___ S. Ct. ___ (___)"
                metadata = {
                    'volume': volume,
                    'page': page,
                    'year': year,
                    'reporter': 'S. Ct.'
                }
            
            rec_out = {
                'case_id': rec['case_id'],
                'case_name': rec['name'],
                'inputs': f'Fill in the citation: {masked_text}',
                'ground_truth': citation_text,
                'metadata': metadata
            }
            dump('bluebook', rec_out, citation_count)
            citation_count += 1
            
            # Limit to avoid too many citations from same case
            break

print(f"Generated {citation_count} Bluebook citation tasks")