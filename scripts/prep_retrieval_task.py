import re, sys, os
from collections import defaultdict
from typing import List
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prep_utils import stream_cap, dump

def extract_legal_concepts(text: str) -> List[str]:
    """Extract legal concepts and topics from case text"""
    concepts = []
    
    # Common legal concept patterns
    legal_terms = [
        r'constitutional?\s+(?:right|issue|question|challenge|violation)',
        r'due\s+process',
        r'equal\s+protection',
        r'first\s+amendment',
        r'fourth\s+amendment',
        r'fifth\s+amendment',
        r'commerce\s+clause',
        r'criminal\s+(?:law|procedure|defense)',
        r'civil\s+(?:rights|procedure|liability)',
        r'contract(?:ual)?\s+(?:law|dispute|breach|interpretation)',
        r'tort\s+(?:law|liability|claim)',
        r'property\s+(?:law|rights|dispute)',
        r'evidence\s+(?:law|admissibility|hearsay)',
        r'jurisdiction(?:al)?\s+(?:issue|question|dispute)',
        r'standing\s+(?:to\s+sue|issue)',
        r'summary\s+judgment',
        r'class\s+action',
        r'habeas\s+corpus',
        r'injunctive\s+relief',
        r'damages\s+(?:award|calculation)',
        r'negligence\s+(?:claim|standard)',
        r'strict\s+liability',
        r'antitrust\s+(?:law|violation)',
        r'securities\s+(?:law|fraud|regulation)',
        r'tax\s+(?:law|liability|evasion)',
        r'bankruptcy\s+(?:law|proceeding|discharge)',
        r'employment\s+(?:law|discrimination|termination)',
        r'environmental\s+(?:law|regulation|violation)',
        r'intellectual\s+property\s+(?:law|infringement)',
        r'family\s+law\s+(?:matter|dispute)',
        r'immigration\s+(?:law|deportation|asylum)',
    ]
    
    text_lower = text.lower()
    for pattern in legal_terms:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        concepts.extend([match.replace('\s+', ' ') for match in matches])
    
    # Extract court types as concepts
    court_patterns = [
        r'supreme\s+court', r'circuit\s+court', r'district\s+court',
        r'bankruptcy\s+court', r'tax\s+court', r'claims\s+court'
    ]
    
    for pattern in court_patterns:
        if re.search(pattern, text_lower):
            concepts.append(pattern.replace(r'\s+', ' '))
    
    # Deduplicate and return
    return list(set(concepts))

def extract_case_facts(text: str, max_length: int = 1500) -> str:
    """Extract factual background from case"""
    # Look for fact patterns
    fact_indicators = [
        r'facts?\s*:?\s*\n',
        r'background\s*:?\s*\n', 
        r'procedural\s+history\s*:?\s*\n',
        r'plaintiff\s+(?:alleges?|claims?)',
        r'defendant\s+(?:argues?|contends?)',
        r'the\s+facts?\s+(?:are|show|indicate|establish)'
    ]
    
    for pattern in fact_indicators:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            # Extract following content
            following_text = text[start_pos:start_pos + max_length]
            # Stop at next major section
            section_breaks = [r'\n\n[A-Z][A-Z\s]+\n', r'\n\nII\.', r'\n\nB\.', r'\n\nAnalysis', r'\n\nDiscussion']
            for break_pattern in section_breaks:
                break_match = re.search(break_pattern, following_text)
                if break_match:
                    following_text = following_text[:break_match.start()]
                    break
            return following_text.strip()
    
    # Fallback: use first substantial paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip().split()) > 30]
    if paragraphs:
        return paragraphs[0][:max_length]
    
    return text[:max_length]

print("Starting retrieval task preparation...")

# Build concept index
concept_index = defaultdict(list)
all_records = []

print("Building concept index...")
for i, rec in enumerate(stream_cap()):
    if i % 1000 == 0:
        print(f"Indexing record {i}...")
    
    concepts = extract_legal_concepts(rec['casebody'])
    if not concepts:  # Skip cases without clear legal concepts
        continue
        
    all_records.append(rec)
    
    for concept in concepts:
        concept_index[concept].append(len(all_records) - 1)  # Store index

print(f"Indexed {len(all_records)} records with {len(concept_index)} unique concepts")

# Generate retrieval tasks
retrieval_count = 0
for idx, rec in enumerate(all_records):
    if idx % 500 == 0:
        print(f"Generating retrieval tasks: {idx}/{len(all_records)}")
    
    concepts = extract_legal_concepts(rec['casebody'])
    if not concepts:
        continue
    
    # Find related cases based on shared concepts
    related_indices = set()
    for concept in concepts:
        if concept in concept_index:
            related_indices.update(concept_index[concept])
    
    # Remove self and ensure we have enough related cases
    related_indices.discard(idx)
    if len(related_indices) < 3:
        continue
    
    # Extract case facts for the query
    facts = extract_case_facts(rec['casebody'])
    if len(facts.split()) < 20:
        continue
    
    # Build list of related case names and IDs
    related_cases = []
    for rel_idx in list(related_indices)[:10]:  # Limit to top 10
        rel_rec = all_records[rel_idx]
        related_cases.append({
            'case_id': rel_rec['case_id'],
            'case_name': rel_rec['name'],
            'shared_concepts': [c for c in concepts if c in extract_legal_concepts(rel_rec['casebody'])]
        })
    
    rec_out = {
        'case_id': rec['case_id'],
        'case_name': rec['name'], 
        'inputs': f'Find 5 cases with similar legal issues and facts:\n\nFacts: {facts}',
        'query_concepts': concepts,
        'related_cases': related_cases,
        'positives': [rc['case_id'] for rc in related_cases]
    }
    
    dump('retrieval', rec_out, retrieval_count)
    retrieval_count += 1
    
    # Incremental writing every 1000 records to prevent data loss
    if retrieval_count % 1000 == 0:
        print(f"Processed {retrieval_count} retrieval tasks, writing checkpoint...")
        # Force flush any pending writes
        sys.stdout.flush()

print(f"Generated {retrieval_count} retrieval tasks")