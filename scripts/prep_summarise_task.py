import re, textwrap, sys, os
from typing import List
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prep_utils import stream_cap, dump

# Patterns for legal document structure
SYLLABUS_PATTERN = re.compile(r'(Syllabus|SYLLABUS)(.*?)(?:Opinion|OPINION|Held|HELD)', re.S|re.I)
PROCEDURAL_PATTERN = re.compile(r'(Appeal|Petition|Writ|Motion)(.*?)(?:Facts|FACTS|Background|BACKGROUND)', re.S|re.I)

def extract_parties(case_name: str) -> List[str]:
    """Extract party names from case name"""
    # Remove common legal words and split on 'v.'
    if ' v. ' in case_name:
        parties = case_name.split(' v. ')
    elif ' v ' in case_name:
        parties = case_name.split(' v ')
    else:
        return []
    
    result = []
    for party in parties[:2]:  # Take first two parties
        # Clean up party name
        clean_party = re.sub(r'[,\.].*$', '', party.strip())
        words = clean_party.split()
        # Take capitalized words (proper nouns)
        party_words = [w for w in words if w and w[0].isupper()]
        if party_words:
            result.append(' '.join(party_words[:3]))  # Max 3 words per party
    
    return result

def create_summary_prompt(text: str, max_length: int = 2048) -> str:
    """Create IRAC summary prompt from case text"""
    # Truncate text to manageable length
    truncated = text[:max_length]
    
    return f"Summarize this legal case using IRAC format (Issue, Rule, Application, Conclusion) in ≤200 words:\n\n{truncated}"

print("Starting IRAC summarization task preparation...")

summary_count = 0
for idx, rec in enumerate(stream_cap()):
    if idx % 1000 == 0:
        print(f"Processed {idx} records, created {summary_count} summaries...")
    
    casebody = rec['casebody']
    case_name = rec['name']
    
    # Skip very short cases
    if len(casebody.split()) < 100:
        continue
    
    # Look for syllabus or structured content
    syllabus_match = SYLLABUS_PATTERN.search(casebody)
    ground_truth_content = ""
    
    if syllabus_match:
        syllabus_text = syllabus_match.group(2).strip()
        ground_truth_content = textwrap.shorten(syllabus_text, 2000, placeholder="...")
    else:
        # Use first substantial paragraph as ground truth
        paragraphs = [p.strip() for p in casebody.split('\n\n') if len(p.strip().split()) > 20]
        if paragraphs:
            ground_truth_content = textwrap.shorten(paragraphs[0], 2000, placeholder="...")
    
    if not ground_truth_content or len(ground_truth_content.split()) < 10:
        continue
    
    # Extract key information
    parties = extract_parties(case_name)
    year = rec['decision_date'][:4] if rec['decision_date'] else "Unknown"
    
    summary_prompt = create_summary_prompt(casebody)
    
    rec_out = {
        'case_id': rec['case_id'],
        'case_name': case_name,
        'inputs': summary_prompt,
        'ground_truth': {
            'summary': ground_truth_content,
            'key_parties': parties,
            'year': year,
            'case_name': case_name
        }
    }
    
    dump('summarise', rec_out, summary_count)
    summary_count += 1

print(f"Generated {summary_count} IRAC summarization tasks")