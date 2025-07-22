import re, sys, os
from collections import defaultdict
from typing import List
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prep_utils import stream_cap, dump

# Legal relationship keywords
RELATIONSHIP_KEYWORDS = {
    'OVERRULES': [
        r'overrule[ds]?',
        r'abrogat(e[ds]?|ion)',
        r'reverse[ds]?\s+and\s+remand',
        r'explicitly\s+overrule',
        r'no\s+longer\s+(?:good\s+law|controlling)',
        r'supersede[ds]?'
    ],
    'DISTINGUISHES': [
        r'distinguish(?:ed|able|ing)?',
        r'(?:factually|legally)\s+distinct',
        r'different\s+(?:facts|circumstances|legal\s+standard)',
        r'inapplicable\s+(?:here|to\s+this\s+case)',
        r'not\s+(?:controlling|applicable|on\s+point)'
    ],
    'AFFIRMS': [
        r'affirm(?:ed|s|ing)?',
        r'uphold[s]?',
        r'confirm[s]?',
        r'(?:consistent|in\s+accord)\s+with',
        r'follow[s]?\s+(?:the\s+)?(?:precedent|holding|rule)',
        r'reaffirm[s]?'
    ],
    'FOLLOWS': [
        r'follow[s]?\s+(?:the\s+)?(?:precedent|holding|decision)',
        r'apply\s+(?:the\s+)?(?:rule|standard|test)',
        r'guided\s+by',
        r'pursuant\s+to',
        r'in\s+accordance\s+with',
        r'adopts?\s+(?:the\s+)?(?:approach|reasoning|standard)'
    ],
    'CITES_POSITIVELY': [
        r'(?:citing|quoting|relying\s+on)',
        r'as\s+(?:held|established|recognized)\s+in',
        r'see\s+(?:also\s+)?[A-Z]',
        r'accord(?:ing\s+to)?',
        r'support(?:ed|ing)\s+(?:by|our\s+holding\s+in)'
    ]
}

def compile_patterns():
    """Compile regex patterns for efficiency"""
    compiled = {}
    for relationship, patterns in RELATIONSHIP_KEYWORDS.items():
        compiled[relationship] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    return compiled

def extract_case_citations(text: str) -> List[str]:
    """Extract case citations from text"""
    citation_patterns = [
        r'([A-Z][A-Za-z\s]+v\.\s+[A-Z][A-Za-z\s]+)',  # Case v. Case
        r'(\d+\s+U\.S\.\s+\d+)',  # U.S. citations
        r'(\d+\s+F\.(?:2d|3d)\s+\d+)',  # Federal citations
        r'(\d+\s+S\.\s?Ct\.\s+\d+)'  # Supreme Court citations
    ]
    
    citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
    
    return list(set(citations))  # Deduplicate

def find_citation_context(text: str, citation: str, window_size: int = 300) -> str:
    """Find context around a citation"""
    # Try to find the citation in text
    citation_pos = text.find(citation)
    if citation_pos == -1:
        # Try case-insensitive search
        citation_lower = citation.lower()
        text_lower = text.lower()
        citation_pos = text_lower.find(citation_lower)
        if citation_pos == -1:
            return ""
    
    # Extract context window
    start = max(0, citation_pos - window_size // 2)
    end = min(len(text), citation_pos + len(citation) + window_size // 2)
    
    return text[start:end].strip()

def classify_relationship(context: str, compiled_patterns: dict) -> str:
    """Classify the relationship between cases based on context"""
    scores = defaultdict(int)
    
    for relationship, patterns in compiled_patterns.items():
        for pattern in patterns:
            matches = pattern.findall(context)
            scores[relationship] += len(matches)
    
    if not scores:
        return 'NONE'
    
    # Return relationship with highest score
    return max(scores.items(), key=lambda x: x[1])[0]

print("Starting entailment/relationship task preparation...")

# Compile regex patterns
compiled_patterns = compile_patterns()

# Process cases and extract relationships
entail_count = 0
citation_cache = {}  # Cache to avoid reprocessing same citations

for idx, rec in enumerate(stream_cap()):
    if idx % 1000 == 0:
        print(f"Processed {idx} records, found {entail_count} relationships...")
    
    casebody = rec['casebody']
    case_id = rec['case_id']
    
    # Extract citations from this case
    citations = extract_case_citations(casebody)
    if not citations:
        continue
    
    # Process each citation
    for citation in citations[:5]:  # Limit to avoid too many per case
        # Create unique pair ID
        pair_id = f"{case_id}__{hash(citation)}"
        
        # Skip if we've seen this citation before
        if citation in citation_cache:
            continue
        citation_cache[citation] = True
        
        # Find context around citation
        context = find_citation_context(casebody, citation)
        if len(context.split()) < 10:  # Skip insufficient context
            continue
        
        # Classify relationship
        relationship = classify_relationship(context, compiled_patterns)
        
        # Skip if no clear relationship found
        if relationship == 'NONE' and entail_count > 1000:  # Allow some NONE examples early on
            continue
        
        rec_out = {
            'pair_id': pair_id,
            'citing_case_id': case_id,
            'citing_case_name': rec['name'],
            'cited_case': citation,
            'inputs': f'What is the relationship between the citing case and the cited case?\n\nContext: {context}',
            'label': relationship,
            'context': context
        }
        
        dump('entail', rec_out, entail_count)
        entail_count += 1

print(f"Generated {entail_count} entailment/relationship classification tasks")