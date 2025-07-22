import re, random, sys, os
from typing import List
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prep_utils import stream_cap, dump

# Updated patterns for CAP format
HOLD_RE = re.compile(r'(Held|Holding|We hold)[:\s]', re.I|re.M)

def extract_sentences(text: str, start_pos: int, max_length: int = 200) -> str:
    """Extract sentence starting from position"""
    # Simple sentence boundary detection
    excerpt = text[start_pos:start_pos + max_length * 2]
    # Find sentence ending
    end_markers = ['. ', '.\n', '? ', '! ']
    min_end = max_length // 2  # Minimum sentence length
    
    for marker in end_markers:
        pos = excerpt.find(marker, min_end)
        if pos > 0:
            return excerpt[:pos + 1].strip()
    
    # Fallback: just take first max_length chars
    return excerpt[:max_length].strip()

print("Starting holding task preparation...")

holds = []
for i, rec in enumerate(stream_cap()):
    if i % 1000 == 0:
        print(f"Processed {i} records, found {len(holds)} holdings...")
    
    casebody = rec['casebody']
    m = HOLD_RE.search(casebody)
    if not m:
        continue
        
    # Extract holding statement
    holding_text = extract_sentences(casebody, m.end())
    if len(holding_text.split()) < 5:  # Skip very short holdings
        continue
        
    year = rec['decision_date'][:4] if rec['decision_date'] else "1900"
    
    holds.append({
        'year': year, 
        'id': rec['case_id'], 
        'txt': holding_text,
        'case_name': rec['name']
    })

print(f"Found {len(holds)} total holdings")

# Generate multiple choice questions
valid_questions = 0
for idx, pos in enumerate(holds):
    if idx % 500 == 0:
        print(f"Generating questions: {idx}/{len(holds)}")
        
    same_year = [h for h in holds if h['year'] == pos['year'] and h['id'] != pos['id']]
    if len(same_year) < 4:
        continue
        
    distractors = random.sample(same_year, 4)
    rec = {
        'case_id': pos['id'],
        'case_name': pos['case_name'],
        'inputs': f"Choose the correct holding from {pos['case_name']}:",
        'choices': [pos['txt']] + [d['txt'] for d in distractors],
        'answer_idx': 0,
        'year': pos['year']
    }
    dump('holding', rec, valid_questions)
    valid_questions += 1

print(f"Generated {valid_questions} holding selection questions")