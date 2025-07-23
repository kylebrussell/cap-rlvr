import re, sys, os, json, pickle
from collections import defaultdict
from typing import List, Dict, Set
import tempfile
import sqlite3
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

class StreamingRetrievalProcessor:
    """Memory-efficient retrieval task processor using SQLite for indexing"""
    
    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'retrieval_index.db')
        self.init_database()
        self.processed_count = 0
        
    def init_database(self):
        """Initialize SQLite database for concept indexing"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Table for case metadata (minimal info)
        self.cursor.execute('''
            CREATE TABLE cases (
                case_idx INTEGER PRIMARY KEY,
                case_id TEXT,
                case_name TEXT,
                concepts TEXT  -- JSON string of concepts
            )
        ''')
        
        # Table for concept -> case mapping
        self.cursor.execute('''
            CREATE TABLE concept_index (
                concept TEXT,
                case_idx INTEGER,
                FOREIGN KEY (case_idx) REFERENCES cases(case_idx)
            )
        ''')
        
        # Index for fast concept lookups
        self.cursor.execute('CREATE INDEX idx_concept ON concept_index(concept)')
        
        self.conn.commit()
        
    def add_case_to_index(self, case_idx: int, rec: dict):
        """Add case to SQLite index"""
        concepts = extract_legal_concepts(rec['casebody'])
        if not concepts:
            return False
            
        # Insert case metadata
        self.cursor.execute('''
            INSERT INTO cases (case_idx, case_id, case_name, concepts)
            VALUES (?, ?, ?, ?)
        ''', (case_idx, rec['case_id'], rec['name'], json.dumps(concepts)))
        
        # Insert concept mappings
        for concept in concepts:
            self.cursor.execute('''
                INSERT INTO concept_index (concept, case_idx)
                VALUES (?, ?)
            ''', (concept, case_idx))
        
        return True
        
    def find_related_cases(self, concepts: List[str], exclude_idx: int, limit: int = 10) -> List[dict]:
        """Find cases with matching concepts"""
        if not concepts:
            return []
            
        # Build query for cases sharing concepts
        placeholders = ','.join(['?' for _ in concepts])
        query = f'''
            SELECT DISTINCT c.case_idx, c.case_id, c.case_name, c.concepts,
                   COUNT(ci.concept) as shared_count
            FROM cases c
            JOIN concept_index ci ON c.case_idx = ci.case_idx
            WHERE ci.concept IN ({placeholders}) AND c.case_idx != ?
            GROUP BY c.case_idx, c.case_id, c.case_name, c.concepts
            ORDER BY shared_count DESC
            LIMIT ?
        '''
        
        params = concepts + [exclude_idx, limit]
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        
        related_cases = []
        for case_idx, case_id, case_name, concepts_json, shared_count in results:
            case_concepts = json.loads(concepts_json)
            shared_concepts = [c for c in concepts if c in case_concepts]
            
            related_cases.append({
                'case_id': case_id,
                'case_name': case_name,
                'shared_concepts': shared_concepts,
                'shared_count': shared_count
            })
        
        return related_cases
        
    def commit_batch(self):
        """Commit current batch to database"""
        self.conn.commit()
        
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

def process_retrieval_streaming():
    """Process retrieval tasks with minimal memory usage"""
    print("Starting streaming retrieval task preparation...")
    
    processor = StreamingRetrievalProcessor()
    
    try:
        # Phase 1: Build index by streaming through data once
        print("Phase 1: Building concept index...")
        case_idx = 0
        valid_cases = 0
        
        for i, rec in enumerate(stream_cap()):
            if i % 1000 == 0:
                print(f"Indexing record {i}... (valid cases: {valid_cases})")
                processor.commit_batch()  # Periodic commits to prevent memory buildup
            
            if processor.add_case_to_index(case_idx, rec):
                valid_cases += 1
                
            case_idx += 1
            
            # Memory safety: limit processing for very large datasets
            if case_idx > 500000:  # Process first 500K records
                print(f"Reached processing limit ({case_idx} records)")
                break
        
        processor.commit_batch()
        print(f"Indexed {valid_cases} valid records from {case_idx} total")
        
        # Phase 2: Generate retrieval tasks by streaming again
        print("Phase 2: Generating retrieval tasks...")
        retrieval_count = 0
        case_idx = 0
        
        for i, rec in enumerate(stream_cap()):
            if i % 500 == 0:
                print(f"Generating tasks: {i} (created: {retrieval_count})")
            
            concepts = extract_legal_concepts(rec['casebody'])
            if not concepts:
                case_idx += 1
                continue
                
            # Find related cases
            related_cases = processor.find_related_cases(concepts, case_idx, limit=10)
            if len(related_cases) < 3:
                case_idx += 1
                continue
            
            # Extract case facts
            facts = extract_case_facts(rec['casebody'])
            if len(facts.split()) < 20:
                case_idx += 1
                continue
            
            # Create retrieval task
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
            case_idx += 1
            
            # Stop early to match original limit
            if case_idx > 500000:
                break
        
        print(f"Generated {retrieval_count} retrieval tasks")
        
    finally:
        processor.close()

if __name__ == "__main__":
    process_retrieval_streaming()