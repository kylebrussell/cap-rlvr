#!/usr/bin/env python3
"""
Reward function for IRAC summarization task.
"""
import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass  # Downloads might fail in some environments
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not available, using basic tokenization")
    NLTK_AVAILABLE = False


class IRACRewardFunction:
    def __init__(self):
        """Initialize IRAC summarization reward function"""
        self.irac_keywords = {
            'issue': ['issue', 'question', 'problem', 'matter', 'dispute', 'whether', 'concern'],
            'rule': ['rule', 'law', 'statute', 'regulation', 'precedent', 'standard', 'test', 'doctrine', 'principle'],
            'application': ['application', 'analysis', 'applying', 'here', 'facts', 'case', 'circumstances', 'situation'],
            'conclusion': ['conclusion', 'conclude', 'held', 'holding', 'therefore', 'thus', 'result', 'accordingly']
        }
        
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if NLTK_AVAILABLE:
            return word_tokenize(text.lower())
        else:
            return re.findall(r'\b\w+\b', text.lower())
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        else:
            # Basic sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def detect_irac_structure(self, text: str) -> Dict[str, float]:
        """Detect presence of IRAC components in text"""
        text_lower = text.lower()
        sentences = self.extract_sentences(text)
        
        component_scores = {}
        
        for component, keywords in self.irac_keywords.items():
            score = 0.0
            
            # Check for explicit headers/labels
            header_patterns = [
                rf'\b{component}s?\b:',  # "Issue:" or "Issues:"
                rf'\b{component}s?\b\s*-',  # "Issue -"
                rf'^{component}s?\b',  # Starting with component name
                rf'\b{component.upper()}s?\b',  # Uppercase version
            ]
            
            for pattern in header_patterns:
                if re.search(pattern, text_lower, re.MULTILINE):
                    score += 0.4  # Bonus for explicit structure
            
            # Check for keyword presence
            keyword_count = 0
            for keyword in keywords:
                keyword_pattern = rf'\b{keyword}s?\b'
                matches = len(re.findall(keyword_pattern, text_lower))
                keyword_count += matches
            
            # Normalize keyword score
            if keyword_count > 0:
                keyword_score = min(0.6, keyword_count * 0.1)
                score += keyword_score
            
            component_scores[component] = min(1.0, score)
        
        return component_scores
    
    def compute_content_quality(self, response: str, ground_truth: Dict) -> float:
        """Compute content quality based on ground truth information"""
        response_tokens = set(self.tokenize_text(response))
        response_tokens = {token for token in response_tokens if token not in self.stop_words}
        
        # Extract ground truth content
        gt_summary = ground_truth.get('summary', '')
        gt_parties = ground_truth.get('key_parties', [])
        gt_case_name = ground_truth.get('case_name', '')
        
        content_score = 0.0
        
        # Compare with ground truth summary
        if gt_summary:
            gt_tokens = set(self.tokenize_text(gt_summary))
            gt_tokens = {token for token in gt_tokens if token not in self.stop_words}
            
            if len(gt_tokens) > 0:
                overlap = len(response_tokens.intersection(gt_tokens))
                content_score += (overlap / len(gt_tokens)) * 0.5
        
        # Check for party names
        party_score = 0.0
        if gt_parties:
            for party in gt_parties:
                party_tokens = set(self.tokenize_text(party))
                party_tokens = {token for token in party_tokens if token not in self.stop_words}
                
                if len(party_tokens.intersection(response_tokens)) > 0:
                    party_score += 1.0 / len(gt_parties)
        
        content_score += party_score * 0.2
        
        # Check for case name mention
        if gt_case_name:
            case_tokens = set(self.tokenize_text(gt_case_name))
            case_tokens = {token for token in case_tokens if token not in self.stop_words}
            
            if len(case_tokens.intersection(response_tokens)) > 0:
                content_score += 0.1
        
        return min(1.0, content_score)
    
    def evaluate_summary_length(self, text: str, target_words: int = 200) -> float:
        """Evaluate if summary length is appropriate"""
        words = self.tokenize_text(text)
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Ideal range: 150-250 words for 200-word target
        min_words = int(target_words * 0.75)  # 150
        max_words = int(target_words * 1.25)  # 250
        
        if min_words <= word_count <= max_words:
            return 1.0
        elif word_count < min_words:
            # Too short penalty
            return word_count / min_words
        else:
            # Too long penalty
            excess = word_count - max_words
            penalty = excess / target_words  # Penalty for each 200 words over
            return max(0.3, 1.0 - penalty)
    
    def evaluate_legal_language(self, text: str) -> float:
        """Evaluate use of appropriate legal language"""
        legal_indicators = [
            r'\bplaintiff\b', r'\bdefendant\b', r'\bcourt\b', r'\bjudge\b',
            r'\bheld\b', r'\bholding\b', r'\bruling\b', r'\bdecision\b',
            r'\bstatute\b', r'\bregulation\b', r'\bprecedent\b', r'\bcase\s+law\b',
            r'\bliable\b', r'\bliability\b', r'\bdamages\b', r'\binjunction\b',
            r'\bconstitutional\b', r'\bdue\s+process\b', r'\bequal\s+protection\b',
            r'\bjurisdiction\b', r'\bappeal\b', r'\bremand\b', r'\breverse\b'
        ]
        
        text_lower = text.lower()
        legal_term_count = 0
        
        for pattern in legal_indicators:
            matches = len(re.findall(pattern, text_lower))
            legal_term_count += matches
        
        # Normalize based on text length
        sentences = self.extract_sentences(text)
        if len(sentences) == 0:
            return 0.0
        
        legal_density = legal_term_count / len(sentences)
        return min(1.0, legal_density / 2.0)  # Target: ~2 legal terms per sentence
    
    def reward(self, sample: Dict, model_output: str) -> float:
        """
        Main reward function for IRAC summarization.
        
        Args:
            sample: Dictionary with 'ground_truth' containing summary info
            model_output: Model's IRAC summary response
            
        Returns:
            Float reward between 0.0 and 1.0
        """
        try:
            if not model_output or not model_output.strip():
                return 0.0
            
            ground_truth = sample.get('ground_truth', {})
            
            # Component 1: IRAC structure detection (40%)
            irac_scores = self.detect_irac_structure(model_output)
            structure_score = sum(irac_scores.values()) / 4.0  # Average of 4 components
            
            # Component 2: Content quality vs ground truth (30%)
            content_score = self.compute_content_quality(model_output, ground_truth)
            
            # Component 3: Appropriate length (15%)
            length_score = self.evaluate_summary_length(model_output)
            
            # Component 4: Legal language usage (15%)
            legal_score = self.evaluate_legal_language(model_output)
            
            # Weighted combination
            total_score = (
                0.40 * structure_score +
                0.30 * content_score +
                0.15 * length_score +
                0.15 * legal_score
            )
            
            return min(1.0, total_score)
            
        except Exception as e:
            print(f"Error computing IRAC reward: {e}")
            return 0.0


def test_irac_reward():
    """Test the IRAC summarization reward function"""
    reward_fn = IRACRewardFunction()
    
    # Test case
    sample = {
        'ground_truth': {
            'summary': 'The plaintiff sued defendant for breach of contract. The court held that the contract was valid and enforceable under state law.',
            'key_parties': ['Plaintiff', 'Defendant'],
            'year': '1995',
            'case_name': 'Smith v. Jones'
        }
    }
    
    # Test responses
    responses = [
        # Good IRAC summary
        """
        Issue: Whether the defendant breached the contract with plaintiff.
        Rule: Under state law, contracts are enforceable if they contain valid consideration and mutual assent.
        Application: Here, the facts show that plaintiff and defendant had a valid agreement with consideration. The defendant failed to perform as agreed.
        Conclusion: The court held that defendant breached the contract and plaintiff is entitled to damages.
        """,
        
        # Missing structure but good content
        """
        The plaintiff sued defendant for breach of contract. The court determined that the contract was valid under state law because it contained proper consideration and mutual assent. Since defendant failed to perform, plaintiff is entitled to damages.
        """,
        
        # Good structure but poor content
        """
        Issue: What happened in this case?
        Rule: There are legal rules.
        Application: The rules apply to the facts.
        Conclusion: The court made a decision.
        """,
        
        # Too short
        "The defendant breached the contract.",
        
        # Too long (would be much longer in practice)
        "A very long summary that goes on and on..." * 50,
    ]
    
    for i, response in enumerate(responses):
        reward = reward_fn.reward(sample, response)
        print(f"Response {i+1}: Reward = {reward:.3f}")
        print(f"Preview: {response[:100]}...")
        print()


if __name__ == '__main__':
    test_irac_reward()