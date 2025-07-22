#!/usr/bin/env python3
"""
Reward function for case relationship/entailment classification task.
"""
import re
from typing import Dict, List, Optional
from difflib import SequenceMatcher


class EntailmentRewardFunction:
    def __init__(self):
        """Initialize case relationship reward function"""
        self.valid_relationships = {
            'OVERRULES', 'DISTINGUISHES', 'AFFIRMS', 'FOLLOWS', 'CITES_POSITIVELY', 'NONE'
        }
        
        # Relationship keywords for fuzzy matching
        self.relationship_aliases = {
            'OVERRULES': ['overrule', 'overruled', 'overruling', 'abrogated', 'superseded', 'reversed'],
            'DISTINGUISHES': ['distinguish', 'distinguished', 'distinguishable', 'different', 'distinct', 'inapplicable'],
            'AFFIRMS': ['affirm', 'affirmed', 'affirming', 'uphold', 'upheld', 'confirm', 'confirmed', 'consistent'],
            'FOLLOWS': ['follow', 'follows', 'followed', 'following', 'apply', 'applies', 'pursuant', 'accordance', 'adopt'],
            'CITES_POSITIVELY': ['cite', 'citing', 'cited', 'quote', 'quoting', 'rely', 'relying', 'support', 'accord'],
            'NONE': ['none', 'no relationship', 'unrelated', 'not applicable', 'unclear']
        }
    
    def extract_relationship_from_response(self, response: str) -> Optional[str]:
        """Extract relationship classification from model response"""
        response_clean = response.strip().upper()
        
        # Direct match with valid relationship names
        for relationship in self.valid_relationships:
            if relationship in response_clean:
                return relationship
        
        # Fuzzy matching with aliases
        response_lower = response.lower()
        best_match = None
        max_matches = 0
        
        for relationship, aliases in self.relationship_aliases.items():
            match_count = 0
            for alias in aliases:
                if alias in response_lower:
                    match_count += 1
            
            if match_count > max_matches:
                max_matches = match_count
                best_match = relationship
        
        # Pattern matching for common response formats
        patterns = [
            r'relationship(?:\s+is)?\s*:?\s*(\w+)',
            r'answer(?:\s+is)?\s*:?\s*(\w+)',
            r'classification(?:\s+is)?\s*:?\s*(\w+)',
            r'the\s+relationship\s+is\s+(\w+)',
            r'this\s+is\s+(?:a|an)\s+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                extracted = match.group(1).upper()
                # Try to map to valid relationship
                for relationship in self.valid_relationships:
                    if relationship.startswith(extracted) or extracted in relationship:
                        return relationship
                # Try fuzzy matching with extracted term
                for relationship, aliases in self.relationship_aliases.items():
                    if extracted.lower() in [alias.upper() for alias in aliases]:
                        return relationship
        
        return best_match
    
    def compute_context_consistency(self, response: str, context: str, predicted_label: str) -> float:
        """Check if predicted relationship is consistent with given context"""
        context_lower = context.lower()
        response_lower = response.lower()
        
        # Define context indicators for each relationship
        context_indicators = {
            'OVERRULES': ['overrule', 'abrogat', 'no longer good law', 'supersede', 'explicitly reject'],
            'DISTINGUISHES': ['distinguish', 'different facts', 'not applicable', 'inapplicable', 'factually distinct'],
            'AFFIRMS': ['affirm', 'uphold', 'consistent with', 'confirm', 'reaffirm'],
            'FOLLOWS': ['follow', 'apply the rule', 'pursuant to', 'in accordance', 'guided by'],
            'CITES_POSITIVELY': ['citing', 'relying on', 'as held in', 'see also', 'accord'],
            'NONE': []  # Absence of clear indicators
        }
        
        if predicted_label not in context_indicators:
            return 0.5  # Neutral score for unknown labels
        
        expected_indicators = context_indicators[predicted_label]
        
        # Count matching indicators in context
        indicator_matches = 0
        for indicator in expected_indicators:
            if indicator in context_lower:
                indicator_matches += 1
        
        # Count contradictory indicators (from other relationships)
        contradictory_matches = 0
        for other_label, other_indicators in context_indicators.items():
            if other_label != predicted_label:
                for indicator in other_indicators:
                    if indicator in context_lower:
                        contradictory_matches += 1
        
        # Score based on supporting vs contradictory evidence
        if len(expected_indicators) == 0:  # NONE case
            return 1.0 if contradictory_matches == 0 else 0.0
        
        support_score = indicator_matches / len(expected_indicators)
        contradiction_penalty = min(0.5, contradictory_matches * 0.1)
        
        return max(0.0, support_score - contradiction_penalty)
    
    def evaluate_response_quality(self, response: str, context: str) -> float:
        """Evaluate quality of reasoning in response"""
        quality_indicators = [
            r'because\b', r'since\b', r'therefore\b', r'thus\b', r'however\b',
            r'on the other hand\b', r'in contrast\b', r'similarly\b', r'unlike\b',
            r'the court\b', r'the case\b', r'the facts\b', r'the holding\b',
            r'precedent\b', r'legal standard\b', r'doctrine\b'
        ]
        
        response_lower = response.lower()
        quality_score = 0.0
        
        # Check for reasoning indicators
        reasoning_count = 0
        for pattern in quality_indicators:
            if re.search(pattern, response_lower):
                reasoning_count += 1
        
        quality_score += min(0.5, reasoning_count * 0.1)  # Up to 0.5 for reasoning
        
        # Check for reference to specific legal concepts from context
        legal_concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.\s+[A-Z][a-z]+', context)
        case_references = 0
        for concept in legal_concepts[:3]:  # Check first 3 case names
            if concept.lower() in response_lower:
                case_references += 1
        
        if legal_concepts:
            quality_score += (case_references / min(len(legal_concepts), 3)) * 0.3
        
        # Length appropriateness (not too short, not too verbose)
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            quality_score += 0.2
        elif 5 <= word_count < 10 or 100 < word_count <= 200:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def reward(self, sample: Dict, model_output: str) -> float:
        """
        Main reward function for case relationship classification.
        
        Args:
            sample: Dictionary with 'label' (ground truth) and 'context'
            model_output: Model's response with relationship classification
            
        Returns:
            Float reward between 0.0 and 1.0
        """
        try:
            ground_truth_label = sample['label']
            context = sample.get('context', '')
            
            if not model_output or not model_output.strip():
                return 0.0
            
            # Extract predicted relationship
            predicted_label = self.extract_relationship_from_response(model_output)
            
            if not predicted_label:
                return 0.1  # Small reward for attempting to respond
            
            # Component 1: Exact match reward (60%)
            exact_match_score = 1.0 if predicted_label == ground_truth_label else 0.0
            
            # Component 2: Context consistency (25%)
            context_score = self.compute_context_consistency(model_output, context, predicted_label)
            
            # Component 3: Response quality (15%)
            quality_score = self.evaluate_response_quality(model_output, context)
            
            # Partial credit for related relationships
            partial_credit = 0.0
            if exact_match_score == 0.0:
                # Give partial credit for reasonable confusions
                reasonable_confusions = {
                    ('AFFIRMS', 'FOLLOWS'): 0.3,
                    ('FOLLOWS', 'CITES_POSITIVELY'): 0.2,
                    ('CITES_POSITIVELY', 'AFFIRMS'): 0.2,
                    ('DISTINGUISHES', 'NONE'): 0.1,
                }
                
                confusion_key = tuple(sorted([ground_truth_label, predicted_label]))
                if confusion_key in reasonable_confusions:
                    partial_credit = reasonable_confusions[confusion_key]
                elif (ground_truth_label, predicted_label) in reasonable_confusions:
                    partial_credit = reasonable_confusions[(ground_truth_label, predicted_label)]
                elif (predicted_label, ground_truth_label) in reasonable_confusions:
                    partial_credit = reasonable_confusions[(predicted_label, ground_truth_label)]
            
            # Combine components
            total_score = (
                0.60 * (exact_match_score + partial_credit) +
                0.25 * context_score +
                0.15 * quality_score
            )
            
            return min(1.0, total_score)
            
        except Exception as e:
            print(f"Error computing entailment reward: {e}")
            return 0.0


def test_entailment_reward():
    """Test the entailment reward function"""
    reward_fn = EntailmentRewardFunction()
    
    # Test cases
    test_cases = [
        {
            'sample': {
                'label': 'OVERRULES',
                'context': 'The court explicitly overruled the previous decision in Smith v. Jones, stating that it was no longer good law.'
            },
            'responses': [
                'OVERRULES',  # Perfect match
                'The relationship is OVERRULES because the court explicitly rejected the prior case.',  # Good reasoning
                'This case overrules the previous decision',  # Correct but different format
                'DISTINGUISHES',  # Wrong classification
                'The court made a decision',  # Vague response
            ]
        },
        {
            'sample': {
                'label': 'CITES_POSITIVELY',
                'context': 'The court cited Johnson v. Smith in support of its holding, relying on that case as precedent.'
            },
            'responses': [
                'CITES_POSITIVELY',  # Perfect
                'This is a positive citation for support',  # Reasonable interpretation
                'FOLLOWS',  # Related but different
                'NONE'  # Wrong
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        sample = test_case['sample']
        print(f"\nTest case {i+1}: {sample['label']}")
        print(f"Context: {sample['context'][:100]}...")
        
        for j, response in enumerate(test_case['responses']):
            reward = reward_fn.reward(sample, response)
            print(f"  Response {j+1}: '{response}' -> Reward: {reward:.3f}")


if __name__ == '__main__':
    test_entailment_reward()