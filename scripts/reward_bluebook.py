#!/usr/bin/env python3
"""
Reward function for Bluebook citation completion task.
"""
import re
from typing import Dict, Tuple, Optional
from difflib import SequenceMatcher


class BluebookRewardFunction:
    def __init__(self):
        """Initialize Bluebook citation reward function"""
        # Common citation patterns for validation
        self.citation_patterns = {
            'us': re.compile(r'(\d+)\s+U\.S\.\s+(\d+)\s+\((\d{4})\)'),
            'f2d': re.compile(r'(\d+)\s+F\.2d\s+(\d+)\s+\(([^)]+)\s+(\d{4})\)'),
            'f3d': re.compile(r'(\d+)\s+F\.3d\s+(\d+)\s+\(([^)]+)\s+(\d{4})\)'),
            'fed': re.compile(r'(\d+)\s+F\.\s+(\d+)\s+\(([^)]+)\s+(\d{4})\)'),
            'sct': re.compile(r'(\d+)\s+S\.\s?Ct\.\s+(\d+)\s+\((\d{4})\)')
        }
    
    def extract_citation_from_response(self, response: str) -> Optional[str]:
        """Extract the completed citation from model response"""
        response = response.strip()
        
        # Look for complete citations in the response
        for pattern in self.citation_patterns.values():
            matches = pattern.findall(response)
            if matches:
                # Return the first complete citation found
                match = matches[0]
                if len(match) == 3:  # US or S.Ct format (vol, page, year)
                    vol, page, year = match
                    if 'U.S.' in response:
                        return f"{vol} U.S. {page} ({year})"
                    elif 'S. Ct.' in response or 'S.Ct.' in response:
                        return f"{vol} S. Ct. {page} ({year})"
                elif len(match) == 4:  # Federal format (vol, page, court, year)
                    vol, page, court, year = match
                    if 'F.2d' in response:
                        return f"{vol} F.2d {page} ({court} {year})"
                    elif 'F.3d' in response:
                        return f"{vol} F.3d {page} ({court} {year})"
                    elif 'F.' in response:
                        return f"{vol} F. {page} ({court} {year})"
        
        # If no complete citation found, try to extract partial information
        # and reconstruct based on the ground truth pattern
        return response.strip()
    
    def parse_citation_components(self, citation: str) -> Dict:
        """Parse citation into components"""
        components = {}
        
        for reporter_type, pattern in self.citation_patterns.items():
            match = pattern.match(citation)
            if match:
                if reporter_type == 'us':
                    components = {
                        'volume': match.group(1),
                        'page': match.group(2),
                        'year': match.group(3),
                        'reporter': 'U.S.'
                    }
                elif reporter_type in ['f2d', 'f3d', 'fed']:
                    reporter_name = 'F.2d' if reporter_type == 'f2d' else ('F.3d' if reporter_type == 'f3d' else 'F.')
                    components = {
                        'volume': match.group(1),
                        'page': match.group(2),
                        'court': match.group(3),
                        'year': match.group(4),
                        'reporter': reporter_name
                    }
                elif reporter_type == 'sct':
                    components = {
                        'volume': match.group(1),
                        'page': match.group(2),
                        'year': match.group(3),
                        'reporter': 'S. Ct.'
                    }
                break
        
        return components
    
    def compute_citation_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Compute accuracy score for citation completion"""
        # Parse both citations
        pred_components = self.parse_citation_components(predicted)
        gt_components = self.parse_citation_components(ground_truth)
        
        if not pred_components or not gt_components:
            # Fallback to string similarity if parsing fails
            return SequenceMatcher(None, predicted.lower(), ground_truth.lower()).ratio()
        
        # Check component-wise accuracy
        total_components = len(gt_components)
        correct_components = 0
        
        for key, gt_value in gt_components.items():
            if key in pred_components:
                pred_value = pred_components[key]
                if pred_value.lower().strip() == gt_value.lower().strip():
                    correct_components += 1
                elif key in ['volume', 'page', 'year']:
                    # For numeric fields, be more lenient with formatting
                    pred_clean = re.sub(r'[^\d]', '', pred_value)
                    gt_clean = re.sub(r'[^\d]', '', gt_value)
                    if pred_clean == gt_clean:
                        correct_components += 0.9  # Slight penalty for formatting
        
        return correct_components / total_components if total_components > 0 else 0.0
    
    def validate_citation_format(self, citation: str) -> float:
        """Validate that citation follows proper Bluebook format"""
        # Check if citation matches any recognized pattern
        for pattern in self.citation_patterns.values():
            if pattern.match(citation.strip()):
                return 1.0
        
        # Partial credit for citations that have correct structure but minor formatting issues
        # Check for basic components: numbers, reporter, parentheses
        has_volume = bool(re.search(r'^\d+', citation.strip()))
        has_reporter = bool(re.search(r'[A-Z]\.[^(]*', citation))
        has_page = bool(re.search(r'\d+', citation[5:] if len(citation) > 5 else ''))
        has_year = bool(re.search(r'\(\d{4}\)', citation))
        
        component_score = sum([has_volume, has_reporter, has_page, has_year]) / 4.0
        return component_score * 0.7  # Partial credit
    
    def reward(self, sample: Dict, model_output: str) -> float:
        """
        Main reward function for Bluebook citation completion.
        
        Args:
            sample: Dictionary with 'ground_truth' citation and 'metadata'
            model_output: Model's response with completed citation
            
        Returns:
            Float reward between 0.0 and 1.0
        """
        try:
            ground_truth = sample['ground_truth']
            
            # Extract citation from model response
            predicted_citation = self.extract_citation_from_response(model_output)
            
            if not predicted_citation:
                return 0.0
            
            # Compute accuracy score
            accuracy_score = self.compute_citation_accuracy(predicted_citation, ground_truth)
            
            # Validate format
            format_score = self.validate_citation_format(predicted_citation)
            
            # Combine scores with weights
            # Accuracy is most important, format validation provides additional reward
            final_score = 0.8 * accuracy_score + 0.2 * format_score
            
            return min(1.0, final_score)
            
        except Exception as e:
            print(f"Error computing Bluebook citation reward: {e}")
            return 0.0


def test_bluebook_reward():
    """Test the Bluebook citation reward function"""
    reward_fn = BluebookRewardFunction()
    
    # Test cases
    test_cases = [
        {
            'sample': {
                'ground_truth': '123 U.S. 456 (1990)',
                'metadata': {'volume': '123', 'page': '456', 'year': '1990', 'reporter': 'U.S.'}
            },
            'responses': [
                '123 U.S. 456 (1990)',  # Perfect match
                '123 U.S. 456 (1990)',  # Perfect match with extra text
                '123 U.S. 456 (1991)',  # Wrong year
                '124 U.S. 456 (1990)',  # Wrong volume
                'The citation is 123 U.S. 456 (1990)',  # Embedded in text
                '123 US 456 1990',  # Missing formatting
                '456 F.2d 789 (2d Cir. 1995)'  # Wrong citation entirely
            ]
        },
        {
            'sample': {
                'ground_truth': '456 F.2d 789 (2d Cir. 1995)',
                'metadata': {'volume': '456', 'page': '789', 'court': '2d Cir.', 'year': '1995', 'reporter': 'F.2d'}
            },
            'responses': [
                '456 F.2d 789 (2d Cir. 1995)',  # Perfect
                '456 F.2d 789 (2nd Cir. 1995)',  # Different court format
                '456 F.2d 790 (2d Cir. 1995)',  # Wrong page
            ]
        }
    ]
    
    for test_case in test_cases:
        sample = test_case['sample']
        print(f"\nGround truth: {sample['ground_truth']}")
        
        for i, response in enumerate(test_case['responses']):
            reward = reward_fn.reward(sample, response)
            print(f"Response {i+1}: '{response}' -> Reward: {reward:.3f}")


if __name__ == '__main__':
    test_bluebook_reward()