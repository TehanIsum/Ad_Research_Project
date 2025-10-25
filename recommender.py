"""
Ad Recommendation Engine

Matches detected demographic data (age, gender, mood) with ad inventory
using a scoring algorithm that considers:
- Target age group match
- Target gender match
- Target mood match
- Ad priority
- Randomness for serendipity

The best-scoring ad is recommended to the user.
"""

import pandas as pd
import random
from typing import Dict, Optional
import os


class AdRecommender:
    """
    Recommends ads based on detected demographic data.
    
    Uses a scoring function that matches user demographics with ad targeting
    parameters and applies priority weighting.
    """
    
    def __init__(self, ads_csv_path: str):
        """
        Initialize the recommender with ad inventory.
        
        Args:
            ads_csv_path: Path to the ads CSV file
        """
        self.ads_csv_path = ads_csv_path
        self.ads_df = None
        self.load_ads()
    
    def load_ads(self):
        """Load ad inventory from CSV file."""
        try:
            if not os.path.exists(self.ads_csv_path):
                raise FileNotFoundError(f"Ads CSV not found: {self.ads_csv_path}")
            
            self.ads_df = pd.read_csv(self.ads_csv_path)
            
            # Validate required columns
            required_cols = ['ad_id', 'title', 'description', 'target_age_groups', 
                           'target_genders', 'target_moods', 'priority']
            
            missing_cols = [col for col in required_cols if col not in self.ads_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            print(f"✓ Loaded {len(self.ads_df)} ads from {self.ads_csv_path}")
            
        except Exception as e:
            print(f"✗ Error loading ads: {e}")
            self.ads_df = None
    
    def score_ad(self, detection: Dict[str, str], ad_row: pd.Series) -> float:
        """
        Calculate match score between detected demographics and ad targeting.
        
        Scoring algorithm:
        - Age match: +10 points if exact match, +5 if 'All' in targets
        - Gender match: +10 points if exact match, +5 if 'All' in targets
        - Mood match: +10 points if exact match, +5 if 'All' in targets
        - Priority: +priority points (1-10)
        - Randomness: +0 to +3 points for serendipity
        
        Args:
            detection: Dictionary with 'age_group', 'gender', 'mood'
            ad_row: Pandas Series representing one ad
            
        Returns:
            Total score (float)
        """
        score = 0.0
        
        # Extract detection data
        detected_age = detection.get('age_group', '').lower()
        detected_gender = detection.get('gender', '').lower()
        detected_mood = detection.get('mood', '').lower()
        
        # Extract ad targeting (convert to lowercase for comparison)
        target_ages = str(ad_row['target_age_groups']).lower().split('|')
        target_genders = str(ad_row['target_genders']).lower().split('|')
        target_moods = str(ad_row['target_moods']).lower().split('|')
        priority = int(ad_row['priority'])
        
        # Clean whitespace
        target_ages = [age.strip() for age in target_ages]
        target_genders = [gender.strip() for gender in target_genders]
        target_moods = [mood.strip() for mood in target_moods]
        
        # Score age match
        if detected_age in target_ages:
            score += 10.0
        elif 'all' in target_ages:
            score += 5.0
        
        # Score gender match
        if detected_gender in target_genders:
            score += 10.0
        elif 'all' in target_genders:
            score += 5.0
        
        # Score mood match
        if detected_mood in target_moods:
            score += 10.0
        elif 'all' in target_moods:
            score += 5.0
        
        # Add priority weight
        score += priority
        
        # Add small random factor for serendipity (0-3 points)
        # This ensures variety and avoids showing the same ad repeatedly
        score += random.uniform(0, 3)
        
        return score
    
    def recommend_ad(self, detection: Dict[str, str]) -> Optional[pd.Series]:
        """
        Recommend the best-fit ad for detected demographics.
        
        Args:
            detection: Dictionary with 'age_group', 'gender', 'mood'
            
        Returns:
            Pandas Series of the recommended ad, or None if no ads available
        """
        if self.ads_df is None or len(self.ads_df) == 0:
            print("✗ No ads available for recommendation")
            return None
        
        # Calculate scores for all ads
        scores = []
        for idx, ad_row in self.ads_df.iterrows():
            score = self.score_ad(detection, ad_row)
            scores.append((score, idx))
        
        # Sort by score (descending)
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Get best ad
        best_score, best_idx = scores[0]
        best_ad = self.ads_df.iloc[best_idx]
        
        return best_ad
    
    def get_ad_summary(self, ad: pd.Series) -> Dict[str, str]:
        """
        Get a formatted summary of an ad for display.
        
        Args:
            ad: Pandas Series representing an ad
            
        Returns:
            Dictionary with formatted ad information
        """
        return {
            'ad_id': str(ad['ad_id']),
            'title': str(ad['title']),
            'description': str(ad['description']),
            'target_age_groups': str(ad['target_age_groups']),
            'target_genders': str(ad['target_genders']),
            'target_moods': str(ad['target_moods']),
            'priority': str(ad['priority']),
            'creative_url': str(ad.get('creative_url', 'N/A'))
        }


# Example usage for testing
if __name__ == '__main__':
    print("AdRecommender Module - Test Mode")
    print("=" * 60)
    
    # Initialize recommender
    ads_path = 'data/ads.csv'
    recommender = AdRecommender(ads_path)
    
    # Test different demographic profiles
    test_profiles = [
        {'age_group': 'young', 'gender': 'Male', 'mood': 'happy'},
        {'age_group': 'adult', 'gender': 'Female', 'mood': 'neutral'},
        {'age_group': 'teen', 'gender': 'Male', 'mood': 'happy'},
        {'age_group': 'senior', 'gender': 'Male', 'mood': 'neutral'},
        {'age_group': 'child', 'gender': 'Female', 'mood': 'happy'},
    ]
    
    print("\nTesting recommendations for different profiles:\n")
    
    for profile in test_profiles:
        print(f"Profile: {profile}")
        
        ad = recommender.recommend_ad(profile)
        
        if ad is not None:
            summary = recommender.get_ad_summary(ad)
            print(f"  → Recommended: {summary['ad_id']} - {summary['title']}")
            print(f"    Targets: {summary['target_age_groups']}, "
                  f"{summary['target_genders']}, {summary['target_moods']}")
        else:
            print("  → No recommendation available")
        
        print()
