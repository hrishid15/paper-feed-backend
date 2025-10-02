from typing import Dict, List, Tuple
from .content_based import ContentBasedFilter

class HybridRecommender:
    def __init__(self):
        self.content_based = ContentBasedFilter()
        self.is_trained = False
    
    def fit(self, interactions: List[Dict], papers_data: List[Dict]):
        """Train the recommender with papers and interactions"""
        self.content_based.fit(papers_data)
        self.is_trained = True
        print("Hybrid model trained successfully (content-based mode)")
    
    def get_recommendations(self, user_id: str, user_interactions: List[Dict] = None, 
                      num_recommendations: int = 10, user_interests: List[str] = None) -> List[Tuple[str, float, str]]:
        """Get paper recommendations with adaptive strategy"""
        if not self.is_trained:
            return []
        
        strategy = self._choose_strategy(user_interactions, user_interests)
        
        if strategy == "interest_based":
            # New user with stated interests - USE CATEGORY-AWARE SEARCH
            recs = self.content_based.get_papers_by_interest(user_interests, num_recommendations, use_category_filter=True)
            return [(paper_id, score, "interest_based") for paper_id, score in recs]
        
        elif strategy == "content_based":
            # User with interaction history - USE CATEGORY-AWARE SEARCH
            recs = self.content_based.get_recommendations(user_interactions, num_recommendations, use_category_filter=True)
            return [(paper_id, score, "content_based") for paper_id, score in recs]
        
        else:  # popular fallback
            recs = self.content_based._get_popular_papers(num_recommendations)
            return [(paper_id, score, "popular") for paper_id, score in recs]
    
    def _choose_strategy(self, user_interactions: List[Dict] = None, user_interests: List[str] = None) -> str:
        """Choose recommendation strategy based on available data"""
        
        # If user has interactions, use content-based
        if user_interactions and len(user_interactions) > 0:
            return "content_based"
        
        # If new user but has stated interests, use interest-based
        if user_interests and len(user_interests) > 0:
            return "interest_based"
        
        # Complete cold start - use popular papers
        return "popular"
    
    def get_feed_with_diversity(self, user_id: str, user_interactions: List[Dict] = None,
                                num_papers: int = 20, user_interests: List[str] = None,
                                diversity_ratio: float = 0.2) -> List[Tuple[str, float, str]]:
        """
        Get paper feed with diversity injection for exploration
        
        Args:
            diversity_ratio: Fraction of recommendations from adjacent/diverse topics (0-1)
        """
        # Get main recommendations
        num_main = int(num_papers * (1 - diversity_ratio))
        num_diverse = num_papers - num_main
        
        main_recs = self.get_recommendations(user_id, user_interactions, num_main, user_interests)
        
        # For diversity, get popular papers from different categories
        # This prevents filter bubbles
        diverse_recs = self.content_based._get_popular_papers(num_diverse * 2)
        
        # Filter out papers already in main recommendations
        main_paper_ids = {paper_id for paper_id, _, _ in main_recs}
        diverse_recs = [(paper_id, score, "diverse") 
                       for paper_id, score in diverse_recs 
                       if paper_id not in main_paper_ids][:num_diverse]
        
        # Interleave main and diverse recommendations
        feed = []
        for i in range(max(len(main_recs), len(diverse_recs))):
            if i < len(main_recs):
                feed.append(main_recs[i])
            if i < len(diverse_recs):
                feed.append(diverse_recs[i])
        
        return feed[:num_papers]
    
    def update_user_interaction(self, user_id: str, paper_id: str, interaction_type: str, rating: float = None, time_spent: int = None):
        # For content-based, we don't need to update the model
        # The user profile is rebuilt from scratch each time using latest interactions
        pass