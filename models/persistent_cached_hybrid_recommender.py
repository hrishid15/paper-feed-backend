import time
from typing import Dict, List, Tuple, Optional
from models.hybrid_recommender import HybridRecommender
from cache.memory_cache import MemoryCache
from database.database_manager import DatabaseManager

class PersistentCachedHybridRecommender:
    def __init__(self, db_path: str = "recommendation_engine.db"):
        self.recommender = HybridRecommender()
        self.cache = MemoryCache()
        self.db = DatabaseManager(db_path)
        self.performance_stats = {"cache_hits": 0, "cache_misses": 0}
    
    async def initialize(self, interactions: List[Dict] = None, papers_data: List[Dict] = None):
        await self.db.initialize()
        await self.cache.connect()
        
        # Add papers to database if provided
        if papers_data:
            for paper in papers_data:
                await self.db.create_or_update_paper(paper)
        
        # Add interactions if provided
        if interactions:
            for interaction in interactions:
                await self.db.record_interaction(
                    interaction['user_id'],
                    interaction['paper_id'],
                    interaction['interaction_type'],
                    interaction.get('rating'),
                    interaction.get('time_spent_seconds')
                )
        
        await self._load_and_train_from_database()
        await self._precompute_popular_items()
        await self._precompute_item_similarities()
        
        print("Persistent cached hybrid recommender initialized")
    
    async def _load_and_train_from_database(self):
        interactions = await self.db.get_all_interactions()
        papers_data = []
        
        all_papers = await self.db.get_all_papers()
        for paper in all_papers:
            papers_data.append({
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "authors": paper["authors"],
                "arxiv_category": paper["arxiv_category"],
                "keywords": paper["keywords"],
                "publication_date": paper["publication_date"],
                "citation_count": paper["citation_count"]
            })
        
        if interactions and papers_data:
            ml_interactions = [
                {
                    "paper_id": interaction["paper_id"],
                    "rating": interaction["rating"],
                    "interaction_type": interaction.get("interaction_type", "view"),
                    "time_spent_seconds": interaction.get("time_spent_seconds", 0)
                }
                for interaction in interactions
            ]
            
            self.recommender.fit(ml_interactions, papers_data)
            print(f"Trained models with {len(ml_interactions)} interactions and {len(papers_data)} papers")
        else:
            print("No data found in database")
    
    async def get_recommendations(self, user_id: str, num_recommendations: int = 5) -> Dict:
        start_time = time.time()
        
        cached_recs = await self.cache.get_user_recommendations(user_id)
        
        if cached_recs:
            self.performance_stats["cache_hits"] += 1
            response_time = (time.time() - start_time) * 1000
            
            return {
                "recommendations": cached_recs[:num_recommendations],
                "user_id": user_id,
                "source": "cache",
                "response_time_ms": f"{response_time:.2f}",
                "timestamp": time.time()
            }
        
        self.performance_stats["cache_misses"] += 1
        
        # Get user interactions
        user_interactions = await self.db.get_user_interactions(user_id, limit=50)
        
        # Get user interests (for cold start)
        user_interests = await self.db.get_user_initial_interests(user_id)
        
        ml_interactions = [
            {
                "paper_id": interaction["paper_id"],
                "rating": interaction["rating"],
                "interaction_type": interaction.get("interaction_type", "view"),
                "time_spent_seconds": interaction.get("time_spent_seconds", 0)
            }
            for interaction in user_interactions
            if interaction.get("paper_id")
        ]

        print(f"DEBUG: user_id={user_id}, user_interests={user_interests}, len(ml_interactions)={len(ml_interactions)}")
        
        # Pass interests to recommender
        recommendations = self.recommender.get_recommendations(
            user_id, ml_interactions, num_recommendations * 2, user_interests=user_interests
        )
        
        await self.cache.set_user_recommendations(user_id, recommendations, ttl=300)
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "recommendations": [
                {"item": item, "score": score, "strategy": strategy}
                for item, score, strategy in recommendations[:num_recommendations]
            ],
            "user_id": user_id,
            "source": "computed",
            "response_time_ms": f"{response_time:.2f}",
            "timestamp": time.time()
        }
    
    async def record_user_interaction(self, user_id: str, paper_id: str, interaction_type: str = "view", 
                                    rating: float = None, time_spent: int = None):
        start_time = time.time()
        
        await self.db.record_interaction(user_id, paper_id, interaction_type, rating, time_spent)
        await self.cache.update_user_interaction(user_id, paper_id, rating if rating else 3.0)
        self.recommender.update_user_interaction(user_id, paper_id, interaction_type, rating, time_spent)
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "user_id": user_id,
            "paper_id": paper_id,
            "interaction_type": interaction_type,
            "rating": rating,
            "status": "recorded",
            "cache_invalidated": True,
            "persisted": True,
            "response_time_ms": f"{response_time:.2f}"
        }
    
    async def get_similar_items(self, item_id: str, num_similar: int = 5) -> Dict:
        start_time = time.time()
        
        cached_similar = await self.cache.get_item_similarity(item_id)
        
        if cached_similar:
            response_time = (time.time() - start_time) * 1000
            return {
                "item_id": item_id,
                "similar_items": cached_similar[:num_similar],
                "source": "cache",
                "response_time_ms": f"{response_time:.2f}"
            }
        
        similar_items = self.recommender.content_based.get_similar_papers(item_id, num_similar * 2)
        await self.cache.set_item_similarity(item_id, similar_items, ttl=86400)
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "item_id": item_id,
            "similar_items": [
                {"item": item, "score": score}
                for item, score in similar_items[:num_similar]
            ],
            "source": "computed",
            "response_time_ms": f"{response_time:.2f}"
        }
    
    async def get_popular_items(self, category: str = "all", num_items: int = 10) -> Dict:
        start_time = time.time()
        
        cached_popular = await self.cache.get_popular_items(category)
        
        if cached_popular:
            response_time = (time.time() - start_time) * 1000
            return {
                "category": category,
                "popular_items": cached_popular[:num_items],
                "source": "cache",
                "response_time_ms": f"{response_time:.2f}"
            }
        
        papers = await self.db.get_all_papers()
        
        if category != "all":
            papers = [paper for paper in papers if paper["arxiv_category"] == category]
        
        popular_items = [
            (paper["paper_id"], paper["current_avg_rating"] * (1 + paper["current_rating_count"] * 0.1))
            for paper in papers
        ]
        popular_items.sort(key=lambda x: x[1], reverse=True)
        
        await self.cache.set_popular_items(category, popular_items, ttl=1800)
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "category": category,
            "popular_items": [
                {"item": item, "score": score}
                for item, score in popular_items[:num_items]
            ],
            "source": "computed",
            "response_time_ms": f"{response_time:.2f}"
        }
    
    async def _precompute_popular_items(self):
        papers = await self.db.get_all_papers()
        categories = set(paper["arxiv_category"] for paper in papers)
        
        for category in categories:
            category_papers = [paper for paper in papers if paper["arxiv_category"] == category]
            popular = [
                (paper["paper_id"], paper["current_avg_rating"] * (1 + paper["current_rating_count"] * 0.1))
                for paper in category_papers
            ]
            popular.sort(key=lambda x: x[1], reverse=True)
            await self.cache.set_popular_items(category, popular, ttl=3600)
        
        all_popular = [
            (paper["paper_id"], paper["current_avg_rating"] * (1 + paper["current_rating_count"] * 0.1))
            for paper in papers
        ]
        all_popular.sort(key=lambda x: x[1], reverse=True)
        await self.cache.set_popular_items("all", all_popular, ttl=3600)

    async def _precompute_item_similarities(self):
        papers = await self.db.get_all_papers()
        
        for paper in papers:
            paper_id = paper["paper_id"]
            if hasattr(self.recommender.content_based, 'papers') and paper_id in self.recommender.content_based.papers:
                similar_papers = self.recommender.content_based.get_similar_papers(paper_id, 10)
                await self.cache.set_item_similarity(paper_id, similar_papers, ttl=86400)
    
    def get_performance_stats(self) -> Dict:
        cache_stats = self.cache.get_cache_stats()
        total_requests = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        
        return {
            "recommendation_requests": total_requests,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_performance": cache_stats,
            "system_performance": self.performance_stats,
            "database_connected": self.db.connection is not None
        }
    
    async def close(self):
        await self.cache.close()
        await self.db.close()