import asyncio
import json
import time
from aiohttp import web, ClientError
from typing import Dict, Any
from models.persistent_cached_hybrid_recommender import PersistentCachedHybridRecommender

class RecommendationAPI:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.recommender = PersistentCachedHybridRecommender()
        self.app = self._create_app()
        self.request_count = 0
        self.start_time = time.time()
    
    def _create_app(self):
        app = web.Application()
        
        # Health & system endpoints
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/stats', self.get_system_stats)
        
        # Paper feed endpoints
        app.router.add_get('/feed/{user_id}', self.get_paper_feed)
        app.router.add_post('/interactions', self.record_interaction)
        
        # Discovery endpoints
        app.router.add_get('/similar/{paper_id}', self.get_similar_papers)
        app.router.add_get('/popular', self.get_popular_papers)
        app.router.add_get('/category/{category}', self.get_papers_by_category)
        
        # User endpoints
        app.router.add_post('/users/{user_id}/interests', self.set_user_interests)
        app.router.add_get('/users/{user_id}/interests', self.get_user_interests)
        
        return app
    
    async def initialize(self, interactions: list = None, papers_data: list = None):
        await self.recommender.initialize(interactions, papers_data)
        print(f"Recommendation API initialized on {self.host}:{self.port}")
    
    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"Server started on http://{self.host}:{self.port}")
    
    async def health_check(self, request):
        return web.json_response({
            "status": "healthy",
            "service": "paper-recommendation-feed",
            "uptime_seconds": int(time.time() - self.start_time),
            "requests_served": self.request_count
        })
    
    async def get_paper_feed(self, request):
        """Get personalized paper feed for TikTok-style scrolling"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            user_id = request.match_info['user_id']
            count = int(request.query.get('count', 20))
            diversity = float(request.query.get('diversity', 0.2))
            
            if count > 100:
                return web.json_response({
                    "error": "Maximum 100 papers per request"
                }, status=400)
            
            result = await self.recommender.get_recommendations(user_id, count)
            
            response_time = (time.time() - start_time) * 1000
            result['api_response_time_ms'] = f"{response_time:.2f}"
            result['feed_type'] = 'personalized'
            
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({
                "error": f"Failed to get paper feed: {str(e)}"
            }, status=500)
    
    async def record_interaction(self, request):
        """Record user interaction with a paper (view, like, save, time spent)"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            data = await request.json()
            
            required_fields = ['user_id', 'paper_id', 'interaction_type']
            for field in required_fields:
                if field not in data:
                    return web.json_response({
                        "error": f"Missing required field: {field}"
                    }, status=400)
            
            user_id = data['user_id']
            paper_id = data['paper_id']
            interaction_type = data['interaction_type']
            rating = data.get('rating')
            time_spent = data.get('time_spent_seconds')
            
            # Validate interaction type
            valid_types = ['view', 'like', 'save', 'skip', 'share']
            if interaction_type not in valid_types:
                return web.json_response({
                    "error": f"Invalid interaction_type. Must be one of: {valid_types}"
                }, status=400)
            
            # Validate rating if provided
            if rating is not None and not (1 <= rating <= 5):
                return web.json_response({
                    "error": "Rating must be between 1 and 5"
                }, status=400)
            
            result = await self.recommender.record_user_interaction(
                user_id, paper_id, interaction_type, rating, time_spent
            )
            
            response_time = (time.time() - start_time) * 1000
            result['api_response_time_ms'] = f"{response_time:.2f}"
            
            return web.json_response(result)
            
        except json.JSONDecodeError:
            return web.json_response({
                "error": "Invalid JSON in request body"
            }, status=400)
        except ValueError as e:
            return web.json_response({
                "error": f"Invalid data: {str(e)}"
            }, status=400)
        except Exception as e:
            return web.json_response({
                "error": f"Failed to record interaction: {str(e)}"
            }, status=500)
    
    async def get_similar_papers(self, request):
        """Find papers similar to a given paper"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            paper_id = request.match_info['paper_id']
            num_similar = int(request.query.get('count', 10))
            
            if num_similar > 50:
                return web.json_response({
                    "error": "Maximum 50 similar papers allowed"
                }, status=400)
            
            result = await self.recommender.get_similar_items(paper_id, num_similar)
            
            response_time = (time.time() - start_time) * 1000
            result['api_response_time_ms'] = f"{response_time:.2f}"
            
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({
                "error": f"Failed to get similar papers: {str(e)}"
            }, status=500)
    
    async def get_popular_papers(self, request):
        """Get popular papers overall or by category"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            category = request.query.get('category', 'all')
            num_papers = int(request.query.get('count', 20))
            
            if num_papers > 100:
                return web.json_response({
                    "error": "Maximum 100 papers allowed"
                }, status=400)
            
            result = await self.recommender.get_popular_items(category, num_papers)
            
            response_time = (time.time() - start_time) * 1000
            result['api_response_time_ms'] = f"{response_time:.2f}"
            
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({
                "error": f"Failed to get popular papers: {str(e)}"
            }, status=500)
    
    async def get_papers_by_category(self, request):
        """Get papers in a specific ArXiv category"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            category = request.match_info['category']
            num_papers = int(request.query.get('count', 20))
            
            result = await self.recommender.get_popular_items(category, num_papers)
            
            response_time = (time.time() - start_time) * 1000
            result['api_response_time_ms'] = f"{response_time:.2f}"
            
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({
                "error": f"Failed to get papers by category: {str(e)}"
            }, status=500)
    
    async def set_user_interests(self, request):
        """Set user's initial research interests (for cold start)"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            user_id = request.match_info['user_id']
            data = await request.json()
            
            if 'interests' not in data:
                return web.json_response({
                    "error": "Missing 'interests' field (should be array of strings)"
                }, status=400)
            
            interests = data['interests']
            
            if not isinstance(interests, list) or not interests:
                return web.json_response({
                    "error": "Interests must be a non-empty array of strings"
                }, status=400)
            
            # CREATE USER FIRST
            await self.recommender.db.create_or_update_user(user_id)
            await self.recommender.db.set_user_initial_interests(user_id, interests)
            
            response_time = (time.time() - start_time) * 1000
            
            return web.json_response({
                "user_id": user_id,
                "interests": interests,
                "status": "updated",
                "response_time_ms": f"{response_time:.2f}"
            })
            
        except Exception as e:
            return web.json_response({
                "error": f"Failed to set user interests: {str(e)}"
            }, status=500)
    
    async def get_user_interests(self, request):
        """Get user's stated research interests"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            user_id = request.match_info['user_id']
            interests = await self.recommender.db.get_user_initial_interests(user_id)
            
            response_time = (time.time() - start_time) * 1000
            
            return web.json_response({
                "user_id": user_id,
                "interests": interests,
                "response_time_ms": f"{response_time:.2f}"
            })
            
        except Exception as e:
            return web.json_response({
                "error": f"Failed to get user interests: {str(e)}"
            }, status=500)
    
    async def get_system_stats(self, request):
        try:
            performance_stats = self.recommender.get_performance_stats()
            
            system_stats = {
                "service_uptime_seconds": int(time.time() - self.start_time),
                "total_api_requests": self.request_count,
                "performance": performance_stats
            }
            
            return web.json_response(system_stats)
            
        except Exception as e:
            return web.json_response({
                "error": f"Failed to get stats: {str(e)}"
            }, status=500)
    
    async def close(self):
        await self.recommender.close()