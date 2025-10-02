import asyncio
import sys
import signal
from api.recommendation_server import RecommendationAPI
from data.sample_papers import SAMPLE_PAPERS, SAMPLE_INTERACTIONS, USER_INTERESTS

class RecommendationSystem:
    def __init__(self):
        self.api = None
        self.running = False
    
    async def start_system(self):
        print("Starting Paper Recommendation Engine...")
        print("=" * 60)
        
        self.api = RecommendationAPI(port=8000)
        
        # Initialize with sample papers and interactions
        await self.api.initialize(interactions=SAMPLE_INTERACTIONS, papers_data=SAMPLE_PAPERS)
        
        # Set user interests for cold start
        for user_id, interests in USER_INTERESTS.items():
            await self.api.recommender.db.create_or_update_user(user_id)
            await self.api.recommender.db.set_user_initial_interests(user_id, interests)
        
        await self.api.start()
        
        self.running = True
        
        print("Server started successfully!")
        print("Database: recommendation_engine.db")
        print("API: http://localhost:8000")
        print("=" * 60)
        print("\nAvailable endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /feed/{user_id} - Get personalized paper feed")
        print("  POST /interactions - Record user interaction")
        print("  GET  /similar/{paper_id} - Get similar papers")
        print("  GET  /popular - Get popular papers")
        print("  GET  /category/{category} - Get papers by category")
        print("  POST /users/{user_id}/interests - Set user interests")
        print("  GET  /users/{user_id}/interests - Get user interests")
        print("  GET  /stats - System statistics")
        print("\nPress Ctrl+C to stop")
        print("=" * 60)
        
        # Keep running
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
    
    async def stop_system(self):
        print("\nShutting down recommendation system...")
        if self.api:
            await self.api.close()
        self.running = False
        print("System stopped")

async def run_system():
    system = RecommendationSystem()
    
    try:
        await system.start_system()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    finally:
        await system.stop_system()

def show_menu():
    print("Paper Recommendation Engine")
    print("=" * 55)
    print("Choose an option:")
    print("1. Start API server")
    print("2. Help")
    print("3. Exit")
    print("=" * 55)

async def show_help():
    print("\nHelp - Paper Recommendation Engine")
    print("=" * 50)
    print("FEATURES:")
    print("  - Semantic paper recommendations using SPECTER embeddings")
    print("  - Category-aware filtering for subfield relevance")
    print("  - Cold-start support with interest-based matching")
    print("  - Real-time learning from user interactions")
    print("  - Sub-100ms response times with caching")
    print()
    print("SAMPLE USERS:")
    print("  - researcher_alice: NLP/Transformers")
    print("  - researcher_bob: Computer Vision")  
    print("  - researcher_carol: Robotics/RL")
    print("  - researcher_dave: Quantum Computing")
    print("  - researcher_eve: Generative Models")
    print("  - researcher_frank: Meta-Learning")
    print()
    print("EXAMPLE API CALLS:")
    print("  curl http://localhost:8000/feed/researcher_alice")
    print("  curl http://localhost:8000/popular?category=cs.CV")
    print("  curl http://localhost:8000/similar/arxiv_2301_001")

async def main():
    while True:
        try:
            show_menu()
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == "1":
                await run_system()
                break
            elif choice == "2":
                await show_help()
                input("\nPress Enter to continue...")
                print("\n" + "=" * 50 + "\n")
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.\n")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break

if __name__ == "__main__":
    if sys.platform != "win32":
        def signal_handler(sig, frame):
            print("\nReceived interrupt signal, shutting down...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)