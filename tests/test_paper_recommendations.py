import asyncio
import sys
sys.path.append('..')
from models.persistent_cached_hybrid_recommender import PersistentCachedHybridRecommender
from data.sample_papers import SAMPLE_PAPERS, SAMPLE_INTERACTIONS, USER_INTERESTS

async def test_paper_recommendation_system():
    print("=" * 60)
    print("TESTING PAPER RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Initialize the system
    system = PersistentCachedHybridRecommender("test_papers.db")
    
    print("\n1. Loading sample papers into database...")
    await system.initialize(interactions=[], papers_data=SAMPLE_PAPERS)
    print(f"   ✓ Loaded {len(SAMPLE_PAPERS)} papers")
    
    print("\n2. Recording sample user interactions...")
    for interaction in SAMPLE_INTERACTIONS:
        await system.record_user_interaction(
            user_id=interaction['user_id'],
            paper_id=interaction['paper_id'],
            interaction_type=interaction['interaction_type'],
            rating=interaction.get('rating'),
            time_spent=interaction.get('time_spent_seconds')
        )
    print(f"   ✓ Recorded {len(SAMPLE_INTERACTIONS)} interactions")
    
    print("\n3. Setting initial interests for users...")
    for user_id, interests in USER_INTERESTS.items():
        await system.db.create_or_update_user(user_id)
        await system.db.set_user_initial_interests(user_id, interests)
    print(f"   ✓ Set interests for {len(USER_INTERESTS)} users")
    
    # Test 1: User with NLP interests (Alice)
    print("\n" + "=" * 60)
    print("TEST 1: Researcher Alice (NLP/Transformers enthusiast)")
    print("=" * 60)
    alice_interactions = await system.db.get_user_interactions("researcher_alice", limit=50)
    print(f"Alice has {len(alice_interactions)} interactions")
    
    result = await system.get_recommendations("researcher_alice", num_recommendations=5)
    print(f"\nTop 5 recommendations for Alice:")
    print(f"Source: {result['source']} | Response time: {result['response_time_ms']}ms\n")
    
    for i, rec in enumerate(result['recommendations'], 1):
        paper = next(p for p in SAMPLE_PAPERS if p['paper_id'] == rec['item'])
        print(f"{i}. {paper['title']}")
        print(f"   Category: {paper['arxiv_category']} | Score: {rec['score']:.3f} | Strategy: {rec['strategy']}")
        print(f"   Keywords: {paper['keywords'][:80]}...")
        print()
    
    # Test 2: User with Computer Vision interests (Bob)
    print("=" * 60)
    print("TEST 2: Researcher Bob (Computer Vision expert)")
    print("=" * 60)
    bob_interactions = await system.db.get_user_interactions("researcher_bob", limit=50)
    print(f"Bob has {len(bob_interactions)} interactions")
    
    result = await system.get_recommendations("researcher_bob", num_recommendations=5)
    print(f"\nTop 5 recommendations for Bob:")
    print(f"Source: {result['source']} | Response time: {result['response_time_ms']}ms\n")
    
    for i, rec in enumerate(result['recommendations'], 1):
        paper = next(p for p in SAMPLE_PAPERS if p['paper_id'] == rec['item'])
        print(f"{i}. {paper['title']}")
        print(f"   Category: {paper['arxiv_category']} | Score: {rec['score']:.3f} | Strategy: {rec['strategy']}")
        print(f"   Keywords: {paper['keywords'][:80]}...")
        print()
    
    # Test 3: Brand new user with stated interests (Dave - Quantum Computing)
    print("=" * 60)
    print("TEST 3: Researcher Dave (NEW USER - Quantum Computing interests)")
    print("=" * 60)
    dave_interests = USER_INTERESTS["researcher_dave"]
    print(f"Dave's interests: {', '.join(dave_interests)}")
    
    # For new users, we need to pass interests differently
    # Let's test the cold start by getting recommendations without interactions
    result = await system.get_recommendations("researcher_dave", num_recommendations=5)
    print(f"\nTop 5 recommendations for Dave (cold start):")
    print(f"Source: {result['source']} | Response time: {result['response_time_ms']}ms\n")
    
    for i, rec in enumerate(result['recommendations'], 1):
        paper = next(p for p in SAMPLE_PAPERS if p['paper_id'] == rec['item'])
        print(f"{i}. {paper['title']}")
        print(f"   Category: {paper['arxiv_category']} | Score: {rec['score']:.3f} | Strategy: {rec['strategy']}")
        print(f"   Keywords: {paper['keywords'][:80]}...")
        print()
    
    # Test 4: Similar papers (find papers similar to Attention paper)
    print("=" * 60)
    print("TEST 4: Finding papers similar to 'Attention Is All You Need'")
    print("=" * 60)
    
    similar_result = await system.get_similar_items("arxiv_2301_001", num_similar=5)
    print(f"Similar papers:")
    print(f"Source: {similar_result['source']} | Response time: {similar_result['response_time_ms']}ms\n")
    
    for i, item in enumerate(similar_result['similar_items'], 1):
        paper = next(p for p in SAMPLE_PAPERS if p['paper_id'] == item['item'])
        print(f"{i}. {paper['title']}")
        print(f"   Similarity score: {item['score']:.3f}")
        print(f"   Category: {paper['arxiv_category']}")
        print()
    
    # Test 5: Popular papers by category
    print("=" * 60)
    print("TEST 5: Popular papers in Computer Vision (cs.CV)")
    print("=" * 60)
    
    popular_result = await system.get_popular_items("cs.CV", num_items=5)
    print(f"Popular CV papers:")
    print(f"Source: {popular_result['source']} | Response time: {popular_result['response_time_ms']}ms\n")
    
    for i, item in enumerate(popular_result['popular_items'], 1):
        paper = next(p for p in SAMPLE_PAPERS if p['paper_id'] == item['item'])
        print(f"{i}. {paper['title']}")
        print(f"   Citations: {paper['citation_count']:,}")
        print(f"   Score: {item['score']:.3f}")
        print()
    
    # Test 6: System performance stats
    print("=" * 60)
    print("TEST 6: System Performance Statistics")
    print("=" * 60)
    
    stats = system.get_performance_stats()
    print(f"Total recommendation requests: {stats['recommendation_requests']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']}")
    print(f"Database connected: {stats['database_connected']}")
    print(f"Cache size: {stats['cache_performance']['cache_size']} items")
    
    await system.close()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)
    print(f"Database saved as: test_papers.db")
    print("You can now inspect the database or run the API server with this data.")

if __name__ == "__main__":
    asyncio.run(test_paper_recommendation_system())