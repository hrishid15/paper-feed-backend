import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFilter:
    def __init__(self):
        self.paper_features = {}
        self.paper_similarity = None
        self.papers = []
        # Use allenai-specter - specifically trained for scientific papers
        # Alternative: 'all-MiniLM-L6-v2' (faster but less specialized)
        print("Loading sentence transformer model for semantic analysis...")
        self.model = SentenceTransformer('allenai-specter')
        self.embeddings = None
    
    def fit(self, papers_data: List[Dict]):
        """Train the content-based filter on research papers using semantic embeddings"""
        self.papers = [paper['paper_id'] for paper in papers_data]
        
        for paper in papers_data:
            self.paper_features[paper['paper_id']] = paper
        
        # Build text for embedding - SPECTER uses [TITLE] [SEP] [ABSTRACT]
        paper_texts = []
        for paper in papers_data:
            # SPECTER format: title + separator + abstract
            text = f"{paper['title']} [SEP] {paper['abstract']}"
            paper_texts.append(text)
        
        print(f"Generating embeddings for {len(paper_texts)} papers...")
        # Generate embeddings (768-dimensional semantic vectors)
        self.embeddings = self.model.encode(paper_texts, show_progress_bar=True)
        
        # Calculate pairwise similarity matrix
        self.paper_similarity = cosine_similarity(self.embeddings)
        
        print(f"Content model trained with {len(papers_data)} papers using semantic embeddings")
    
    def _get_related_categories(self, categories: List[str]) -> List[str]:
        """Get related ArXiv categories for a given set of categories"""
        category_groups = {
            'nlp': ['cs.CL', 'cs.LG', 'cs.AI'],
            'vision': ['cs.CV', 'cs.LG', 'cs.AI'],
            'ml': ['cs.LG', 'cs.AI', 'stat.ML'],
            'robotics': ['cs.RO', 'cs.LG', 'cs.AI'],
            'quantum': ['quant-ph', 'cs.ET', 'physics.comp-ph'],
            'theory': ['cs.DS', 'cs.CC', 'math.OC'],
            'security': ['cs.CR', 'cs.SE', 'cs.NI']
        }
        
        related = set()
        for cat in categories:
            related.add(cat)
            # Find which group this category belongs to
            for group, group_cats in category_groups.items():
                if cat in group_cats:
                    related.update(group_cats)
        
        return list(related)

    def get_recommendations(self, user_interactions: List[Dict], num_recommendations: int = 5, 
                       use_category_filter: bool = True) -> List[Tuple[str, float]]:
        """Get paper recommendations based on user's interaction history"""
        if not user_interactions:
            return self._get_popular_papers(num_recommendations)
        
        # Build user profile
        user_profile = self._build_user_profile(user_interactions)
        
        # Get categories from user's interaction history
        user_categories = set()
        for interaction in user_interactions:
            paper_id = interaction['paper_id']
            if paper_id in self.papers:
                category = self.paper_features[paper_id].get('arxiv_category')
                if category:
                    user_categories.add(category)
        
        # Determine which papers to consider
        if use_category_filter and user_categories:
            # Get related categories
            related_categories = self._get_related_categories(list(user_categories))
            
            # Filter to papers in related categories
            candidate_indices = []
            candidate_papers = []
            for i, paper_id in enumerate(self.papers):
                paper_category = self.paper_features[paper_id].get('arxiv_category')
                if paper_category in related_categories:
                    candidate_indices.append(i)
                    candidate_papers.append(paper_id)
            
            if not candidate_indices:
                # Fallback: no papers in related categories, use all
                candidate_indices = list(range(len(self.papers)))
                candidate_papers = self.papers.copy()
            
            # Calculate similarities only for candidate papers
            candidate_embeddings = self.embeddings[candidate_indices]
            similarities = cosine_similarity([user_profile], candidate_embeddings)[0]
            
            # Get papers user has already interacted with
            interacted_papers = {interaction['paper_id'] for interaction in user_interactions}
            
            # Build recommendations
            recommendations = []
            for i, paper_id in enumerate(candidate_papers):
                if paper_id not in interacted_papers:
                    recommendations.append((paper_id, similarities[i]))
        else:
            # Original behavior: no filtering
            similarities = cosine_similarity([user_profile], self.embeddings)[0]
            interacted_papers = {interaction['paper_id'] for interaction in user_interactions}
            
            recommendations = []
            for i, paper_id in enumerate(self.papers):
                if paper_id not in interacted_papers:
                    recommendations.append((paper_id, similarities[i]))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [(paper_id, float(score)) for paper_id, score in recommendations[:num_recommendations]]
    
    def _build_user_profile(self, interactions: List[Dict]):
        """Build user profile from their interaction history with implicit feedback weighting"""
        # Weight interactions based on type and rating
        weighted_embeddings = []
        
        for interaction in interactions:
            paper_id = interaction['paper_id']
            
            if paper_id not in self.papers:
                continue
            
            # Calculate weight based on interaction type and rating
            weight = 0.0
            
            if interaction.get('interaction_type') == 'save':
                weight = 1.5  # Saved papers are strong positive signal
            elif interaction.get('interaction_type') == 'like':
                weight = 1.2
            elif interaction.get('interaction_type') == 'view':
                weight = 0.5
            
            # Add rating weight if available
            if interaction.get('rating'):
                rating = interaction['rating']
                if rating >= 4:
                    weight += 1.0
                elif rating >= 3:
                    weight += 0.5
                else:
                    weight -= 0.5  # Negative signal for low ratings
            
            # Add time spent weight (if available)
            time_spent = interaction.get('time_spent_seconds', 0)
            if time_spent > 180:  # More than 3 minutes
                weight += 0.3
            elif time_spent > 60:  # More than 1 minute
                weight += 0.1
            
            if weight > 0:
                paper_idx = self.papers.index(paper_id)
                paper_embedding = self.embeddings[paper_idx]
                weighted_embeddings.append((paper_embedding, weight))
        
        if not weighted_embeddings:
            # Return zero vector if no valid interactions
            return np.zeros(self.embeddings.shape[1])
        
        # Build weighted average of embeddings
        user_profile = np.zeros(self.embeddings.shape[1])
        total_weight = 0
        
        for embedding, weight in weighted_embeddings:
            user_profile += weight * embedding
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            user_profile /= total_weight
        
        return user_profile
    
    def get_similar_papers(self, paper_id: str, num_similar: int = 5) -> List[Tuple[str, float]]:
        """Find papers similar to a given paper using semantic similarity"""
        if paper_id not in self.papers:
            return []
        
        paper_idx = self.papers.index(paper_id)
        similarities = self.paper_similarity[paper_idx]
        
        similar_papers = []
        for i, similarity in enumerate(similarities):
            if i != paper_idx:
                similar_papers.append((self.papers[i], similarity))
        
        similar_papers.sort(key=lambda x: x[1], reverse=True)
        return [(paper_id, float(score)) for paper_id, score in similar_papers[:num_similar]]
    
    def get_papers_by_interest(self, interests: List[str], num_papers: int = 10, 
                          use_category_filter: bool = True) -> List[Tuple[str, float]]:
        """Get papers matching user's initial interests using semantic matching"""
        # Create a query from user interests
        interest_text = " [SEP] ".join(interests)
        
        # Generate embedding for the interest query
        interest_embedding = self.model.encode([interest_text])[0]
        
        # Infer categories from interests (simple keyword matching)
        interest_lower = interest_text.lower()
        inferred_categories = []
        
        category_keywords = {
            'cs.CL': ['nlp', 'language', 'text', 'translation', 'linguistic'],
            'cs.CV': ['vision', 'image', 'visual', 'detection', 'segmentation', 'video'],
            'cs.LG': ['machine learning', 'deep learning', 'neural network'],
            'cs.RO': ['robot', 'manipulation', 'control', 'visuomotor'],
            'quant-ph': ['quantum', 'qubit', 'entanglement'],
            'cs.AI': ['artificial intelligence', 'planning', 'reasoning']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in interest_lower for keyword in keywords):
                inferred_categories.append(category)
        
        # Filter by inferred categories if found
        if use_category_filter and inferred_categories:
            related_categories = self._get_related_categories(inferred_categories)
            
            candidate_indices = []
            candidate_papers = []
            for i, paper_id in enumerate(self.papers):
                paper_category = self.paper_features[paper_id].get('arxiv_category')
                if paper_category in related_categories:
                    candidate_indices.append(i)
                    candidate_papers.append(paper_id)
            
            if candidate_indices:
                candidate_embeddings = self.embeddings[candidate_indices]
                similarities = cosine_similarity([interest_embedding], candidate_embeddings)[0]
                paper_scores = [(candidate_papers[i], similarities[i]) for i in range(len(candidate_papers))]
            else:
                # Fallback to all papers
                similarities = cosine_similarity([interest_embedding], self.embeddings)[0]
                paper_scores = [(self.papers[i], similarities[i]) for i in range(len(self.papers))]
        else:
            # No filtering
            similarities = cosine_similarity([interest_embedding], self.embeddings)[0]
            paper_scores = [(self.papers[i], similarities[i]) for i in range(len(self.papers))]
        
        paper_scores.sort(key=lambda x: x[1], reverse=True)
        return [(paper_id, float(score)) for paper_id, score in paper_scores[:num_papers]]
    
    def _get_popular_papers(self, num_papers: int) -> List[Tuple[str, float]]:
        """Fallback: return popular papers based on citation count"""
        # Sort by citation count (stored in paper_features)
        papers_with_citations = []
        for paper_id in self.papers:
            citation_count = self.paper_features[paper_id].get('citation_count', 0)
            papers_with_citations.append((paper_id, citation_count))
        
        papers_with_citations.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores to 0-1 range
        max_citations = papers_with_citations[0][1] if papers_with_citations else 1
        normalized = [(paper_id, citations / max(max_citations, 1)) 
                      for paper_id, citations in papers_with_citations]
        
        return normalized[:num_papers]