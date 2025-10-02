import sqlite3
import json
import time
import asyncio
from typing import Dict, List, Optional
import aiosqlite

class DatabaseManager:
    def __init__(self, db_path: str = "recommendation_engine.db"):
        self.db_path = db_path
        self.connection = None
    
    async def initialize(self):
        self.connection = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        print(f"Database initialized at {self.db_path}")
    
    async def _create_tables(self):
        # Users table - add initial_interests field
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_interactions INTEGER DEFAULT 0,
                initial_interests TEXT
            )
        """)
        
        # Papers table (replacing items)
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT NOT NULL,
                arxiv_category TEXT NOT NULL,
                keywords TEXT,
                publication_date TEXT,
                citation_count INTEGER DEFAULT 0,
                pdf_url TEXT,
                average_rating REAL DEFAULT 0.0,
                total_ratings INTEGER DEFAULT 0
            )
        """)
        
        # User interactions with papers
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                rating REAL,
                time_spent_seconds INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, paper_id, interaction_type)
            )
        """)
        
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON user_interactions(user_id)")
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_interactions_paper ON user_interactions(paper_id)")
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(arxiv_category)")
        
        await self.connection.commit()
    
    async def create_or_update_user(self, user_id: str):
        await self.connection.execute("""
            INSERT INTO users (user_id, last_active)
            VALUES (?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                last_active = CURRENT_TIMESTAMP
        """, (user_id,))
        await self.connection.commit()
    
    async def get_user_stats(self, user_id: str) -> Optional[Dict]:
        cursor = await self.connection.execute("""
            SELECT u.*, COUNT(ui.id) as interaction_count
            FROM users u
            LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
            WHERE u.user_id = ?
            GROUP BY u.user_id
        """, (user_id,))
        
        row = await cursor.fetchone()
        if row:
            return {
                "user_id": row[0],
                "created_at": row[1],
                "last_active": row[2],
                "total_interactions": row[3],
                "interaction_count": row[4] or 0
            }
        return None
    
    async def get_user_interactions(self, user_id: str, limit: int = 50) -> List[Dict]:
        cursor = await self.connection.execute("""
            SELECT ui.paper_id, ui.rating, ui.timestamp, ui.interaction_type, ui.time_spent_seconds
            FROM user_interactions ui
            WHERE ui.user_id = ?
            ORDER BY ui.timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        
        rows = await cursor.fetchall()
        return [{
            "paper_id": row[0],
            "rating": row[1],
            "timestamp": row[2],
            "interaction_type": row[3],
            "time_spent_seconds": row[4]
        } for row in rows]
    
    async def get_all_interactions(self) -> List[Dict]:
        cursor = await self.connection.execute("""
            SELECT user_id, paper_id, rating, interaction_type, time_spent_seconds
            FROM user_interactions
            ORDER BY timestamp
        """)
        
        rows = await cursor.fetchall()
        return [{
            "user_id": row[0],
            "paper_id": row[1],
            "rating": row[2],
            "interaction_type": row[3],
            "time_spent_seconds": row[4]
        } for row in rows]

    async def create_or_update_paper(self, paper_data: Dict):
        """Create or update a research paper"""
        await self.connection.execute("""
            INSERT INTO papers (paper_id, title, abstract, authors, arxiv_category, keywords, publication_date, citation_count, pdf_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                title = excluded.title,
                abstract = excluded.abstract,
                authors = excluded.authors,
                arxiv_category = excluded.arxiv_category,
                keywords = excluded.keywords,
                publication_date = excluded.publication_date,
                citation_count = excluded.citation_count,
                pdf_url = excluded.pdf_url
        """, (
            paper_data['paper_id'],
            paper_data['title'],
            paper_data['abstract'],
            paper_data['authors'],
            paper_data['arxiv_category'],
            paper_data.get('keywords', ''),
            paper_data.get('publication_date', ''),
            paper_data.get('citation_count', 0),
            paper_data.get('pdf_url', '')
        ))
        await self.connection.commit()

    async def get_all_papers(self) -> List[Dict]:
        """Get all papers with ratings"""
        cursor = await self.connection.execute("""
            SELECT p.*, AVG(ui.rating) as avg_rating, COUNT(ui.id) as rating_count
            FROM papers p
            LEFT JOIN user_interactions ui ON p.paper_id = ui.paper_id AND ui.rating IS NOT NULL
            GROUP BY p.paper_id
            ORDER BY avg_rating DESC
        """)
        
        rows = await cursor.fetchall()
        return [{
            "paper_id": row[0],
            "title": row[1],
            "abstract": row[2],
            "authors": row[3],
            "arxiv_category": row[4],
            "keywords": row[5],
            "publication_date": row[6],
            "citation_count": row[7],
            "pdf_url": row[8],
            "current_avg_rating": row[10] or 0.0,
            "current_rating_count": row[11] or 0
        } for row in rows]

    async def get_papers_by_category(self, category: str) -> List[Dict]:
        """Get papers by ArXiv category"""
        cursor = await self.connection.execute("""
            SELECT * FROM papers
            WHERE arxiv_category = ?
            ORDER BY citation_count DESC
        """, (category,))
        
        rows = await cursor.fetchall()
        return [{
            "paper_id": row[0],
            "title": row[1],
            "abstract": row[2],
            "authors": row[3],
            "arxiv_category": row[4],
            "keywords": row[5],
            "publication_date": row[6],
            "citation_count": row[7],
            "pdf_url": row[8]
        } for row in rows]

    async def record_interaction(self, user_id: str, paper_id: str, interaction_type: str, rating: float = None, time_spent: int = None):
        """Record user interaction with a paper"""
        await self.create_or_update_user(user_id)
        
        await self.connection.execute("""
            INSERT INTO user_interactions (user_id, paper_id, interaction_type, rating, time_spent_seconds)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, paper_id, interaction_type) DO UPDATE SET
                rating = excluded.rating,
                time_spent_seconds = excluded.time_spent_seconds,
                timestamp = CURRENT_TIMESTAMP
        """, (user_id, paper_id, interaction_type, rating, time_spent))
        
        await self.connection.execute("""
            UPDATE users SET 
                total_interactions = (
                    SELECT COUNT(*) FROM user_interactions WHERE user_id = ?
                ),
                last_active = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (user_id, user_id))
        
        await self.connection.commit()

    async def set_user_initial_interests(self, user_id: str, interests: List[str]):
        """Set user's initial research interests"""
        await self.connection.execute("""
            UPDATE users SET initial_interests = ?
            WHERE user_id = ?
        """, (','.join(interests), user_id))
        await self.connection.commit()

    async def get_user_initial_interests(self, user_id: str) -> List[str]:
        """Get user's initial interests"""
        cursor = await self.connection.execute("""
            SELECT initial_interests FROM users WHERE user_id = ?
        """, (user_id,))
        
        row = await cursor.fetchone()
        if row and row[0]:
            return row[0].split(',')
        return []
    
    async def close(self):
        if self.connection:
            await self.connection.close()
            print("Database connection closed")