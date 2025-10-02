import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, UTC
from typing import List, Dict
import time
import json


class ArxivScraper:
    """
    Scraper for arXiv papers using the official arXiv API.

    API Documentation: https://arxiv.org/help/api/user-manual
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, delay: float = 3.0):
        """Initialize the scraper."""
        self.delay = delay
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()

    def search_papers(
        self,
        query: str = None,
        category: str = None,
        categories: List[str] = None,
        max_results: int = 100,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> List[Dict]:
        """Search for papers on arXiv."""
        self._rate_limit()

        search_query = []
        if query:
            search_query.append(f"all:{query}")
        if category:
            search_query.append(f"cat:{category}")
        if categories:
            or_query = "+OR+".join([f"cat:{c}" for c in categories])
            search_query.append(f"({or_query})")

        if not search_query:
            search_query = ["all:*"]  # fallback: everything

        search_query = "+AND+".join(search_query)

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        response = requests.get(self.BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        return self._parse_response(response.text)

    def get_papers_by_category(
        self, category: str, max_results: int = 100, start: int = 0
    ) -> List[Dict]:
        """Get papers from a specific arXiv category."""
        return self.search_papers(category=category, max_results=max_results, start=start)

    def get_recent_papers(
        self, category: str, days: int = 7, max_results: int = 100
    ) -> List[Dict]:
        """Get recent papers from the last N days."""
        papers = self.search_papers(
            category=category,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending",
        )

        cutoff = datetime.now(UTC) - timedelta(days=days)
        filtered = []
        for p in papers:
            try:
                pub_date = datetime.strptime(
                    p["publication_date"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=UTC)
                if pub_date >= cutoff:
                    filtered.append(p)
            except Exception:
                continue
        return filtered

    def _parse_response(self, xml_text: str) -> List[Dict]:
        """Parse arXiv API XML response into paper dictionaries."""
        root = ET.fromstring(xml_text)

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        papers = []
        for entry in root.findall("atom:entry", ns):
            paper = self._parse_entry(entry, ns)
            papers.append(paper)

        return papers

    def _parse_entry(self, entry, ns) -> Dict:
        """Parse a single paper entry from XML."""
        arxiv_id = (
            entry.find("atom:id", ns).text.split("/abs/")[-1]
            if entry.find("atom:id", ns) is not None
            else None
        )

        title = (
            entry.find("atom:title", ns).text.strip().replace("\n", " ")
            if entry.find("atom:title", ns) is not None
            else ""
        )

        abstract = (
            entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            if entry.find("atom:summary", ns) is not None
            else ""
        )

        authors = [
            author.find("atom:name", ns).text
            for author in entry.findall("atom:author", ns)
            if author.find("atom:name", ns) is not None
        ]

        categories = [c.get("term") for c in entry.findall("atom:category", ns)]

        primary_category = entry.find("arxiv:primary_category", ns)
        primary_cat = (
            primary_category.get("term")
            if primary_category is not None
            else categories[0] if categories else None
        )

        published = (
            entry.find("atom:published", ns).text
            if entry.find("atom:published", ns) is not None
            else ""
        )

        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_link = link.get("href")
                break

        return {
            "paper_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "primary_category": primary_cat,
            "all_categories": categories,
            "publication_date": published,
            "pdf_url": pdf_link,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}"
            if arxiv_id
            else None,
        }

    def bulk_scrape_category(
        self, category: str, total_papers: int = 1000, batch_size: int = 100
    ) -> List[Dict]:
        """Scrape a large number of papers from a category using pagination."""
        all_papers = []
        batch_size = min(batch_size, 100)

        for start in range(0, total_papers, batch_size):
            print(f"Fetching {category} papers {start} to {start + batch_size}...")
            papers = self.get_papers_by_category(
                category=category, max_results=batch_size, start=start
            )
            if not papers:
                print("No more papers found.")
                break
            all_papers.extend(papers)
            print(f"Total papers scraped so far: {len(all_papers)}")

        return all_papers

    def bulk_scrape_all(
        self, total_papers: int = 1000, batch_size: int = 100
    ) -> List[Dict]:
        """Scrape papers across all categories."""
        all_papers = []
        batch_size = min(batch_size, 100)

        for start in range(0, total_papers, batch_size):
            print(f"Fetching ALL papers {start} to {start + batch_size}...")
            papers = self.search_papers(
                query="*",  # all papers
                max_results=batch_size,
                start=start,
                sort_by="submittedDate",
                sort_order="descending",
            )
            if not papers:
                print("No more papers found.")
                break
            all_papers.extend(papers)
            print(f"Total papers scraped so far: {len(all_papers)}")

        # Deduplicate by paper_id
        unique = {p["paper_id"]: p for p in all_papers}
        return list(unique.values())

    def save_to_json(self, papers: List[Dict], filename: str):
        """Save papers to a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(papers)} papers to {filename}")


# Example usage
if __name__ == "__main__":
    scraper = ArxivScraper()

    print("Scraping 1000 most recent arXiv papers (all categories)...")
    all_papers = scraper.bulk_scrape_all(total_papers=1000, batch_size=100)
    scraper.save_to_json(all_papers, "arxiv_all_papers.json")

    print("\nScraping 200 Computer Vision papers...")
    cv_papers = scraper.bulk_scrape_category("cs.CV", total_papers=200, batch_size=100)
    scraper.save_to_json(cv_papers, "arxiv_cv_papers.json")