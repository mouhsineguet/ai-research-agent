import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ArxivAPI:
    """ArXiv API client"""
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search ArXiv for papers"""
        try:
            search_query = f"all:{query}"
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_arxiv_response(content)
                    else:
                        logger.error(f"ArXiv API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error searching ArXiv: {str(e)}")
            return []

    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ArXiv XML response"""
        papers = []
        try:
            root = ET.fromstring(xml_content)
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            entries = root.findall('atom:entry', namespaces)
            for entry in entries:
                try:
                    title = entry.find('atom:title', namespaces)
                    title_text = title.text.strip().replace('\n', ' ') if title is not None else "Unknown Title"
                    summary = entry.find('atom:summary', namespaces)
                    abstract_text = summary.text.strip().replace('\n', ' ') if summary is not None else "No abstract available"
                    authors = []
                    author_elements = entry.findall('atom:author', namespaces)
                    for author in author_elements:
                        name = author.find('atom:name', namespaces)
                        if name is not None:
                            authors.append(name.text)
                    published = entry.find('atom:published', namespaces)
                    pub_date = published.text if published is not None else ""
                    year = 0
                    if pub_date:
                        try:
                            year = int(pub_date[:4])
                        except:
                            pass
                    id_elem = entry.find('atom:id', namespaces)
                    url = id_elem.text if id_elem is not None else ""
                    categories = []
                    category_elements = entry.findall('atom:category', namespaces)
                    for cat in category_elements:
                        term = cat.get('term')
                        if term:
                            categories.append(term)
                    paper = {
                        'title': title_text,
                        'authors': ', '.join(authors),
                        'abstract': abstract_text,
                        'year': year,
                        'url': url,
                        'source': 'arXiv',
                        'categories': categories,
                        'citation_count': 0  # ArXiv doesn't provide citation counts
                    }
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing ArXiv entry: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error parsing ArXiv XML: {str(e)}")
        return papers

# Optional: test block for standalone testing
if __name__ == "__main__":
    import asyncio
    async def test_arxiv():
        api = ArxivAPI()
        query = "transformer neural networks"
        papers = await api.search(query, max_results=3)
        print(f"Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {paper['authors']}")
            print(f"   Year: {paper['year']}")
            print(f"   URL: {paper['url']}")
            print(f"   Abstract: {paper['abstract'][:120]}...")
    asyncio.run(test_arxiv()) 