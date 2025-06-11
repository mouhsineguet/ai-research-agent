"""
Tool Agent - Handles literature search across multiple APIs
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import json
import re

logger = logging.getLogger(__name__)

class ToolAgent:
    """Agent responsible for searching and retrieving papers from various sources"""
    
    def __init__(self):
        self.sources = {
            'arxiv': ArxivAPI(),
            'pubmed': PubMedAPI(),
            'semantic_scholar': SemanticScholarAPI()
        }
    
    async def search_papers(self, query: str, sources: List[str], max_papers: int = 20) -> List[Dict[str, Any]]:
        """Search for papers across specified sources"""
        
        logger.info(f"Searching for papers: {query} across sources: {sources}")
        
        all_papers = []
        papers_per_source = max_papers // len(sources) if sources else max_papers
        
        # Search each source concurrently
        tasks = []
        for source in sources:
            if source in self.sources:
                task = self.sources[source].search(query, papers_per_source)
                tasks.append(task)
        
        # Execute searches
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Error searching {sources[i]}: {str(result)}")
                continue
            
            if isinstance(result, list):
                all_papers.extend(result)
        
        # Deduplicate and rank papers
        unique_papers = self._deduplicate_papers(all_papers)
        ranked_papers = self._rank_papers(unique_papers, query)
        
        # Return top papers
        final_papers = ranked_papers[:max_papers]
        
        logger.info(f"✅ Found {len(final_papers)} papers")
        return final_papers
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity"""
        
        if not papers:
            return []
        
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            
            # Simple deduplication by title
            title_key = re.sub(r'[^\w\s]', '', title)
            title_key = ' '.join(title_key.split())
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _rank_papers(self, papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank papers by relevance and quality metrics"""
        
        query_terms = set(query.lower().split())
        
        for paper in papers:
            score = 0
            
            # Title relevance
            title = paper.get('title', '').lower()
            title_matches = sum(1 for term in query_terms if term in title)
            score += title_matches * 3
            
            # Abstract relevance
            abstract = paper.get('abstract', '').lower()
            abstract_matches = sum(1 for term in query_terms if term in abstract)
            score += abstract_matches
            
            # Recency bonus (papers from last 2 years)
            try:
                year = paper.get('year', 0)
                current_year = datetime.now().year
                if year >= current_year - 2:
                    score += 2
                elif year >= current_year - 5:
                    score += 1
            except:
                pass
            
            # Citation count bonus
            citations = paper.get('citation_count', 0)
            if citations > 100:
                score += 3
            elif citations > 50:
                score += 2
            elif citations > 10:
                score += 1
            
            paper['relevance_score'] = score
        
        # Sort by score descending
        return sorted(papers, key=lambda x: x.get('relevance_score', 0), reverse=True)


class ArxivAPI:
    """ArXiv API client"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search ArXiv for papers"""
        
        try:
            # Prepare search parameters
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
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                try:
                    # Extract paper information
                    title = entry.find('atom:title', namespaces)
                    title_text = title.text.strip().replace('\n', ' ') if title is not None else "Unknown Title"
                    
                    summary = entry.find('atom:summary', namespaces)
                    abstract_text = summary.text.strip().replace('\n', ' ') if summary is not None else "No abstract available"
                    
                    # Authors
                    authors = []
                    author_elements = entry.findall('atom:author', namespaces)
                    for author in author_elements:
                        name = author.find('atom:name', namespaces)
                        if name is not None:
                            authors.append(name.text)
                    
                    # Publication date
                    published = entry.find('atom:published', namespaces)
                    pub_date = published.text if published is not None else ""
                    year = 0
                    if pub_date:
                        try:
                            year = int(pub_date[:4])
                        except:
                            pass
                    
                    # URL
                    id_elem = entry.find('atom:id', namespaces)
                    url = id_elem.text if id_elem is not None else ""
                    
                    # Categories
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


class PubMedAPI:
    """PubMed API client"""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.search_url = f"{self.base_url}/esearch.fcgi"
        self.fetch_url = f"{self.base_url}/efetch.fcgi"
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search PubMed for papers"""
        
        try:
            # First, search for PMIDs
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            async with aiohttp.ClientSession() as session:
                # Get PMIDs
                async with session.get(self.search_url, params=search_params) as response:
                    if response.status != 200:
                        logger.error(f"PubMed search error: {response.status}")
                        return []
                    
                    search_data = await response.json()
                    pmids = search_data.get('esearchresult', {}).get('idlist', [])
                
                if not pmids:
                    return []
                
                # Fetch paper details
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(pmids),
                    'retmode': 'xml'
                }
                
                async with session.get(self.fetch_url, params=fetch_params) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_pubmed_response(content)
                    else:
                        logger.error(f"PubMed fetch error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return []
    
    def _parse_pubmed_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            articles = root.findall('.//PubmedArticle')
            
            for article in articles:
                try:
                    # Title
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "Unknown Title"
                    
                    # Abstract
                    abstract_elem = article.find('.//AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                    
                    # Authors
                    authors = []
                    author_list = article.find('.//AuthorList')
                    if author_list is not None:
                        for author in author_list.findall('Author'):
                            last_name = author.find('LastName')
                            first_name = author.find('ForeName')
                            if last_name is not None and first_name is not None:
                                authors.append(f"{first_name.text} {last_name.text}")
                    
                    # Publication year
                    year_elem = article.find('.//PubDate/Year')
                    year = int(year_elem.text) if year_elem is not None else 0
                    
                    # PMID for URL
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                    
                    # Journal
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
                    
                    paper = {
                        'title': title,
                        'authors': ', '.join(authors),
                        'abstract': abstract,
                        'year': year,
                        'url': url,
                        'source': 'PubMed',
                        'journal': journal,
                        'pmid': pmid,
                        'citation_count': 0  # PubMed doesn't provide citation counts directly
                    }
                    
                    papers.append(paper)
                
                except Exception as e:
                    logger.warning(f"Error parsing PubMed article: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing PubMed XML: {str(e)}")
        
        return papers


class SemanticScholarAPI:
    """Semantic Scholar API client"""
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers"""
        
        try:
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'paperId,title,abstract,authors,year,citationCount,url,venue,referenceCount'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_semantic_scholar_response(data)
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {str(e)}")
            return []
    
    def _parse_semantic_scholar_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Semantic Scholar JSON response"""
        
        papers = []
        
        try:
            papers_data = data.get('data', [])
            
            for paper_data in papers_data:
                try:
                    # Extract authors
                    authors = []
                    author_list = paper_data.get('authors', [])
                    for author in author_list:
                        name = author.get('name', '')
                        if name:
                            authors.append(name)
                    
                    paper = {
                        'title': paper_data.get('title', 'Unknown Title'),
                        'authors': ', '.join(authors),
                        'abstract': paper_data.get('abstract', 'No abstract available'),
                        'year': paper_data.get('year', 0),
                        'url': paper_data.get('url', ''),
                        'source': 'Semantic Scholar',
                        'venue': paper_data.get('venue', 'Unknown Venue'),
                        'citation_count': paper_data.get('citationCount', 0),
                        'reference_count': paper_data.get('referenceCount', 0),
                        'paper_id': paper_data.get('paperId', '')
                    }
                    
                    papers.append(paper)
                
                except Exception as e:
                    logger.warning(f"Error parsing Semantic Scholar paper: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar response: {str(e)}")
        
        return papers


# Example usage and testing
if __name__ == "__main__":
    async def test_tool_agent():
        """Test the tool agent"""
        agent = ToolAgent()
        
        query = "transformer neural networks"
        sources = ['arxiv', 'semantic_scholar']
        
        papers = await agent.search_papers(query, sources, max_papers=5)
        
        print(f"Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {paper['authors']}")
            print(f"   Year: {paper['year']}")
            print(f"   Source: {paper['source']}")
            print(f"   Score: {paper.get('relevance_score', 'N/A')}")
    
    # Run test
    asyncio.run(test_tool_agent()) 