"""
Summarizer Agent - Analyzes and synthesizes research papers
"""

import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import asyncio

logger = logging.getLogger(__name__)

class PaperSummary(BaseModel):
    """Structure for individual paper summary"""
    paper_title: str = Field(description="Title of the paper")
    key_findings: List[str] = Field(description="Main findings and contributions")
    methodology: str = Field(description="Research methodology used")
    significance: str = Field(description="Significance and impact of the work")
    limitations: List[str] = Field(description="Limitations mentioned or identified")
    relevance_score: int = Field(description="Relevance to the query (1-10)")

class ResearchSynthesis(BaseModel):
    """Structure for overall research synthesis"""
    main_themes: List[str] = Field(description="Main themes across papers")
    key_insights: List[str] = Field(description="Key insights from the literature")
    research_gaps: List[str] = Field(description="Identified research gaps")
    conflicting_findings: List[str] = Field(description="Any conflicting findings")
    consensus_areas: List[str] = Field(description="Areas of consensus")
    future_directions: List[str] = Field(description="Suggested future research directions")

class SummarizerAgent:
    """Agent responsible for summarizing and synthesizing research papers"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.summary_parser = PydanticOutputParser(pydantic_object=PaperSummary)
        self.synthesis_parser = PydanticOutputParser(pydantic_object=ResearchSynthesis)
        
        # Create prompts
        self.individual_summary_prompt = self._create_individual_summary_prompt()
        self.synthesis_prompt = self._create_synthesis_prompt()
    
    def _create_individual_summary_prompt(self) -> ChatPromptTemplate:
        """Create prompt for individual paper summaries"""
        
        system_prompt = """You are a Research Summarization Expert. Analyze the given research paper and provide a structured summary.

Focus on:
1. Key findings and contributions
2. Methodology and approach
3. Significance and impact
4. Limitations and weaknesses
5. Relevance to the research query

Be objective, accurate, and concise. Extract the most important information that would be useful for a literature review.

{format_instructions}"""

        human_prompt = """Research Query: {query}

Paper Details:
Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Please provide a structured summary of this paper in relation to the research query."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_synthesis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for research synthesis"""
        
        system_prompt = """You are a Research Synthesis Expert. Analyze multiple research papers and provide a comprehensive synthesis.

Your task is to:
1. Identify main themes across papers
2. Extract key insights and patterns
3. Identify research gaps and opportunities
4. Note any conflicting findings
5. Highlight areas of consensus
6. Suggest future research directions

Be analytical, objective, and provide actionable insights for researchers.

{format_instructions}"""

        human_prompt = """Research Query: {query}

Paper Summaries:
{summaries}

Please provide a comprehensive synthesis of these research papers."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    async def summarize_papers(self, papers: List[Dict[str, Any]], query: str, style: str = "detailed") -> List[Dict[str, Any]]:
        """Summarize individual papers and create overall synthesis"""
        
        logger.info(f"Summarizing {len(papers)} papers in {style} style")
        
        # Summarize individual papers
        individual_summaries = []
        for i, paper in enumerate(papers):
            try:
                summary = await self._summarize_individual_paper(paper, query)
                individual_summaries.append({
                    'paper_index': i,
                    'original_paper': paper,
                    'summary': summary
                })
            except Exception as e:
                logger.warning(f"Error summarizing paper {i}: {str(e)}")
                # Create fallback summary
                fallback_summary = self._create_fallback_summary(paper, query)
                individual_summaries.append({
                    'paper_index': i,
                    'original_paper': paper,
                    'summary': fallback_summary
                })
        
        # Create overall synthesis
        try:
            synthesis = await self._create_synthesis(individual_summaries, query)
        except Exception as e:
            logger.error(f"Error creating synthesis: {str(e)}")
            synthesis = self._create_fallback_synthesis(individual_summaries, query)
        
        # Format output based on style
        formatted_summaries = self._format_summaries(individual_summaries, synthesis, style)
        
        logger.info("✅ Paper summarization completed")
        return formatted_summaries
    
    async def _summarize_individual_paper(self, paper: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Summarize a single paper"""
        
        # Prepare prompt inputs
        prompt_inputs = {
            "query": query,
            "title": paper.get('title', 'Unknown Title'),
            "authors": paper.get('authors', 'Unknown Authors'),
            "year": paper.get('year', 'Unknown Year'),
            "abstract": paper.get('abstract', 'No abstract available'),
            "format_instructions": self.summary_parser.get_format_instructions()
        }
        
        # Generate summary
        response = await self.llm.ainvoke(
            self.individual_summary_prompt.format_messages(**prompt_inputs)
        )
        
        # Parse response
        try:
            summary = self.summary_parser.parse(response.content)
            return summary.dict()
        except Exception as e:
            logger.warning(f"Failed to parse summary, using fallback: {str(e)}")
            return self._create_fallback_summary(paper, query)
    
    async def _create_synthesis(self, summaries: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Create overall research synthesis"""
        
        # Prepare summaries text
        summaries_text = ""
        for i, summary_data in enumerate(summaries):
            summary = summary_data['summary']
            summaries_text += f"\nPaper {i+1}:\n"
            summaries_text += f"Title: {summary.get('paper_title', 'Unknown')}\n"
            summaries_text += f"Key Findings: {', '.join(summary.get('key_findings', []))}\n"
            summaries_text += f"Methodology: {summary.get('methodology', 'Unknown')}\n"
            summaries_text += f"Significance: {summary.get('significance', 'Unknown')}\n"
            summaries_text += f"Limitations: {', '.join(summary.get('limitations', []))}\n"
            summaries_text += f"Relevance: {summary.get('relevance_score', 'Unknown')}/10\n"
        
        # Prepare prompt inputs
        prompt_inputs = {
            "query": query,
            "summaries": summaries_text,
            "format_instructions": self.synthesis_parser.get_format_instructions()
        }
        
        # Generate synthesis
        response = await self.llm.ainvoke(
            self.synthesis_prompt.format_messages(**prompt_inputs)
        )
        
        # Parse response
        try:
            synthesis = self.synthesis_parser.parse(response.content)
            return synthesis.dict()
        except Exception as e:
            logger.warning(f"Failed to parse synthesis, using fallback: {str(e)}")
            return self._create_fallback_synthesis(summaries, query)
    
    def _create_fallback_summary(self, paper: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Create a basic fallback summary"""
        
        return {
            'paper_title': paper.get('title', 'Unknown Title'),
            'key_findings': ['Key findings extraction failed'],
            'methodology': 'Methodology not analyzed',
            'significance': 'Significance assessment not available',
            'limitations': ['Limitations not identified'],
            'relevance_score': 5
        }
    
    def _create_fallback_synthesis(self, summaries: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Create a basic fallback synthesis"""
        
        return {
            'main_themes': ['Theme analysis not available'],
            'key_insights': ['Insights extraction failed'],
            'research_gaps': ['Gap analysis not completed'],
            'conflicting_findings': ['No conflicts identified'],
            'consensus_areas': ['Consensus analysis not available'],
            'future_directions': ['Future directions not determined']
        }
    
    def _format_summaries(self, individual_summaries: List[Dict[str, Any]], synthesis: Dict[str, Any], style: str) -> List[Dict[str, Any]]:
        """Format summaries based on requested style"""
        
        if style == "brief":
            return self._format_brief_summaries(individual_summaries, synthesis)
        elif style == "technical":
            return self._format_technical_summaries(individual_summaries, synthesis)
        else:  # detailed
            return self._format_detailed_summaries(individual_summaries, synthesis)
    
    def _format_brief_summaries(self, individual_summaries: List[Dict[str, Any]], synthesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format brief summaries"""
        
        brief_summaries = []
        
        # Brief individual summaries
        for summary_data in individual_summaries:
            summary = summary_data['summary']
            paper = summary_data['original_paper']
            
            brief_summary = {
                'type': 'individual_brief',
                'title': summary.get('paper_title', 'Unknown'),
                'authors': paper.get('authors', 'Unknown'),
                'year': paper.get('year', 'Unknown'),
                'key_points': summary.get('key_findings', [])[:2],  # Top 2 findings
                'relevance': summary.get('relevance_score', 5),
                'url': paper.get('url', '')
            }
            brief_summaries.append(brief_summary)
        
        # Brief synthesis
        brief_synthesis = {
            'type': 'synthesis_brief',
            'main_themes': synthesis.get('main_themes', [])[:3],  # Top 3 themes
            'key_insights': synthesis.get('key_insights', [])[:3],  # Top 3 insights
            'research_gaps': synthesis.get('research_gaps', [])[:2]  # Top 2 gaps
        }
        brief_summaries.append(brief_synthesis)
        
        return brief_summaries
    
    def _format_detailed_summaries(self, individual_summaries: List[Dict[str, Any]], synthesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format detailed summaries"""
        
        detailed_summaries = []
        
        # Detailed individual summaries
        for summary_data in individual_summaries:
            summary = summary_data['summary']
            paper = summary_data['original_paper']
            
            detailed_summary = {
                'type': 'individual_detailed',
                'title': summary.get('paper_title', 'Unknown'),
                'authors': paper.get('authors', 'Unknown'),
                'year': paper.get('year', 'Unknown'),
                'abstract': paper.get('abstract', 'No abstract')[:500] + "...",
                'key_findings': summary.get('key_findings', []),
                'methodology': summary.get('methodology', 'Unknown'),
                'significance': summary.get('significance', 'Unknown'),
                'limitations': summary.get('limitations', []),
                'relevance_score': summary.get('relevance_score', 5),
                'source': paper.get('source', 'Unknown'),
                'url': paper.get('url', ''),
                'citation_count': paper.get('citation_count', 0)
            }
            detailed_summaries.append(detailed_summary)
        
        # Detailed synthesis
        detailed_synthesis = {
            'type': 'synthesis_detailed',
            'main_themes': synthesis.get('main_themes', []),
            'key_insights': synthesis.get('key_insights', []),
            'research_gaps': synthesis.get('research_gaps', []),
            'conflicting_findings': synthesis.get('conflicting_findings', []),
            'consensus_areas': synthesis.get('consensus_areas', []),
            'future_directions': synthesis.get('future_directions', [])
        }
        detailed_summaries.append(detailed_synthesis)
        
        return detailed_summaries
    
    def _format_technical_summaries(self, individual_summaries: List[Dict[str, Any]], synthesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format technical summaries with focus on methodology and implementation details"""
        
        technical_summaries = []
        
        # Technical individual summaries
        for summary_data in individual_summaries:
            summary = summary_data['summary']
            paper = summary_data['original_paper']
            
            technical_summary = {
                'type': 'individual_technical',
                'title': summary.get('paper_title', 'Unknown'),
                'authors': paper.get('authors', 'Unknown'),
                'year': paper.get('year', 'Unknown'),
                'venue': paper.get('venue', 'Unknown'),
                'doi': paper.get('doi', ''),
                'methodology': summary.get('methodology', 'Unknown'),
                'key_findings': summary.get('key_findings', []),
                'technical_contributions': self._extract_technical_contributions(summary),
                'experimental_setup': self._extract_experimental_details(summary),
                'limitations': summary.get('limitations', []),
                'reproducibility': self._assess_reproducibility(summary, paper),
                'relevance_score': summary.get('relevance_score', 5),
                'citation_count': paper.get('citation_count', 0),
                'source': paper.get('source', 'Unknown'),
                'url': paper.get('url', '')
            }
            technical_summaries.append(technical_summary)
        
        # Technical synthesis
        technical_synthesis = {
            'type': 'synthesis_technical',
            'methodological_trends': self._identify_methodological_trends(individual_summaries),
            'technical_gaps': synthesis.get('research_gaps', []),
            'implementation_challenges': self._identify_implementation_challenges(individual_summaries),
            'reproducibility_concerns': self._identify_reproducibility_issues(individual_summaries),
            'future_technical_directions': synthesis.get('future_directions', []),
            'recommended_methodologies': self._recommend_methodologies(individual_summaries),
            'conflicting_approaches': synthesis.get('conflicting_findings', [])
        }
        technical_summaries.append(technical_synthesis)
        
        return technical_summaries
    
    def _extract_technical_contributions(self, summary: Dict[str, Any]) -> List[str]:
        """Extract technical contributions from summary"""
        contributions = []
        key_findings = summary.get('key_findings', [])
        
        for finding in key_findings:
            if any(keyword in finding.lower() for keyword in ['algorithm', 'method', 'approach', 'technique', 'model']):
                contributions.append(finding)
        
        return contributions if contributions else ['No specific technical contributions identified']
    
    def _extract_experimental_details(self, summary: Dict[str, Any]) -> str:
        """Extract experimental setup details"""
        methodology = summary.get('methodology', '')
        
        if any(keyword in methodology.lower() for keyword in ['experiment', 'evaluation', 'dataset', 'benchmark']):
            return methodology
        
        return 'Experimental setup not clearly described'
    
    def _assess_reproducibility(self, summary: Dict[str, Any], paper: Dict[str, Any]) -> str:
        """Assess reproducibility based on available information"""
        methodology = summary.get('methodology', '').lower()
        
        if 'code' in paper.get('url', '').lower() or 'github' in paper.get('url', '').lower():
            return 'High - Code available'
        elif any(keyword in methodology for keyword in ['dataset', 'open', 'public']):
            return 'Medium - Public datasets used'
        elif any(keyword in methodology for keyword in ['proprietary', 'private', 'confidential']):
            return 'Low - Proprietary data/methods'
        else:
            return 'Unknown - Insufficient information'
    
    def _identify_methodological_trends(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Identify methodological trends across papers"""
        methodologies = []
        
        for summary_data in summaries:
            methodology = summary_data['summary'].get('methodology', '').lower()
            methodologies.append(methodology)
        
        # Simple trend identification (could be enhanced with NLP)
        trends = []
        if sum('machine learning' in m for m in methodologies) > len(methodologies) * 0.3:
            trends.append('Machine learning approaches are prevalent')
        if sum('deep learning' in m for m in methodologies) > len(methodologies) * 0.2:
            trends.append('Deep learning methods are increasingly used')
        if sum('statistical' in m for m in methodologies) > len(methodologies) * 0.2:
            trends.append('Statistical methods remain important')
        
        return trends if trends else ['No clear methodological trends identified']
    
    def _identify_implementation_challenges(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Identify common implementation challenges"""
        challenges = []
        limitations = []
        
        for summary_data in summaries:
            limitations.extend(summary_data['summary'].get('limitations', []))
        
        # Analyze limitations for common challenges
        limitation_text = ' '.join(limitations).lower()
        
        if 'data' in limitation_text and ('quality' in limitation_text or 'availability' in limitation_text):
            challenges.append('Data quality and availability issues')
        if 'computational' in limitation_text or 'scalability' in limitation_text:
            challenges.append('Computational scalability concerns')
        if 'generalization' in limitation_text or 'generalizability' in limitation_text:
            challenges.append('Generalization to new domains/datasets')
        if 'evaluation' in limitation_text or 'benchmark' in limitation_text:
            challenges.append('Standardized evaluation challenges')
        
        return challenges if challenges else ['No specific implementation challenges identified']
    
    def _identify_reproducibility_issues(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Identify reproducibility concerns"""
        concerns = []
        
        # Check for common reproducibility issues
        for summary_data in summaries:
            limitations = summary_data['summary'].get('limitations', [])
            
            for limitation in limitations:
                if any(keyword in limitation.lower() for keyword in ['reproduce', 'replication', 'code', 'implementation']):
                    concerns.append(f"Reproducibility concern: {limitation}")
        
        return concerns if concerns else ['No specific reproducibility concerns identified']
    
    def _recommend_methodologies(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Recommend methodologies based on analysis"""
        recommendations = []
        
        # Analyze successful methodologies
        high_relevance_methods = []
        for summary_data in summaries:
            if summary_data['summary'].get('relevance_score', 0) >= 8:
                high_relevance_methods.append(summary_data['summary'].get('methodology', ''))
        
        if high_relevance_methods:
            recommendations.append(f"Consider methodologies from highly relevant papers: {', '.join(set(high_relevance_methods))}")
        
        recommendations.append("Ensure reproducibility by providing code and detailed experimental setup")
        recommendations.append("Use standardized evaluation metrics and benchmarks where possible")
        
        return recommendations
    
    # Additional utility methods
    def export_summaries(self, summaries: List[Dict[str, Any]], format: str = "json") -> str:
        """Export summaries in various formats"""
        if format == "json":
            return json.dumps(summaries, indent=2)
        elif format == "markdown":
            return self._convert_to_markdown(summaries)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _convert_to_markdown(self, summaries: List[Dict[str, Any]]) -> str:
        """Convert summaries to markdown format"""
        markdown = "# Research Paper Summaries\n\n"
        
        for summary in summaries:
            if summary.get('type', '').startswith('individual'):
                markdown += f"## {summary.get('title', 'Unknown Title')}\n\n"
                markdown += f"**Authors:** {summary.get('authors', 'Unknown')}\n"
                markdown += f"**Year:** {summary.get('year', 'Unknown')}\n"
                markdown += f"**Relevance Score:** {summary.get('relevance_score', 'N/A')}/10\n\n"
                
                if 'key_findings' in summary:
                    markdown += "**Key Findings:**\n"
                    for finding in summary['key_findings']:
                        markdown += f"- {finding}\n"
                    markdown += "\n"
                
                if 'methodology' in summary:
                    markdown += f"**Methodology:** {summary['methodology']}\n\n"
                
                if 'limitations' in summary:
                    markdown += "**Limitations:**\n"
                    for limitation in summary['limitations']:
                        markdown += f"- {limitation}\n"
                    markdown += "\n"
                
                markdown += "---\n\n"
            
            elif summary.get('type', '').startswith('synthesis'):
                markdown += "## Research Synthesis\n\n"
                
                if 'main_themes' in summary:
                    markdown += "### Main Themes\n"
                    for theme in summary['main_themes']:
                        markdown += f"- {theme}\n"
                    markdown += "\n"
                
                if 'key_insights' in summary:
                    markdown += "### Key Insights\n"
                    for insight in summary['key_insights']:
                        markdown += f"- {insight}\n"
                    markdown += "\n"
                
                if 'research_gaps' in summary:
                    markdown += "### Research Gaps\n"
                    for gap in summary['research_gaps']:
                        markdown += f"- {gap}\n"
                    markdown += "\n"
        
        return markdown


# Example usage
async def main():
    """Example usage of the SummarizerAgent"""
    
    # Initialize the agent
    agent = SummarizerAgent()
    
    # Example papers data
    papers = [
        {
            'title': 'Attention Is All You Need',
            'authors': 'Vaswani et al.',
            'year': '2017',
            'abstract': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms...',
            'url': 'https://arxiv.org/abs/1706.03762',
            'source': 'arXiv'
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'authors': 'Devlin et al.',
            'year': '2018',
            'abstract': 'We introduce a new language representation model called BERT...',
            'url': 'https://arxiv.org/abs/1810.04805',
            'source': 'arXiv'
        }
    ]
    
    # Summarize papers
    query = "transformer architectures in natural language processing"
    summaries = await agent.summarize_papers(papers, query, style="detailed")
    
    # Export results
    json_output = agent.export_summaries(summaries, format="json")
    markdown_output = agent.export_summaries(summaries, format="markdown")
    
    print("JSON Output:")
    print(json_output)
    print("\nMarkdown Output:")
    print(markdown_output)

if __name__ == "__main__":
    asyncio.run(main()) 