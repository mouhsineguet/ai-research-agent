"""
Critique Agent - Evaluates research quality and provides critical analysis
"""

import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
from .llm_config import LLMConfig, initialize_llm

logger = logging.getLogger(__name__)

class QualityAssessment(BaseModel):
    """Structure for quality assessment results"""
    overall_score: int = Field(description="Overall quality score (1-10)")
    relevance_score: int = Field(description="Relevance to query (1-10)")
    credibility_score: int = Field(description="Credibility assessment (1-10)")
    completeness_score: int = Field(description="Completeness of coverage (1-10)")
    methodology_score: int = Field(description="Methodology quality (1-10)")
    strengths: List[str] = Field(description="Key strengths of the research")
    weaknesses: List[str] = Field(description="Identified weaknesses")
    bias_indicators: List[str] = Field(description="Potential bias indicators")

class GapAnalysis(BaseModel):
    """Structure for gap analysis results"""
    missing_areas: List[str] = Field(description="Areas not adequately covered")
    conflicting_evidence: List[str] = Field(description="Conflicting findings")
    methodological_gaps: List[str] = Field(description="Methodological limitations")
    temporal_gaps: List[str] = Field(description="Temporal coverage gaps")
    geographic_gaps: List[str] = Field(description="Geographic coverage gaps")
    theoretical_gaps: List[str] = Field(description="Theoretical framework gaps")

class CritiqueResult(BaseModel):
    """Structure for complete critique results"""
    quality_assessment: QualityAssessment
    gap_analysis: GapAnalysis
    recommendations: List[str] = Field(description="Specific recommendations for improvement")
    needs_human_input: bool = Field(description="Whether human input is required")
    confidence_level: str = Field(description="Confidence in the critique (High/Medium/Low)")
    critical_issues: List[str] = Field(description="Critical issues that need attention")

class CritiqueAgent:
    """Agent responsible for evaluating research quality and providing critical analysis"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            llm_config = LLMConfig(
                provider="groq",
                model_name="gemma2-9b-it",
                temperature=0.1
            )
        self.llm = initialize_llm(llm_config)
        self.critique_parser = PydanticOutputParser(pydantic_object=CritiqueResult)
        
        # Create prompts
        self.quality_assessment_prompt = self._create_quality_assessment_prompt()
        self.gap_analysis_prompt = self._create_gap_analysis_prompt()
        self.critique_prompt = self._create_critique_prompt()
    
    def _create_quality_assessment_prompt(self) -> ChatPromptTemplate:
        """Create prompt for quality assessment"""
        
        system_prompt = """You are a Research Quality Assessment Expert. Evaluate the quality of research findings and synthesis.

Assessment Criteria:
1. Relevance: How well does the research address the original query?
2. Credibility: Are the sources reliable and methods sound?
3. Completeness: Is the coverage comprehensive and balanced?
4. Methodology: Are the research methods appropriate and well-executed?
5. Bias: Are there any potential biases in the analysis?

Provide objective, evidence-based assessments with specific examples."""

        human_prompt = """Research Query: {query}

Research Papers Found: {paper_count}
Paper Titles: {paper_titles}

Research Summary: {summary}

Please assess the quality of this research synthesis."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_gap_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for gap analysis"""
        
        system_prompt = """You are a Research Gap Analysis Expert. Identify gaps, limitations, and areas for improvement in the research.

Focus on:
1. Missing areas or topics not covered
2. Conflicting evidence or findings
3. Methodological limitations
4. Temporal or geographic gaps
5. Theoretical framework gaps
6. Sample size or scope limitations

Be specific and actionable in your analysis."""

        human_prompt = """Research Query: {query}

Papers Analyzed: {paper_count}
Research Summary: {summary}

Please identify gaps and limitations in this research."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_critique_prompt(self) -> ChatPromptTemplate:
        """Create comprehensive critique prompt"""
        
        system_prompt = """You are a Research Critique Expert. Provide a comprehensive evaluation of research quality and generate actionable recommendations.

Your role is to:
1. Assess overall research quality across multiple dimensions
2. Identify gaps, limitations, and potential biases
3. Generate specific, actionable recommendations
4. Determine if human input is needed for complex decisions
5. Provide confidence levels in your assessment

Be thorough, objective, and constructive in your critique.

{format_instructions}"""

        human_prompt = """Research Query: {query}

Research Context:
- Papers Found: {paper_count}
- Paper Titles: {paper_titles}
- Summary: {summary}
- User Preferences: {user_preferences}

Please provide a comprehensive critique of this research."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    async def evaluate_research(self, query: str, papers: List[Dict[str, Any]], summaries: List[Dict[str, Any]], user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate research quality and provide comprehensive critique"""
        
        logger.info(f"Evaluating research quality for query: {query[:100]}...")
        
        try:
            # Prepare summary text
            summary_text = self._format_summaries_for_critique(summaries)
            paper_titles = [paper.get('title', 'Unknown') for paper in papers]
            
            # Prepare prompt inputs
            prompt_inputs = {
                "query": query,
                "paper_count": len(papers),
                "paper_titles": "\n".join([f"- {title}" for title in paper_titles]),
                "summary": summary_text,
                "user_preferences": json.dumps(user_preferences or {}),
                "format_instructions": self.critique_parser.get_format_instructions()
            }
            
            # Generate critique
            response = await self.llm.ainvoke(
                self.critique_prompt.format_messages(**prompt_inputs)
            )
            
            # Parse response
            try:
                critique_result = self.critique_parser.parse(response.content)
                result = critique_result.dict()
            except Exception as e:
                logger.warning(f"Failed to parse critique, using fallback: {str(e)}")
                result = self._create_fallback_critique(query, papers, summaries, user_preferences)
            
            # Add metadata
            result['evaluation_timestamp'] = self._get_timestamp()
            result['papers_evaluated'] = len(papers)
            result['query'] = query
            
            logger.info("✅ Research evaluation completed")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating research: {str(e)}")
            return self._create_fallback_critique(query, papers, summaries, user_preferences)
    
    def _format_summaries_for_critique(self, summaries: List[Dict[str, Any]]) -> str:
        """Format summaries for critique analysis"""
        
        if not summaries:
            return "No summaries available"
        
        formatted_text = ""
        for i, summary_data in enumerate(summaries):
            if isinstance(summary_data, dict) and 'summary' in summary_data:
                summary = summary_data['summary']
                formatted_text += f"\nPaper {i+1}:\n"
                formatted_text += f"Title: {summary.get('paper_title', 'Unknown')}\n"
                formatted_text += f"Key Findings: {', '.join(summary.get('key_findings', []))}\n"
                formatted_text += f"Methodology: {summary.get('methodology', 'Unknown')}\n"
                formatted_text += f"Limitations: {', '.join(summary.get('limitations', []))}\n"
                formatted_text += f"Relevance: {summary.get('relevance_score', 'Unknown')}/10\n"
            else:
                # Handle case where summary_data is already a summary
                formatted_text += f"\nPaper {i+1}:\n{str(summary_data)}\n"
        
        return formatted_text
    
    def _create_fallback_critique(self, query: str, papers: List[Dict[str, Any]], summaries: List[Dict[str, Any]], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback critique when AI analysis fails"""
        
        return {
            "quality_assessment": {
                "overall_score": 5,
                "relevance_score": 5,
                "credibility_score": 5,
                "completeness_score": 5,
                "methodology_score": 5,
                "strengths": ["Research covers the basic query requirements"],
                "weaknesses": ["Limited analysis due to processing error"],
                "bias_indicators": ["Unable to assess bias due to processing error"]
            },
            "gap_analysis": {
                "missing_areas": ["Unable to identify gaps due to processing error"],
                "conflicting_evidence": [],
                "methodological_gaps": [],
                "temporal_gaps": [],
                "geographic_gaps": [],
                "theoretical_gaps": []
            },
            "recommendations": [
                "Review the research manually for quality assessment",
                "Consider expanding the search scope",
                "Verify source credibility independently"
            ],
            "needs_human_input": True,
            "confidence_level": "Low",
            "critical_issues": ["Processing error occurred during critique generation"],
            "evaluation_timestamp": self._get_timestamp(),
            "papers_evaluated": len(papers),
            "query": query
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def assess_bias(self, summaries: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Specifically assess potential biases in the research"""
        
        logger.info("Assessing potential biases in research")
        
        bias_indicators = []
        
        # Check for publication bias
        publication_years = []
        for summary_data in summaries:
            if isinstance(summary_data, dict) and 'original_paper' in summary_data:
                paper = summary_data['original_paper']
                year = paper.get('year', 'Unknown')
                if year != 'Unknown':
                    try:
                        publication_years.append(int(year))
                    except ValueError:
                        pass
        
        if publication_years:
            year_range = max(publication_years) - min(publication_years)
            if year_range < 5:
                bias_indicators.append("Limited temporal coverage - may miss recent developments")
            if max(publication_years) < 2020:
                bias_indicators.append("Research may be outdated - consider more recent sources")
        
        # Check for source diversity
        sources = set()
        for summary_data in summaries:
            if isinstance(summary_data, dict) and 'original_paper' in summary_data:
                paper = summary_data['original_paper']
                source = paper.get('source', 'unknown')
                sources.add(source)
        
        if len(sources) < 3:
            bias_indicators.append("Limited source diversity - may introduce source bias")
        
        # Check for methodological bias
        methodologies = []
        for summary_data in summaries:
            if isinstance(summary_data, dict) and 'summary' in summary_data:
                summary = summary_data['summary']
                methodology = summary.get('methodology', '')
                if methodology:
                    methodologies.append(methodology.lower())
        
        if methodologies:
            empirical_count = sum(1 for m in methodologies if 'empirical' in m or 'experiment' in m)
            theoretical_count = sum(1 for m in methodologies if 'theoretical' in m or 'review' in m)
            
            if empirical_count == 0:
                bias_indicators.append("No empirical studies found - may lack practical validation")
            if theoretical_count == 0:
                bias_indicators.append("No theoretical frameworks found - may lack conceptual grounding")
        
        return {
            "bias_indicators": bias_indicators,
            "source_diversity": len(sources),
            "temporal_coverage": year_range if publication_years else "Unknown",
            "methodological_diversity": len(set(methodologies))
        }
    
    async def generate_recommendations(self, critique_result: Dict[str, Any], query: str) -> List[str]:
        """Generate specific recommendations based on critique"""
        
        recommendations = []
        
        # Quality-based recommendations
        quality = critique_result.get('quality_assessment', {})
        overall_score = quality.get('overall_score', 5)
        
        if overall_score < 6:
            recommendations.append("Consider expanding search to include more recent or relevant sources")
            recommendations.append("Review source credibility and methodology of included papers")
        
        if quality.get('relevance_score', 5) < 6:
            recommendations.append("Refine search terms to better match the research query")
            recommendations.append("Consider alternative search strategies or databases")
        
        # Gap-based recommendations
        gaps = critique_result.get('gap_analysis', {})
        missing_areas = gaps.get('missing_areas', [])
        
        if missing_areas:
            recommendations.append(f"Address missing areas: {', '.join(missing_areas[:3])}")
        
        if gaps.get('conflicting_evidence'):
            recommendations.append("Investigate conflicting findings with additional sources")
        
        # Methodology recommendations
        if quality.get('methodology_score', 5) < 6:
            recommendations.append("Include papers with more rigorous methodology")
            recommendations.append("Consider systematic review approaches")
        
        # General recommendations
        recommendations.append("Validate key findings with independent sources")
        recommendations.append("Consider practical implications and applications")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def should_require_human_input(self, critique_result: Dict[str, Any]) -> bool:
        """Determine if human input is required based on critique results"""
        
        # Check for critical issues
        critical_issues = critique_result.get('critical_issues', [])
        if critical_issues:
            return True
        
        # Check confidence level
        confidence = critique_result.get('confidence_level', 'Medium')
        if confidence == 'Low':
            return True
        
        # Check quality scores
        quality = critique_result.get('quality_assessment', {})
        overall_score = quality.get('overall_score', 5)
        if overall_score < 4:
            return True
        
        # Check for significant gaps
        gaps = critique_result.get('gap_analysis', {})
        if len(gaps.get('missing_areas', [])) > 3:
            return True
        
        return False
    
    def export_critique_report(self, critique_result: Dict[str, Any], format: str = "markdown") -> str:
        """Export critique results in various formats"""
        
        if format == "markdown":
            return self._convert_to_markdown(critique_result)
        elif format == "json":
            return json.dumps(critique_result, indent=2)
        else:
            return str(critique_result)
    
    def _convert_to_markdown(self, critique_result: Dict[str, Any]) -> str:
        """Convert critique results to markdown format"""
        
        markdown = "# Research Critique Report\n\n"
        
        # Quality Assessment
        quality = critique_result.get('quality_assessment', {})
        markdown += "## Quality Assessment\n\n"
        markdown += f"- **Overall Score**: {quality.get('overall_score', 'N/A')}/10\n"
        markdown += f"- **Relevance Score**: {quality.get('relevance_score', 'N/A')}/10\n"
        markdown += f"- **Credibility Score**: {quality.get('credibility_score', 'N/A')}/10\n"
        markdown += f"- **Completeness Score**: {quality.get('completeness_score', 'N/A')}/10\n"
        markdown += f"- **Methodology Score**: {quality.get('methodology_score', 'N/A')}/10\n\n"
        
        # Strengths and Weaknesses
        markdown += "### Strengths\n"
        for strength in quality.get('strengths', []):
            markdown += f"- {strength}\n"
        markdown += "\n"
        
        markdown += "### Weaknesses\n"
        for weakness in quality.get('weaknesses', []):
            markdown += f"- {weakness}\n"
        markdown += "\n"
        
        # Gap Analysis
        gaps = critique_result.get('gap_analysis', {})
        markdown += "## Gap Analysis\n\n"
        
        markdown += "### Missing Areas\n"
        for area in gaps.get('missing_areas', []):
            markdown += f"- {area}\n"
        markdown += "\n"
        
        markdown += "### Conflicting Evidence\n"
        for conflict in gaps.get('conflicting_evidence', []):
            markdown += f"- {conflict}\n"
        markdown += "\n"
        
        # Recommendations
        markdown += "## Recommendations\n\n"
        for rec in critique_result.get('recommendations', []):
            markdown += f"- {rec}\n"
        markdown += "\n"
        
        # Critical Issues
        critical_issues = critique_result.get('critical_issues', [])
        if critical_issues:
            markdown += "## Critical Issues\n\n"
            for issue in critical_issues:
                markdown += f"- ⚠️ {issue}\n"
            markdown += "\n"
        
        # Metadata
        markdown += "## Metadata\n\n"
        markdown += f"- **Confidence Level**: {critique_result.get('confidence_level', 'Unknown')}\n"
        markdown += f"- **Human Input Required**: {'Yes' if critique_result.get('needs_human_input') else 'No'}\n"
        markdown += f"- **Papers Evaluated**: {critique_result.get('papers_evaluated', 'Unknown')}\n"
        markdown += f"- **Evaluation Timestamp**: {critique_result.get('evaluation_timestamp', 'Unknown')}\n"
        
        return markdown

# Test function
async def test_critique_agent():
    """Test the critique agent functionality"""
    
    # Create test data
    test_query = "What are the latest developments in transformer architecture for natural language processing?"
    test_papers = [
        {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "year": 2017,
            "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Devlin et al."],
            "year": 2018,
            "abstract": "We introduce a new language representation model called BERT."
        }
    ]
    
    test_summaries = [
        {
            "summary": {
                "paper_title": "Attention Is All You Need",
                "key_findings": ["Transformer architecture", "Self-attention mechanism"],
                "methodology": "Neural machine translation",
                "limitations": ["Limited to sequence tasks"],
                "relevance_score": 9
            }
        },
        {
            "summary": {
                "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "key_findings": ["Bidirectional training", "Masked language modeling"],
                "methodology": "Pre-training and fine-tuning",
                "limitations": ["Computational cost"],
                "relevance_score": 8
            }
        }
    ]
    
    # Test the agent
    agent = CritiqueAgent()
    result = await agent.evaluate_research(test_query, test_papers, test_summaries)
    
    print("Critique Agent Test Results:")
    print(json.dumps(result, indent=2))
    
    # Test bias assessment
    bias_result = await agent.assess_bias(test_summaries, test_query)
    print("\nBias Assessment:")
    print(json.dumps(bias_result, indent=2))
    
    # Test recommendations
    recommendations = await agent.generate_recommendations(result, test_query)
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"- {rec}")
    
    # Test markdown export
    markdown_report = agent.export_critique_report(result, "markdown")
    print("\nMarkdown Report:")
    print(markdown_report)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_critique_agent()) 