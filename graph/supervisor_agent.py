"""
Supervisor Agent - Coordinates final output and human interaction
Responsible for synthesizing research results, managing human feedback, and ensuring quality output
"""

import json
import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from llm_config import LLMConfig, initialize_llm

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """Agent responsible for finalizing research output and managing human interaction"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            llm_config = LLMConfig(
            provider="groq",
            model_name="llama-3.2-70b-versatile",
            temperature=0.1
        )
        self.llm = initialize_llm(llm_config)
        self.langsmith_client = Client()
        self.supervision_prompt = self._create_supervision_prompt()
        self.feedback_prompt = self._create_feedback_prompt()
    
    def _create_supervision_prompt(self) -> ChatPromptTemplate:
        """Create the supervision prompt template"""
        
        system_prompt = """You are a Research Supervisor Agent. Your role is to synthesize research findings and coordinate the final output.

Your responsibilities:
1. Synthesize research findings from multiple papers
2. Evaluate the quality and completeness of the research
3. Generate actionable recommendations
4. Identify gaps in the research
5. Coordinate with human feedback when needed
6. Ensure the final output is clear, comprehensive, and useful

You should structure your output to include:
- Executive Summary: Brief overview of findings
- Key Findings: Main research insights
- Research Gaps: Areas needing further investigation
- Recommendations: Actionable next steps
- Quality Assessment: Evaluation of research quality
- Human Feedback Integration: How user input influenced the output"""

        human_prompt = """Research Query: {query}

Research Papers: {papers}

Paper Summaries: {summaries}

Critique Results: {critique}

Recommendations: {recommendations}

Human Feedback: {human_feedback}

Please synthesize this information into a comprehensive research report."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_feedback_prompt(self) -> ChatPromptTemplate:
        """Create the feedback handling prompt template"""
        
        system_prompt = """You are a Research Feedback Handler. Your role is to process and integrate human feedback into the research process.

Your responsibilities:
1. Analyze human feedback for key points
2. Identify areas needing revision
3. Suggest specific improvements
4. Maintain research quality standards
5. Ensure feedback is properly integrated

Consider:
- Research scope and objectives
- Quality of evidence
- Clarity of presentation
- Completeness of analysis
- Actionability of recommendations"""

        human_prompt = """Current Research Output: {current_output}

Human Feedback: {feedback}

Please analyze this feedback and suggest how to improve the research output."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    async def finalize_research(
        self,
        query: str,
        papers: List[Dict[str, Any]],
        summaries: List[str],
        critique: str,
        recommendations: List[str],
        human_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synthesize research findings and generate final output"""
        
        try:
            logger.info("Supervisor: Finalizing research output")
            
            # Prepare prompt inputs
            prompt_inputs = {
                "query": query,
                "papers": json.dumps(papers, indent=2),
                "summaries": "\n\n".join(summaries),
                "critique": critique,
                "recommendations": json.dumps(recommendations, indent=2),
                "human_feedback": human_feedback or "No human feedback provided"
            }
            
            # Generate final output
            response = await self.llm.ainvoke(
                self.supervision_prompt.format_messages(**prompt_inputs)
            )
            
            # Process and structure the output
            final_output = self._process_final_output(response.content)
            
            # Log to LangSmith
            await self._log_to_langsmith(
                query=query,
                papers_count=len(papers),
                has_human_feedback=bool(human_feedback),
                output_structure=final_output.keys()
            )
            
            logger.info("✅ Research output finalized successfully")
            return final_output
            
        except Exception as e:
            logger.error(f"Error finalizing research: {str(e)}")
            return self._create_fallback_output(query, papers, summaries)
    
    async def process_human_feedback(
        self,
        current_output: Dict[str, Any],
        feedback: str
    ) -> Dict[str, Any]:
        """Process and integrate human feedback"""
        
        try:
            logger.info("Supervisor: Processing human feedback")
            
            # Prepare prompt inputs
            prompt_inputs = {
                "current_output": json.dumps(current_output, indent=2),
                "feedback": feedback
            }
            
            # Generate feedback analysis
            response = await self.llm.ainvoke(
                self.feedback_prompt.format_messages(**prompt_inputs)
            )
            
            # Process feedback and update output
            updated_output = self._integrate_feedback(current_output, response.content)
            
            logger.info("✅ Human feedback processed successfully")
            return updated_output
            
        except Exception as e:
            logger.error(f"Error processing human feedback: {str(e)}")
            return current_output
    
    def _process_final_output(self, content: str) -> Dict[str, Any]:
        """Process and structure the final output"""
        
        try:
            # Try to parse as JSON if it's in JSON format
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_text = content[json_start:json_end].strip()
                return json.loads(json_text)
            
            # Otherwise, structure the content
            sections = content.split("\n\n")
            structured_output = {
                "executive_summary": "",
                "key_findings": [],
                "research_gaps": [],
                "recommendations": [],
                "quality_assessment": {},
                "feedback_integration": ""
            }
            
            current_section = None
            for section in sections:
                if "Executive Summary" in section:
                    current_section = "executive_summary"
                    structured_output[current_section] = section.split(":", 1)[1].strip()
                elif "Key Findings" in section:
                    current_section = "key_findings"
                elif "Research Gaps" in section:
                    current_section = "research_gaps"
                elif "Recommendations" in section:
                    current_section = "recommendations"
                elif "Quality Assessment" in section:
                    current_section = "quality_assessment"
                elif "Human Feedback" in section:
                    current_section = "feedback_integration"
                    structured_output[current_section] = section.split(":", 1)[1].strip()
                elif current_section and current_section in ["key_findings", "research_gaps", "recommendations"]:
                    structured_output[current_section].append(section.strip())
                elif current_section == "quality_assessment":
                    try:
                        key, value = section.split(":", 1)
                        structured_output[current_section][key.strip()] = value.strip()
                    except ValueError:
                        continue
            
            return structured_output
            
        except Exception as e:
            logger.error(f"Error processing final output: {str(e)}")
            return self._create_fallback_output("", [], [])
    
    def _integrate_feedback(self, current_output: Dict[str, Any], feedback_analysis: str) -> Dict[str, Any]:
        """Integrate human feedback into the current output"""
        
        try:
            # Extract feedback points
            feedback_points = []
            for line in feedback_analysis.split("\n"):
                if line.strip().startswith("- "):
                    feedback_points.append(line.strip()[2:])
            
            # Update output based on feedback
            updated_output = current_output.copy()
            
            for point in feedback_points:
                if "summary" in point.lower():
                    updated_output["executive_summary"] = self._update_section(
                        updated_output["executive_summary"],
                        point
                    )
                elif "finding" in point.lower():
                    updated_output["key_findings"].append(point)
                elif "gap" in point.lower():
                    updated_output["research_gaps"].append(point)
                elif "recommend" in point.lower():
                    updated_output["recommendations"].append(point)
            
            # Add feedback integration note
            updated_output["feedback_integration"] = (
                f"Updated based on human feedback: {', '.join(feedback_points[:3])}..."
            )
            
            return updated_output
            
        except Exception as e:
            logger.error(f"Error integrating feedback: {str(e)}")
            return current_output
    
    def _update_section(self, current: str, update: str) -> str:
        """Update a section with new content"""
        return f"{current}\n\nUpdated: {update}"
    
    def _create_fallback_output(self, query: str, papers: List[Dict[str, Any]], summaries: List[str]) -> Dict[str, Any]:
        """Create a basic fallback output if processing fails"""
        
        return {
            "executive_summary": f"Research findings for: {query}",
            "key_findings": [s[:200] + "..." for s in summaries[:3]],
            "research_gaps": ["Unable to identify gaps due to processing error"],
            "recommendations": ["Review the research findings manually"],
            "quality_assessment": {
                "completeness": "Partial",
                "reliability": "Needs review",
                "actionability": "Limited"
            },
            "feedback_integration": "No feedback processed due to error"
        }
    
    async def _log_to_langsmith(
        self,
        query: str,
        papers_count: int,
        has_human_feedback: bool,
        output_structure: List[str]
    ) -> None:
        """Log research completion to LangSmith"""
        
        try:
            self.langsmith_client.create_run(
                name="research_completion",
                inputs={
                    "query": query,
                    "papers_processed": papers_count,
                    "human_feedback_included": has_human_feedback
                },
                outputs={
                    "output_sections": list(output_structure),
                    "status": "completed"
                },
                run_type="chain"
            )
        except Exception as e:
            logger.warning(f"Failed to log to LangSmith: {str(e)}")
    
    async def test_supervisor():
        """Test the supervisor agent"""
        supervisor = SupervisorAgent()
        
        # Test data
        test_query = "Recent advances in transformer architectures"
        test_papers = [
            {
                "title": "Test Paper 1",
                "abstract": "Test abstract 1",
                "authors": ["Author 1"],
                "year": 2023
            }
        ]
        test_summaries = ["Test summary 1"]
        test_critique = "Test critique"
        test_recommendations = ["Test recommendation 1"]
        
        # Test finalization
        output = await supervisor.finalize_research(
            query=test_query,
            papers=test_papers,
            summaries=test_summaries,
            critique=test_critique,
            recommendations=test_recommendations
        )
        
        print("Test Output:", json.dumps(output, indent=2))
        
        # Test feedback processing
        feedback = "The summary could be more detailed and include more recent papers"
        updated = await supervisor.process_human_feedback(output, feedback)
        
        print("\nUpdated Output:", json.dumps(updated, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_supervisor()) 