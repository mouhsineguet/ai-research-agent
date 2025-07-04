"""
Planner Agent - Creates research plans and coordinates workflow
"""

import json
import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from .llm_config import LLMConfig, initialize_llm

logger = logging.getLogger(__name__)

class PlannerAgent:
    """Agent responsible for creating research plans and coordinating workflow"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            llm_config = LLMConfig(
            provider="groq",
            model_name="gemma2-9b-it",
            temperature=0.1
        )
        self.llm = initialize_llm(llm_config)
        self.planning_prompt = self._create_planning_prompt()
    
    def _create_planning_prompt(self) -> ChatPromptTemplate:
        """Create the planning prompt template"""
        
        system_prompt = """You are a Research Planning Agent. Your role is to analyze user research queries and create comprehensive research plans.

Your responsibilities:
1. Analyze the research question to understand scope and requirements
2. Identify key concepts, terms, and research areas
3. Determine appropriate search strategies
4. Plan the research workflow
5. Consider user preferences and constraints

You should output a structured research plan in JSON format with the following components:
- research_scope: Brief description of what needs to be researched
- key_concepts: List of important terms and concepts
- search_strategy: How to approach the literature search
- expected_paper_types: Types of papers to look for (reviews, empirical, theoretical, etc.)
- quality_criteria: How to evaluate paper relevance and quality
- synthesis_approach: How to synthesize findings
- estimated_complexity: Simple/Medium/Complex based on query scope

Consider the user's research background, preferences, and any previous context."""

        human_prompt = """Research Query: {query}

User Preferences:
- Research Areas: {research_areas}
- Preferred Sources: {preferred_sources}
- Summary Style: {summary_style}
- Max Papers: {max_papers}

Previous Context: {memory_context}

Please create a comprehensive research plan for this query."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    async def create_plan(self, query: str, user_preferences: Dict[str, Any], memory_context: str = "") -> Dict[str, Any]:
        """Create a research plan based on the query and preferences"""
        
        try:
            logger.info(f"Creating research plan for query: {query[:100]}...")
            
            # Prepare prompt inputs
            prompt_inputs = {
                "query": query,
                "research_areas": user_preferences.get('research_areas', []),
                "preferred_sources": user_preferences.get('preferred_sources', []),
                "summary_style": user_preferences.get('summary_style', 'detailed'),
                "max_papers": user_preferences.get('max_papers', 20),
                "memory_context": memory_context or "No previous context"
            }
            
            # Generate plan
            response = await self.llm.ainvoke(
                self.planning_prompt.format_messages(**prompt_inputs)
            )
            
            # Parse the response
            plan_text = response.content
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in the response
                if "```json" in plan_text:
                    json_start = plan_text.find("```json") + 7
                    json_end = plan_text.find("```", json_start)
                    json_text = plan_text[json_start:json_end].strip()
                else:
                    # Try to parse the entire response as JSON
                    json_text = plan_text
                
                plan = json.loads(json_text)
                
            except json.JSONDecodeError:
                # Fallback: create a basic plan structure
                logger.warning("Could not parse plan as JSON, creating basic plan")
                plan = self._create_fallback_plan(query, user_preferences)
            
            # Validate and enhance plan
            plan = self._validate_and_enhance_plan(plan, query, user_preferences)
            
            logger.info("✅ Research plan created successfully")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating research plan: {str(e)}")
            return self._create_fallback_plan(query, user_preferences)
    
    def _create_fallback_plan(self, query: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback plan if AI planning fails"""
        
        return {
            "research_scope": f"Literature review on: {query}",
            "key_concepts": self._extract_key_concepts(query),
            "search_strategy": {
                "primary_terms": query.split()[:5],  # First 5 words as primary terms
                "sources": preferences.get('preferred_sources', ['arxiv']),
                "filters": {
                    "max_papers": preferences.get('max_papers', 20),
                    "language": "English"
                }
            },
            "expected_paper_types": ["research", "review", "survey"],
            "quality_criteria": [
                "Relevance to query",
                "Citation count",
                "Publication venue",
                "Recency"
            ],
            "synthesis_approach": "Thematic analysis with key findings extraction",
            "estimated_complexity": "Medium",
            "workflow_steps": [
                "Search literature",
                "Filter and rank papers",
                "Extract key information",
                "Synthesize findings",
                "Generate recommendations"
            ]
        }
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query using simple heuristics"""
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'are', 'is', 'how', 'why', 'when', 'where'}
        
        words = query.lower().split()
        key_concepts = [word for word in words if word not in stop_words and len(word) > 3]
        
        return key_concepts[:10]  # Return top 10 concepts
    
    def _validate_and_enhance_plan(self, plan: Dict[str, Any], query: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance the generated plan"""
        
        # Ensure required fields exist
        required_fields = [
            'research_scope', 'key_concepts', 'search_strategy', 
            'expected_paper_types', 'quality_criteria', 'synthesis_approach'
        ]
        
        for field in required_fields:
            if field not in plan:
                if field == 'key_concepts':
                    plan[field] = self._extract_key_concepts(query)
                elif field == 'search_strategy':
                    plan[field] = {
                        "primary_terms": query.split()[:5],
                        "sources": preferences.get('preferred_sources', ['arxiv'])
                    }
                else:
                    plan[field] = f"To be determined for {field}"
        
        # Add metadata
        plan['created_for_query'] = query
        plan['user_preferences'] = preferences
        plan['plan_version'] = "1.0"
        
        # Add workflow steps if not present
        if 'workflow_steps' not in plan:
            plan['workflow_steps'] = [
                "Literature search",
                "Paper filtering and ranking", 
                "Content extraction and analysis",
                "Synthesis and summary generation",
                "Quality assessment and critique",
                "Recommendation generation"
            ]
        
        return plan
    
    async def refine_plan(self, original_plan: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        """Refine the research plan based on feedback"""
        
        refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Research Planning Agent. Refine the given research plan based on the provided feedback."),
            ("human", f"Original Plan: {json.dumps(original_plan, indent=2)}\n\nFeedback: {feedback}\n\nPlease provide a refined research plan.")
        ])
        
        try:
            response = await self.llm.ainvoke(refinement_prompt.format_messages())
            # Parse and return refined plan
            refined_plan_text = response.content
            
            # Try to extract JSON
            if "```json" in refined_plan_text:
                json_start = refined_plan_text.find("```json") + 7
                json_end = refined_plan_text.find("```", json_start)
                json_text = refined_plan_text[json_start:json_end].strip()
                refined_plan = json.loads(json_text)
            else:
                refined_plan = original_plan  # Fallback to original
            
            logger.info("✅ Research plan refined successfully")
            return refined_plan
            
        except Exception as e:
            logger.error(f"Error refining plan: {str(e)}")
            return original_plan
    
    def get_search_terms(self, plan: Dict[str, Any]) -> List[str]:
        """Extract search terms from the plan"""
        
        search_terms = []
        
        # From key concepts
        if 'key_concepts' in plan:
            search_terms.extend(plan['key_concepts'])
        
        # From search strategy
        if 'search_strategy' in plan and 'primary_terms' in plan['search_strategy']:
            search_terms.extend(plan['search_strategy']['primary_terms'])
        
        # Remove duplicates and return
        return list(set(search_terms))
    
    def estimate_effort(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the effort required for the research plan"""
        
        complexity = plan.get('estimated_complexity', 'Medium')
        max_papers = plan.get('user_preferences', {}).get('max_papers', 20)
        
        effort_mapping = {
            'Simple': {'time_estimate': '15-30 minutes', 'confidence': 'High'},
            'Medium': {'time_estimate': '30-60 minutes', 'confidence': 'Medium'},
            'Complex': {'time_estimate': '60+ minutes', 'confidence': 'Medium'}
        }
        
        base_effort = effort_mapping.get(complexity, effort_mapping['Medium'])
        
        # Adjust based on number of papers
        if max_papers > 30:
            base_effort['time_estimate'] = base_effort['time_estimate'].replace('30', '45').replace('60', '90')
        
        return {
            'complexity': complexity,
            'estimated_time': base_effort['time_estimate'],
            'confidence_level': base_effort['confidence'],
            'factors': {
                'paper_count': max_papers,
                'research_scope': plan.get('research_scope', 'Unknown'),
                'synthesis_complexity': plan.get('synthesis_approach', 'Standard')
            }
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_planner():
        """Test the planner agent"""
        planner = PlannerAgent()
        
        query = "What are the latest developments in transformer architectures for natural language processing?"
        preferences = {
            'research_areas': ['NLP', 'Deep Learning'],
            'preferred_sources': ['arxiv', 'pubmed'],
            'summary_style': 'detailed',
            'max_papers': 15
        }
        
        plan = await planner.create_plan(query, preferences)
        print("Generated Plan:")
        print(json.dumps(plan, indent=2))
        
        # Test effort estimation
        effort = planner.estimate_effort(plan)
        print("\nEffort Estimation:")
        print(json.dumps(effort, indent=2))
    
    # Run test
    asyncio.run(test_planner())