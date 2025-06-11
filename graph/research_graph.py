"""
Research Graph - Main LangGraph implementation
Coordinates all agents in the research assistant workflow
"""

from typing import Dict, List, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Import agents
from .planner_agent import PlannerAgent
from .tool_agent import ToolAgent
from .summarizer_agent import SummarizerAgent
from .critique_agent import CritiqueAgent
from .supervisor_agent import SupervisorAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchState:
    """State definition for the research graph"""
    query: str
    user_preferences: Dict[str, Any]
    memory: str
    papers: List[Dict[str, Any]]
    summaries: List[str]
    critique: str
    recommendations: List[str]
    human_feedback: str
    final_output: str
    requires_human_input: bool
    iteration_count: int
    current_agent: str
    plan: Dict[str, Any]
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_research_graph():
    """Create and configure the research graph"""
    
    # Initialize agents
    planner = PlannerAgent()
    tool_agent = ToolAgent()
    summarizer = SummarizerAgent()
    critique_agent = CritiqueAgent()
    supervisor = SupervisorAgent()
    
    # Define node functions
    async def planner_node(state: ResearchState) -> Dict[str, Any]:
        """Plan the research workflow"""
        logger.info("🧠 Planner Agent: Creating research plan")
        
        plan = await planner.create_plan(
            query=state.query,
            user_preferences=state.user_preferences,
            memory_context=state.memory
        )
        
        return {
            "plan": plan,
            "current_agent": "planner",
            "iteration_count": state.iteration_count + 1
        }
    
    async def tool_node(state: ResearchState) -> Dict[str, Any]:
        """Search and retrieve papers"""
        logger.info("🔍 Tool Agent: Searching for papers")
        
        papers = await tool_agent.search_papers(
            query=state.query,
            sources=state.user_preferences.get('preferred_sources', ['arxiv']),
            max_papers=state.user_preferences.get('max_papers', 20)
        )
        
        return {
            "papers": papers,
            "current_agent": "tool"
        }
    
    async def summarizer_node(state: ResearchState) -> Dict[str, Any]:
        """Summarize retrieved papers"""
        logger.info("📝 Summarizer Agent: Creating summaries")
        
        summaries = await summarizer.summarize_papers(
            papers=state.papers,
            query=state.query,
            style=state.user_preferences.get('summary_style', 'detailed')
        )
        
        return {
            "summaries": summaries,
            "current_agent": "summarizer"
        }
    
    async def critique_node(state: ResearchState) -> Dict[str, Any]:
        """Critique and evaluate the research"""
        logger.info("🔍 Critique Agent: Evaluating research quality")
        
        critique_result = await critique_agent.evaluate_research(
            query=state.query,
            papers=state.papers,
            summaries=state.summaries
        )
        
        return {
            "critique": critique_result["critique"],
            "recommendations": critique_result["recommendations"],
            "requires_human_input": critique_result["needs_human_input"],
            "current_agent": "critique"
        }
    
    async def supervisor_node(state: ResearchState) -> Dict[str, Any]:
        """Supervise and coordinate the final output"""
        logger.info("👨‍💼 Supervisor Agent: Finalizing research")
        
        final_result = await supervisor.finalize_research(
            query=state.query,
            papers=state.papers,
            summaries=state.summaries,
            critique=state.critique,
            recommendations=state.recommendations,
            human_feedback=state.human_feedback
        )
        
        return {
            "final_output": final_result,
            "current_agent": "supervisor"
        }
    
    async def human_input_node(state: ResearchState) -> Dict[str, Any]:
        """Handle human input requirement"""
        logger.info("👤 Human Input Required")
        
        # In a real implementation, this would pause for human input
        # For now, we'll simulate it
        return {
            "human_feedback": "Human feedback processed",
            "requires_human_input": False,
            "current_agent": "human"
        }
    
    # Define routing logic
    def should_continue(state: ResearchState) -> Literal["tool", "human_input", "end"]:
        """Determine next step in the workflow"""
        
        current = state.current_agent
        
        if current == "planner":
            return "tool"
        elif current == "tool":
            return "summarizer"
        elif current == "summarizer":
            if state.user_preferences.get('enable_critique', True):
                return "critique"
            else:
                return "supervisor"
        elif current == "critique":
            if state.requires_human_input and state.user_preferences.get('human_feedback', True):
                return "human_input"
            else:
                return "supervisor"
        elif current == "human":
            return "supervisor"
        else:
            return "end"
    
    def needs_human_input(state: ResearchState) -> Literal["human_input", "supervisor"]:
        """Check if human input is needed"""
        if state.requires_human_input:
            return "human_input"
        return "supervisor"
    
    # Create the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("human_input", human_input_node)
    
    # Add edges
    workflow.set_entry_point("planner")
    
    # Linear flow with conditional branches
    workflow.add_edge("planner", "tool")
    workflow.add_edge("tool", "summarizer")
    
    # Conditional edge from summarizer
    workflow.add_conditional_edges(
        "summarizer",
        lambda state: "critique" if state.user_preferences.get('enable_critique', True) else "supervisor",
        {
            "critique": "critique",
            "supervisor": "supervisor"
        }
    )
    
    # Conditional edge from critique
    workflow.add_conditional_edges(
        "critique",
        needs_human_input,
        {
            "human_input": "human_input", 
            "supervisor": "supervisor"
        }
    )
    
    workflow.add_edge("human_input", "supervisor")
    workflow.add_edge("supervisor", END)
    
    # Add memory
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app

class GraphExecutor:
    """Helper class to execute the research graph"""
    
    def __init__(self):
        self.graph = create_research_graph()
    
    async def execute_research(self, query: str, preferences: Dict[str, Any], memory_context: str = "") -> Dict[str, Any]:
        """Execute the complete research workflow"""
        
        # Initialize state
        initial_state = ResearchState(
            query=query,
            user_preferences=preferences,
            memory=memory_context,
            papers=[],
            summaries=[],
            critique="",
            recommendations=[],
            human_feedback="",
            final_output="",
            requires_human_input=False,
            iteration_count=0,
            current_agent="",
            plan={}
        )
        
        try:
            # Execute the graph
            result = await self.graph.ainvoke(initial_state)
            
            logger.info("✅ Research workflow completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in research workflow: {str(e)}")
            raise e
    
    def get_graph_visualization(self) -> str:
        """Get a visual representation of the graph"""
        try:
            return self.graph.get_graph().draw_mermaid()
        except:
            return "Graph visualization not available"

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_graph():
        """Test the research graph"""
        executor = GraphExecutor()
        
        test_query = "What are the latest developments in transformer architectures?"
        test_preferences = {
            'preferred_sources': ['arxiv'],
            'max_papers': 5,
            'summary_style': 'detailed',
            'enable_critique': True,
            'human_feedback': False
        }
        
        result = await executor.execute_research(test_query, test_preferences)
        print("Research Result:", result)
    
    # Run test
    asyncio.run(test_graph())