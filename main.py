#!/usr/bin/env python3
"""
AI Research Assistant for Technical Literature Review
Main application entry point with Streamlit interface
"""

import streamlit as st
import asyncio
import os
from typing import Dict, Any
from datetime import datetime
import uuid

# Import our graph and components
from graph.research_graph import create_research_graph
from memory.memory_manager import MemoryManager
from evaluation.langsmith_config import setup_langsmith

# Set up page config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'memory_manager' not in st.session_state:
        st.session_state.memory_manager = MemoryManager()
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'research_areas': [],
            'preferred_sources': ['arxiv', 'pubmed', 'semantic_scholar'],
            'summary_style': 'detailed'
        }

async def process_research_query(query: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Process research query through the agent graph"""
    
    # Initialize graph if not exists
    if st.session_state.graph is None:
        st.session_state.graph = create_research_graph()
    
    # Prepare input state
    input_state = {
        "query": query,
        "user_preferences": preferences,
        "memory": st.session_state.memory_manager.get_context(),
        "papers": [],
        "summaries": [],
        "critique": "",
        "recommendations": [],
        "human_feedback": "",
        "final_output": "",
        "requires_human_input": False,
        "iteration_count": 0,
        #"thread_id": str(uuid.uuid4())
    }
    
    # Run the graph
    result = await st.session_state.graph.ainvoke(input_state)
    
    # Update memory
    st.session_state.memory_manager.add_interaction(query, result.get("final_output", ""))
    
    return result

def main():
    """Main application interface"""
    
    # Initialize session state
    initialize_session_state()
    
    # Setup LangSmith
    setup_langsmith()
    
    # Main header
    st.title("🔬 AI Research Assistant")
    st.subheader("Advanced Technical Literature Review with Multi-Agent System")
    
    # Sidebar for preferences
    with st.sidebar:
        st.header("⚙️ Preferences")
        
        # Research areas
        research_areas = st.text_area(
            "Research Areas of Interest",
            value="\n".join(st.session_state.user_preferences['research_areas']),
            help="Enter your research areas, one per line"
        )
        if research_areas:
            st.session_state.user_preferences['research_areas'] = [
                area.strip() for area in research_areas.split('\n') if area.strip()
            ]
        
        # Preferred sources
        sources = st.multiselect(
            "Preferred Sources",
            options=['arxiv', 'pubmed', 'semantic_scholar'],
            default=st.session_state.user_preferences['preferred_sources']
        )
        st.session_state.user_preferences['preferred_sources'] = sources
        
        # Summary style
        summary_style = st.selectbox(
            "Summary Style",
            options=['brief', 'detailed', 'technical'],
            index=['brief', 'detailed', 'technical'].index(
                st.session_state.user_preferences['summary_style']
            )
        )
        st.session_state.user_preferences['summary_style'] = summary_style
        
        st.divider()
        
        # Memory section
        st.header("🧠 Memory")
        if st.button("Clear Memory"):
            st.session_state.memory_manager.clear()
            st.success("Memory cleared!")
        
        # Show current context
        context = st.session_state.memory_manager.get_context()
        if context:
            st.text_area("Current Context", value=context, height=100, disabled=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Research Query")
        
        # Query input
        query = st.text_area(
            "Enter your research question:",
            height=100,
            placeholder="e.g., What are the latest developments in transformer architectures for natural language processing?"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            max_papers = st.slider("Maximum papers to analyze", 5, 50, 20)
            enable_critique = st.checkbox("Enable critique agent", value=True)
            human_feedback = st.checkbox("Require human feedback", value=True)
        
        # Process button
        if st.button("🚀 Start Research", type="primary", disabled=not query):
            with st.spinner("Processing your research query..."):
                
                # Update preferences
                preferences = st.session_state.user_preferences.copy()
                preferences.update({
                    'max_papers': max_papers,
                    'enable_critique': enable_critique,
                    'human_feedback': human_feedback
                })
                
                try:
                    # Process query
                    result = asyncio.run(process_research_query(query, preferences))
                    
                    # Store in history
                    st.session_state.research_history.append({
                        'timestamp': datetime.now(),
                        'query': query,
                        'result': result
                    })
                    
                    # Display results
                    st.success("Research completed!")
                    
                    # Show results tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📄 Papers", "🔍 Critique", "💡 Recommendations"])
                    
                    with tab1:
                        st.subheader("Research Summary")
                        st.write(result.get("final_output", "No summary available"))
                    
                    with tab2:
                        st.subheader("Analyzed Papers")
                        papers = result.get("papers", [])
                        for i, paper in enumerate(papers):
                            with st.expander(f"Paper {i+1}: {paper.get('title', 'Unknown Title')}"):
                                st.write(f"**Authors:** {paper.get('authors', 'Unknown')}")
                                st.write(f"**Year:** {paper.get('year', 'Unknown')}")
                                st.write(f"**Abstract:** {paper.get('abstract', 'No abstract available')}")
                                if paper.get('url'):
                                    st.write(f"**URL:** {paper['url']}")
                    
                    with tab3:
                        st.subheader("Critique and Analysis")
                        critique = result.get("critique", "No critique available")
                        st.write(critique)
                    
                    with tab4:
                        st.subheader("Recommendations")
                        recommendations = result.get("recommendations", [])
                        for rec in recommendations:
                            st.write(f"• {rec}")
                    
                    # Human feedback section if required
                    if result.get("requires_human_input", False):
                        st.warning("Human feedback required!")
                        feedback = st.text_area("Please provide your feedback:")
                        if st.button("Submit Feedback"):
                            # Process feedback (implement feedback loop)
                            st.success("Feedback submitted!")
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    
    with col2:
        st.header("📚 Research History")
        
        if st.session_state.research_history:
            for i, item in enumerate(reversed(st.session_state.research_history[-5:])):  # Show last 5
                with st.expander(f"Query {len(st.session_state.research_history) - i}"):
                    st.write(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Query:** {item['query'][:100]}...")
                    if st.button(f"View Details", key=f"detail_{i}"):
                        st.session_state[f"show_detail_{i}"] = True
        else:
            st.info("No research history yet. Start by entering a query!")
        
        # Statistics
        st.header("📊 Statistics")
        st.metric("Total Queries", len(st.session_state.research_history))
        st.metric("Memory Items", len(st.session_state.memory_manager.interactions))

if __name__ == "__main__":
    main()