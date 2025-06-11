# AI Research Assistant for Technical Literature Review

An intelligent multi-agent system designed to streamline technical literature review by automating research tasks, managing information overload, and providing structured insights through AI-powered analysis.

## 🎯 Project Overview

This AI Research Assistant is a sophisticated system that helps researchers navigate and analyze technical literature efficiently. It employs a multi-agent architecture to:
- Process user research questions
- Search across multiple academic APIs (ArXiv, PubMed, Semantic Scholar)
- Generate comprehensive paper summaries and critiques
- Suggest research directions and next steps
- Maintain context through persistent memory
- Incorporate human feedback in decision-making
- Provide traceable and evaluable research processes

## 🧠 Why This Project Matters

- **Solves Real Problems**: Addresses information overload in research by automating literature review tasks
- **Advanced Architecture**: Implements cutting-edge AI concepts including:
  - Large Language Models (LLMs)
  - Persistent Memory Systems
  - Tool Integration
  - Evaluation Frameworks
  - Human-in-the-Loop Design
- **Technical Excellence**: Demonstrates deep understanding of:
  - LangGraph for agent orchestration
  - Agentic patterns and workflows
  - Modern AI system design

## 🛠️ Key Features

| Feature | Technology / Methodology |
|---------|-------------------------|
| Multi-agent Coordination | LangGraph nodes, state machine logic |
| Supervisor & Worker Agents | Agentic patterns (Planner, Critique, Supervisor) |
| Tool Integration | ArXiv, PubMed, Semantic Scholar APIs |
| Memory Management | LangGraph with LangChain ConversationBuffer |
| Human-in-the-Loop | Interactive feedback and confirmation |
| Evaluation & Tracing | LangSmith integration for monitoring and scoring |

## 📋 Project Architecture

```
User Input
   |
   v
[Planner Agent] ---> Decides which agents/tools to trigger
   |
   +---> [Tool Agent] --> Search APIs, retrieve metadata & abstracts
   |
   +---> [Summarizer Agent] --> Summarizes relevant papers
   |
   +---> [Critique Agent] --> Checks if answer quality is sufficient
   |
   +---> [Memory Manager] --> Persists session history, user preferences
   |
   v
[Supervisor Agent]
   |
   +---> If output is OK --> Done
   +---> If unclear --> Ask human for input (Human-in-the-loop)
   |
   v
[Output + LangSmith Trace + Score]
```

## 📁 Project Structure

```
├── README.md
├── main.py                 # Main entry point and agent orchestration
├── graph/                  # Agent definitions and orchestration
│   ├── planner_agent.py    # Research strategy and task planning
│   ├── summarizer_agent.py # Paper summarization and analysis
│   ├── tool_agent.py       # API integration and tool management
│   ├── critique_agent.py   # Quality assessment and validation
│   └── supervisor_agent.py # Agent coordination and human interaction
├── tools/                  # External API integrations
│   └── arxiv_api.py        # ArXiv API client
├── memory/                 # State and context management
│   └── memory_manager.py   # Persistent memory implementation
├── evaluation/            # Monitoring and evaluation
│   └── langsmith_config.py # LangSmith integration
├── .env                   # Environment configuration
└── requirements.txt       # Project dependencies
```

## 🚀 Getting Started

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys for:
     - OpenAI
     - LangSmith
     - ArXiv
     - PubMed
     - Semantic Scholar

## 💡 Usage

1. Start the research assistant:
   ```bash
   python main.py
   ```
2. Enter your research question
3. Review and provide feedback on:
   - Search results
   - Paper summaries
   - Research suggestions
4. Access evaluation metrics through LangSmith

## 🔄 Workflow

1. **Input Processing**: User research question is received and parsed
2. **Planning**: Planner agent determines research strategy
3. **Execution**: Tool agent searches APIs for relevant papers
4. **Analysis**: Summarizer and Critique agents process papers
5. **Review**: Supervisor agent coordinates with human feedback
6. **Output**: Final research summary with next steps
7. **Evaluation**: LangSmith traces and scores the process

## 📊 Evaluation

The system uses LangSmith for:
- Tracing agent interactions
- Collecting human feedback
- Scoring response quality
- Monitoring system performance
- Identifying improvement areas

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - See LICENSE file for details 