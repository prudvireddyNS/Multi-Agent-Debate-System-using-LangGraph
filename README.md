# Multi-Agent Debate System using LangGraph

A sophisticated and configurable debate simulation system where two AI agents engage in structured arguments using LangGraph's workflow orchestration. This project demonstrates advanced AI agent coordination with dynamic persona switching and comprehensive debate evaluation.

## 🎯 Overview

This system implements a complete, dynamic debate workflow with:

- **8 Configurable AI Personas**: Choose from Scientist, Philosopher, Economist, Lawyer, Environmentalist, Technologist, Sociologist, and Ethicist
- **Flexible Debate Configuration**: Customize rounds, temperature, judge style, and more
- **Interactive CLI Setup**: Easy-to-use command-line interface for debate configuration
- **Smart Memory Management**: Agents receive context from opponent's arguments and their own previous statements for coherent rebuttals
- **Automated Judging**: Impartial judge evaluates the debate with comprehensive analysis and clear winner declaration
- **Complete Logging**: Full transcript with timestamps saved to log files for review and analysis
- **Argument Validation**: Built-in detection and prevention of repetitive arguments

## 🗝️ Architecture

### DAG Structure

The Directed Acyclic Graph (DAG) ensures a logical and controlled flow for the debate:

```
START → UserInputNode → MemoryNode (Init) → Agent Selection
                                              ↓
                    ┌─────────────────────────┴─────────────────────────┐
                    ↓                                                   ↓
                 AgentA (Persona 1)                              AgentB (Persona 2)
                    ↓                                                   ↓
                    └─────────────────────────┬─────────────────────────┘
                                              ↓
                                      MemoryNode (Update)
                                              ↓
                                    [Check Round Limit?]
                                    ↙                    ↘
                            NO: Loop Back           YES: Continue
                                    ↓
                                 JudgeNode
                                    ↓
                                  END
```

### Node Descriptions

1. **UserInputNode**: Captures the debate topic from the user via CLI
2. **MemoryNode (Init)**: Initializes debate state, round counter, speaker tracking, and memory summary
3. **AgentA & AgentB**: Generate arguments based on assigned personas, with access to:
   - Debate topic
   - Opponent's last argument (if rebuttals enabled)
   - Their own previous arguments (last 2)
   - Persona-specific instructions
4. **MemoryNode (Update)**: Appends the latest argument to debate history after each turn
5. **JudgeNode**: After all rounds complete, analyzes the full transcript and provides:
   - Comprehensive debate summary
   - Analysis of each debater's strengths
   - Winner declaration with detailed justification

### State Management

The `DebateState` TypedDict tracks:
- `topic`: Debate subject
- `round_number`: Current round counter
- `current_speaker`: Tracks whose turn it is ('agent_a' or 'agent_b')
- `arguments`: List of all arguments with round, speaker, and content
- `memory_summary`: Running summary of the debate
- `agent_a_arguments` & `agent_b_arguments`: Individual argument histories
- `winner`: Final winner declaration
- `judgment`: Complete judge evaluation
- `config`: Configuration dictionary

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Google API Key (for Gemini models)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd multi-agent-debate-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY1=your_google_api_key_here
   ```

## 🎮 Usage

### Running the Debate

Execute the main script:

```bash
python main1.py
```

### Configuration Options

The system offers two configuration modes:

1. **Default Mode** (press 'n' or Enter):
   - Scientist vs Philosopher
   - 8 rounds
   - Temperature: 0.7
   - Comprehensive judge style

2. **Interactive Mode** (press 'y'):
   - Choose any two personas from 8 options
   - Set number of rounds
   - Adjust LLM temperature (0.0-1.0)
   - Select judge style (comprehensive/brief)

### Example Session

```
================================================================================
MULTI-AGENT DEBATE SYSTEM - LANGGRAPH (ENHANCED)
================================================================================
Log file: debate_log_20251010_143520.txt

Use interactive configuration? (y/n) [default: n]: y

================================================================================
DEBATE CONFIGURATION
================================================================================

Available personas:
1. Scientist
2. Philosopher
3. Economist
4. Lawyer
5. Environmentalist
6. Technologist
7. Sociologist
8. Ethicist

Select Agent A persona (1-8) [default: 1-Scientist]: 3
Select Agent B persona (1-8) [default: 2-Philosopher]: 5

Number of debate rounds [default: 8]: 4

LLM temperature (0.0-1.0) [default: 0.7]: 0.8

Judge style (comprehensive/brief) [default: comprehensive]: comprehensive

Configuration set:
  Agent A: Economist
  Agent B: Environmentalist
  Rounds: 4
  Temperature: 0.8
  Judge Style: comprehensive

================================================================================
NODE: UserInputNode
================================================================================

Enter topic for debate: Should carbon taxes be universally adopted?

Topic set: Should carbon taxes be universally adopted?
Initializing debate between Economist and Environmentalist...

[Round 1] Economist: Carbon taxes efficiently internalize environmental 
externalities, but universal adoption without considering varied economic 
contexts could disproportionately harm developing nations and low-income 
populations...

[Round 1] Environmentalist: The climate crisis demands urgent global action, 
and carbon taxes embody the "polluter pays" principle essential for driving 
systemic change toward sustainability...

[... Additional rounds continue ...]

================================================================================
FINAL JUDGMENT
================================================================================

SUMMARY: The debate examined universal carbon tax adoption through economic and 
environmental lenses. The Economist emphasized market efficiency while warning 
of implementation challenges and equity concerns. The Environmentalist stressed 
climate urgency and moral imperatives for immediate action.

ECONOMIST STRENGTHS: Provided nuanced analysis of economic impacts, particularly 
on vulnerable populations. Suggested practical alternatives and implementation 
considerations that demonstrated policy sophistication.

ENVIRONMENTALIST STRENGTHS: Maintained strong ethical framing with compelling 
urgency arguments. Effectively connected carbon pricing to broader sustainability 
goals and planetary health imperatives.

WINNER: Environmentalist
JUSTIFICATION: While both presented valid perspectives, the Environmentalist's 
arguments carried greater weight given the time-sensitive nature of climate change. 
The moral framework of "polluter pays" combined with ecological urgency provided 
a more compelling case than purely economic efficiency concerns.

================================================================================
Winner: Environmentalist
Total Rounds: 4
Full log saved to: debate_log_20251010_143520.txt

✅ Debate completed! Check debate_log_20251010_143520.txt for full transcript.
```

## 📁 Project Structure

```
multi-agent-debate-system/
├── main1.py               # Main application with LangGraph workflow
├── requirements.txt       # Python dependencies (langgraph, langchain-google-genai, etc.)
├── README.md              # This documentation file
├── .env                   # Environment variables (create this - not tracked)
└── debate_log_*.txt       # Generated timestamped log files
```

## 🔧 Configuration & Customization

### DebateConfig Class

The `DebateConfig` class supports:
- `total_rounds`: Number of debate rounds (default: 8)
- `agent_a_persona` & `agent_b_persona`: Selected persona keys (default: "scientist", "philosopher")
- `temperature`: LLM temperature for response variation (default: 0.7)
- `model`: Gemini model identifier (default: "gemini-2.0-flash-exp")
- `judge_style`: "comprehensive" or "brief" (default: "comprehensive")
- `argument_length`: Expected argument length (default: "2-3 sentences")
- `enable_rebuttals`: Whether agents see opponent's last argument (default: True)

### Adding New Personas

Add entries to the `PERSONAS` dictionary in `main1.py`:

```python
"your_persona": {
    "name": "Your Persona Name",
    "system_prompt": "You are a [role] who values [principles].",
    "instruction": "Make a compelling argument. Be concise (2-3 sentences). Focus on [key aspects]."
}
```

### Adjusting the LLM

Modify the `get_llm()` function or `DebateConfig` class:

```python
return ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Change model here
    temperature=config.temperature,
    api_key=os.getenv("GOOGLE_API_KEY1"),
    timeout=30,
    max_retries=2
)
```

### Configuration File Support

The system includes methods for saving/loading configurations:
- `config.save_to_file("my_config.json")` - Save current configuration
- `config.load_from_file("my_config.json")` - Load saved configuration

## 🛠️ Technical Features

### Error Handling
- API timeout protection (30 seconds)
- Automatic retry logic (2 retries)
- Graceful fallback for failed argument generation
- Comprehensive error logging

### Argument Quality Control
- Repetition detection and regeneration
- Duplicate argument filtering
- Context-aware response generation
- Memory-based coherence checking

### Logging System
- Timestamped log files
- Complete state transition tracking
- Both console and file output
- Structured argument history

## 📋 Requirements

Key dependencies (see `requirements.txt`):
- `langgraph` - Workflow orchestration framework
- `langchain-google-genai` - Google Gemini integration
- `langchain-core` - Core LangChain functionality
- `python-dotenv` - Environment variable management

## 🤝 Acknowledgments

- LangGraph team for the powerful workflow framework
- Google for the Gemini AI models
- LangChain community for excellent documentation and tools

---

**Framework**: LangGraph  
**Model**: Google Gemini 2.0 Flash  
**Python Version**: 3.10+