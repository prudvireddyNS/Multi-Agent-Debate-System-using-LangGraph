"""
Multi-Agent Debate System using LangGraph
Enhanced Version with Configurable Features
"""

import os
import json
from datetime import datetime
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Validate API key
if not os.getenv("GOOGLE_API_KEY1"):
    raise ValueError("GOOGLE_API_KEY1 not found in environment variables. Please check your .env file.")

# Initialize logging
LOG_FILE = f"debate_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message: str):
    """Log message to both console and file"""
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# Configurable Personas Dictionary
PERSONAS = {
    "scientist": {
        "name": "Scientist",
        "system_prompt": "You are a rational scientist who values empirical evidence and practical outcomes.",
        "instruction": "Make a compelling scientific argument. Be concise (2-3 sentences). Focus on evidence, data, and practical implications."
    },
    "philosopher": {
        "name": "Philosopher",
        "system_prompt": "You are a thoughtful philosopher who values ethics, autonomy, and deeper meaning.",
        "instruction": "Make a compelling philosophical argument. Be concise (2-3 sentences). Focus on ethics, principles, and long-term implications."
    },
    "economist": {
        "name": "Economist",
        "system_prompt": "You are an economist who analyzes market dynamics, incentives, and economic efficiency.",
        "instruction": "Make a compelling economic argument. Be concise (2-3 sentences). Focus on costs, benefits, market forces, and economic impact."
    },
    "lawyer": {
        "name": "Lawyer",
        "system_prompt": "You are a legal expert who focuses on laws, regulations, rights, and legal precedents.",
        "instruction": "Make a compelling legal argument. Be concise (2-3 sentences). Focus on legal frameworks, rights, precedents, and constitutional matters."
    },
    "environmentalist": {
        "name": "Environmentalist",
        "system_prompt": "You are an environmental activist who prioritizes ecological sustainability and planetary health.",
        "instruction": "Make a compelling environmental argument. Be concise (2-3 sentences). Focus on ecological impact, sustainability, and long-term planetary health."
    },
    "technologist": {
        "name": "Technologist",
        "system_prompt": "You are a technology innovator who focuses on progress, disruption, and technological solutions.",
        "instruction": "Make a compelling technological argument. Be concise (2-3 sentences). Focus on innovation, efficiency, and technological advancement."
    },
    "sociologist": {
        "name": "Sociologist",
        "system_prompt": "You are a sociologist who analyzes social structures, cultural dynamics, and community impact.",
        "instruction": "Make a compelling sociological argument. Be concise (2-3 sentences). Focus on social impact, community welfare, and cultural considerations."
    },
    "ethicist": {
        "name": "Ethicist",
        "system_prompt": "You are an ethicist who examines moral principles, values, and ethical implications.",
        "instruction": "Make a compelling ethical argument. Be concise (2-3 sentences). Focus on moral principles, fairness, and ethical considerations."
    }
}

# Configuration Class
class DebateConfig:
    def __init__(self):
        self.total_rounds = 8
        self.agent_a_persona = "scientist"
        self.agent_b_persona = "philosopher"
        self.temperature = 0.7
        self.model = "gemini-2.0-flash-exp"
        self.judge_style = "comprehensive"  # or "brief"
        self.argument_length = "2-3 sentences"
        self.enable_rebuttals = True
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                self.__dict__.update(config_data)
                log_message(f"Configuration loaded from {config_file}")
        except FileNotFoundError:
            log_message(f"Config file {config_file} not found. Using defaults.")
        except Exception as e:
            log_message(f"Error loading config: {e}. Using defaults.")
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file"""
        with open(config_file, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
        log_message(f"Configuration saved to {config_file}")
    
    def interactive_setup(self):
        """Interactive CLI configuration"""
        log_message("\n" + "="*80)
        log_message("DEBATE CONFIGURATION")
        log_message("="*80)
        
        # Choose personas
        print("\nAvailable personas:")
        for i, persona in enumerate(PERSONAS.keys(), 1):
            print(f"{i}. {PERSONAS[persona]['name']}")
        
        try:
            choice_a = input(f"\nSelect Agent A persona (1-{len(PERSONAS)}) [default: 1-Scientist]: ").strip()
            if choice_a and choice_a.isdigit():
                self.agent_a_persona = list(PERSONAS.keys())[int(choice_a) - 1]
            
            choice_b = input(f"Select Agent B persona (1-{len(PERSONAS)}) [default: 2-Philosopher]: ").strip()
            if choice_b and choice_b.isdigit():
                self.agent_b_persona = list(PERSONAS.keys())[int(choice_b) - 1]
            
            # Number of rounds
            rounds_input = input(f"\nNumber of debate rounds [default: 8]: ").strip()
            if rounds_input and rounds_input.isdigit():
                self.total_rounds = int(rounds_input)
            
            # Temperature
            temp_input = input(f"LLM temperature (0.0-1.0) [default: 0.7]: ").strip()
            if temp_input:
                try:
                    self.temperature = float(temp_input)
                except ValueError:
                    pass
            
            # Judge style
            judge_input = input(f"Judge style (comprehensive/brief) [default: comprehensive]: ").strip().lower()
            if judge_input in ['comprehensive', 'brief']:
                self.judge_style = judge_input
            
        except Exception as e:
            log_message(f"Configuration error: {e}. Using defaults.")
        
        log_message(f"\nConfiguration set:")
        log_message(f"  Agent A: {PERSONAS[self.agent_a_persona]['name']}")
        log_message(f"  Agent B: {PERSONAS[self.agent_b_persona]['name']}")
        log_message(f"  Rounds: {self.total_rounds}")
        log_message(f"  Temperature: {self.temperature}")
        log_message(f"  Judge Style: {self.judge_style}")

# Global config instance
config = DebateConfig()

# Define the state structure
class DebateState(TypedDict):
    topic: str
    round_number: int
    current_speaker: str
    arguments: list
    memory_summary: str
    agent_a_arguments: list
    agent_b_arguments: list
    winner: str
    judgment: str
    config: dict

# Initialize LLM
def get_llm():
    """Get configured LLM instance"""
    try:
        return ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            api_key=os.getenv("GOOGLE_API_KEY1"),
            timeout=30,
            max_retries=2
        )
    except Exception as e:
        log_message(f"ERROR: Failed to initialize Google AI client - {str(e)}")
        raise

# Node 1: User Input Node
def user_input_node(state: DebateState) -> DebateState:
    """Accept debate topic from user"""
    log_message("\n" + "="*80)
    log_message("NODE: UserInputNode")
    log_message("="*80)
    
    if not state.get('topic'):
        topic = input("\nEnter topic for debate: ").strip()
        if not topic:
            topic = "Should AI be regulated like medicine?"
            log_message("No topic provided. Using default topic.")
        state['topic'] = topic
    
    log_message(f"Topic set: {state['topic']}")
    log_message(f"Initializing debate between {PERSONAS[config.agent_a_persona]['name']} and {PERSONAS[config.agent_b_persona]['name']}...")
    
    return state

# Node 2: Memory Initialization Node
def memory_init_node(state: DebateState) -> DebateState:
    """Initialize memory and debate parameters"""
    log_message("\n" + "="*80)
    log_message("NODE: MemoryNode (Initialization)")
    log_message("="*80)
    
    state['round_number'] = 0
    state['current_speaker'] = 'agent_a'
    state['arguments'] = []
    state['memory_summary'] = f"Debate Topic: {state['topic']}\n\nDebate Summary:\n"
    state['agent_a_arguments'] = []
    state['agent_b_arguments'] = []
    state['config'] = config.__dict__.copy()
    
    log_message("Memory initialized successfully")
    log_message(f"Starting round: 1")
    log_message(f"First speaker: {PERSONAS[config.agent_a_persona]['name']} (Agent A)")
    
    return state

# Node 3: Agent A
def agent_a_node(state: DebateState) -> DebateState:
    """Agent A makes an argument"""
    persona = PERSONAS[config.agent_a_persona]
    
    log_message("\n" + "="*80)
    log_message(f"NODE: AgentA ({persona['name']}) - Round {state['round_number'] + 1}")
    log_message("="*80)
    
    if state['current_speaker'] != 'agent_a':
        log_message("ERROR: Not Agent A's turn!")
        return state
    
    # Prepare context
    context = f"You are a {persona['name']} debating the topic: {state['topic']}\n\n"
    
    if config.enable_rebuttals and state['agent_b_arguments']:
        context += f"Opponent's last argument: {state['agent_b_arguments'][-1]}\n\n"
    
    if state['agent_a_arguments']:
        context += f"Your previous arguments: {', '.join(state['agent_a_arguments'][-2:])}\n\n"
    
    context += persona['instruction']
    
    # Generate argument
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=persona['system_prompt']),
            HumanMessage(content=context)
        ]
        
        response = llm.invoke(messages)
        argument = response.content.strip()
        
        # Validate no repetition
        if argument in state['agent_a_arguments']:
            log_message("WARNING: Repeated argument detected, regenerating...")
            response = llm.invoke(messages + [HumanMessage(content="Provide a different argument.")])
            argument = response.content.strip()
        
    except Exception as e:
        log_message(f"ERROR: Agent A failed to generate argument - {str(e)}")
        argument = "I acknowledge the complexity of this issue and defer to further research."
    
    log_message(f"[Round {state['round_number'] + 1}] {persona['name']}: {argument}")

    # Create new argument dictionary
    new_argument_dict = {
        'round': state['round_number'] + 1,
        'speaker': persona['name'],
        'argument': argument
    }
    
    # Update state
    updated_state = dict(state)
    updated_state['arguments'] = state['arguments'] + [new_argument_dict]
    updated_state['agent_a_arguments'] = state['agent_a_arguments'] + [argument]
    updated_state['round_number'] = state['round_number'] + 1
    updated_state['current_speaker'] = 'agent_b'
    
    return updated_state

# Node 4: Agent B
def agent_b_node(state: DebateState) -> DebateState:
    """Agent B makes an argument"""
    persona = PERSONAS[config.agent_b_persona]
    
    log_message("\n" + "="*80)
    log_message(f"NODE: AgentB ({persona['name']}) - Round {state['round_number'] + 1}")
    log_message("="*80)
    
    if state['current_speaker'] != 'agent_b':
        log_message("ERROR: Not Agent B's turn!")
        return state
    
    # Prepare context
    context = f"You are a {persona['name']} debating the topic: {state['topic']}\n\n"
    
    if config.enable_rebuttals and state['agent_a_arguments']:
        context += f"Opponent's last argument: {state['agent_a_arguments'][-1]}\n\n"
    
    if state['agent_b_arguments']:
        context += f"Your previous arguments: {', '.join(state['agent_b_arguments'][-2:])}\n\n"
    
    context += persona['instruction']
    
    # Generate argument
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=persona['system_prompt']),
            HumanMessage(content=context)
        ]
        
        response = llm.invoke(messages)
        argument = response.content.strip()
        
        # Validate no repetition
        if argument in state['agent_b_arguments']:
            log_message("WARNING: Repeated argument detected, regenerating...")
            response = llm.invoke(messages + [HumanMessage(content="Provide a different argument.")])
            argument = response.content.strip()
        
    except Exception as e:
        log_message(f"ERROR: Agent B failed to generate argument - {str(e)}")
        argument = "I concede this point requires deeper contemplation beyond our current framework."
    
    log_message(f"[Round {state['round_number'] + 1}] {persona['name']}: {argument}")

    # Create new argument dictionary
    new_argument_dict = {
        'round': state['round_number'] + 1,
        'speaker': persona['name'],
        'argument': argument
    }
    
    # Update state
    updated_state = dict(state)
    updated_state['arguments'] = state['arguments'] + [new_argument_dict]
    updated_state['agent_b_arguments'] = state['agent_b_arguments'] + [argument]
    updated_state['round_number'] = state['round_number'] + 1
    updated_state['current_speaker'] = 'agent_a'
    
    return updated_state

# Node 5: Memory Update Node
def memory_update_node(state: DebateState) -> DebateState:
    """Update memory with latest arguments"""
    log_message("\n" + "="*80)
    log_message("NODE: MemoryNode (Update)")
    log_message("="*80)
    
    if state['arguments']:
        latest_arg = state['arguments'][-1]
        memory_summary = state['memory_summary'] + f"Round {latest_arg['round']} - {latest_arg['speaker']}: {latest_arg['argument']}\n\n"
        
        log_message(f"Memory updated with Round {latest_arg['round']}")
        log_message(f"Total rounds completed: {state['round_number']}")
        
        updated_state = dict(state)
        updated_state['memory_summary'] = memory_summary
        return updated_state
    
    return state

# Node 6: Judge Node
def judge_node(state: DebateState) -> DebateState:
    """Evaluate debate and declare winner"""
    log_message("\n" + "="*80)
    log_message("NODE: JudgeNode (Final Evaluation)")
    log_message("="*80)
    
    # Remove duplicates
    unique_arguments = []
    seen = set()
    for arg in state['arguments']:
        key = (arg['round'], arg['speaker'])
        if key not in seen:
            seen.add(key)
            unique_arguments.append(arg)
    
    log_message(f"Total arguments collected: {len(state['arguments'])}")
    log_message(f"Unique arguments: {len(unique_arguments)}")
    
    # Prepare debate transcript
    debate_transcript = f"Topic: {state['topic']}\n\n"
    for arg in unique_arguments:
        debate_transcript += f"[Round {arg['round']}] {arg['speaker']}: {arg['argument']}\n\n"
    
    # Choose judge prompt based on config
    if config.judge_style == "brief":
        judge_prompt = f"""Evaluate this debate and declare a winner.

{debate_transcript}

Respond in this EXACT format:
WINNER: [{PERSONAS[config.agent_a_persona]['name']} or {PERSONAS[config.agent_b_persona]['name']}]
REASON: [One sentence explaining why]"""
    else:  # comprehensive
        judge_prompt = f"""You are an impartial judge evaluating a debate.

{debate_transcript}

Provide:
1. A comprehensive summary of the debate (3-4 sentences)
2. Analysis of each debater's strengths
3. Declare the winner ({PERSONAS[config.agent_a_persona]['name']} or {PERSONAS[config.agent_b_persona]['name']})
4. Provide clear logical justification for your decision

Format your response as:
SUMMARY: [summary]
{PERSONAS[config.agent_a_persona]['name'].upper()} STRENGTHS: [analysis]
{PERSONAS[config.agent_b_persona]['name'].upper()} STRENGTHS: [analysis]
WINNER: [{PERSONAS[config.agent_a_persona]['name']}/{PERSONAS[config.agent_b_persona]['name']}]
JUSTIFICATION: [detailed reasoning]"""
    
    try:
        log_message("Invoking judge LLM...")
        
        llm = get_llm()
        messages = [
            SystemMessage(content="You are a debate judge. Be objective and thorough."),
            HumanMessage(content=judge_prompt)
        ]
        
        response = llm.invoke(messages, config={"timeout": 30})
        judgment = response.content.strip()
        
        log_message(f"Judge response received: {judgment[:200]}...")
        
        # Parse winner - Look for explicit winner declaration first
        judgment_lower = judgment.lower()
        agent_a_name = PERSONAS[config.agent_a_persona]['name'].lower()
        agent_b_name = PERSONAS[config.agent_b_persona]['name'].lower()
        
        # First, check for explicit WINNER: declaration
        winner_line = None
        for line in judgment.split('\n'):
            if 'winner:' in line.lower():
                winner_line = line.lower()
                break
        
        if winner_line:
            # Extract winner from the winner line specifically
            if agent_a_name in winner_line:
                winner = PERSONAS[config.agent_a_persona]['name']
            elif agent_b_name in winner_line:
                winner = PERSONAS[config.agent_b_persona]['name']
            elif "tie" in winner_line or "draw" in winner_line:
                winner = "Tie"
            else:
                winner = "Undecided"
                log_message("WARNING: Could not parse winner from WINNER line")
        else:
            # Fallback: search entire judgment
            if agent_a_name in judgment_lower and agent_b_name not in judgment_lower:
                winner = PERSONAS[config.agent_a_persona]['name']
            elif agent_b_name in judgment_lower and agent_a_name not in judgment_lower:
                winner = PERSONAS[config.agent_b_persona]['name']
            elif "tie" in judgment_lower or "draw" in judgment_lower:
                winner = "Tie"
            else:
                winner = "Undecided"
                log_message("WARNING: Could not determine clear winner from judgment")
        
    except Exception as e:
        log_message(f"ERROR: Judge evaluation failed - {str(e)}")
        judgment = f"Unable to complete evaluation. Error: {str(e)}"
        winner = "Undecided"
    
    updated_state = dict(state)
    updated_state['winner'] = winner
    updated_state['judgment'] = judgment
    
    log_message("\n" + "="*80)
    log_message("FINAL JUDGMENT")
    log_message("="*80)
    log_message(f"\n{judgment}\n")
    log_message(f"Winner: {winner}")
    log_message("="*80)
    
    return updated_state

# Routing function
def should_continue(state: DebateState) -> str:
    """Determine next node based on round number"""
    if state['round_number'] >= config.total_rounds:
        return 'judge'
    elif state['current_speaker'] == 'agent_a':
        return 'agent_a'
    else:
        return 'agent_b'

# Build the graph
def build_debate_graph():
    """Construct the LangGraph debate workflow"""
    workflow = StateGraph(DebateState)
    
    # Add nodes
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("memory_init", memory_init_node)
    workflow.add_node("agent_a", agent_a_node)
    workflow.add_node("agent_b", agent_b_node)
    workflow.add_node("memory_update", memory_update_node)
    workflow.add_node("judge", judge_node)
    
    # Add edges
    workflow.set_entry_point("user_input")
    workflow.add_edge("user_input", "memory_init")
    workflow.add_conditional_edges(
        "memory_init",
        should_continue,
        {
            'agent_a': 'agent_a',
            'agent_b': 'agent_b',
            'judge': 'judge'
        }
    )
    workflow.add_edge("agent_a", "memory_update")
    workflow.add_edge("agent_b", "memory_update")
    workflow.add_conditional_edges(
        "memory_update",
        should_continue,
        {
            'agent_a': 'agent_a',
            'agent_b': 'agent_b',
            'judge': 'judge'
        }
    )
    workflow.add_edge("judge", END)
    
    return workflow.compile()

# Main execution
if __name__ == "__main__":
    log_message("="*80)
    log_message("MULTI-AGENT DEBATE SYSTEM - LANGGRAPH (ENHANCED)")
    log_message("="*80)
    log_message(f"Log file: {LOG_FILE}")
    
    try:
        # Configuration
        use_config = input("\nUse interactive configuration? (y/n) [default: n]: ").strip().lower()
        if use_config == 'y':
            config.interactive_setup()
        else:
            log_message(f"Using default configuration: {PERSONAS[config.agent_a_persona]['name']} vs {PERSONAS[config.agent_b_persona]['name']}, {config.total_rounds} rounds")
        
        # Build and run the graph
        app = build_debate_graph()
        
        # Initialize state
        initial_state = {
            'topic': '',
            'round_number': 0,
            'current_speaker': 'agent_a',
            'arguments': [],
            'memory_summary': '',
            'agent_a_arguments': [],
            'agent_b_arguments': [],
            'winner': '',
            'judgment': '',
            'config': {}
        }
        
        # Run the debate
        final_state = app.invoke(initial_state)
        
        log_message("\n" + "="*80)
        log_message("DEBATE COMPLETED")
        log_message("="*80)
        log_message(f"Winner: {final_state['winner']}")
        log_message(f"Total Rounds: {final_state['round_number']}")
        log_message(f"Full log saved to: {LOG_FILE}")
        
        print(f"\n✅ Debate completed! Check {LOG_FILE} for full transcript.")
        
    except Exception as e:
        log_message(f"\n❌ CRITICAL ERROR: {str(e)}")
        log_message("Debate terminated abnormally.")
        raise