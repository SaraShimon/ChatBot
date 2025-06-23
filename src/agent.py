from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from src.tools import add_vendor_tool, update_user_tool
from src.config import LLM
from src.models import State

# Define the tools to be used
tools = [add_vendor_tool, update_user_tool]

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a smart assistant that uses tools to help users."),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the tool-aware agent
agent = create_tool_calling_agent(
    llm=LLM,
    tools=tools,
    prompt=prompt
)

# Wrap the agent in an executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # Optional: for robustness
)


# Function to run the agent with LangGraph-compatible state
def run_agent(state: State) -> dict:
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)

    if last_human is None:
        raise ValueError("No HumanMessage found in messages")

    result = agent_executor.invoke({"messages": state["messages"]})

    return {
        "messages": state["messages"] + [AIMessage(content=result["output"])]
    }


if __name__ == '__main__':
    response = run_agent({"messages": [HumanMessage(content="Please update user 1 to status closed.")]})
    print(response['messages'][-1].content)
