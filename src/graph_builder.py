# src/graph_builder.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.models import State
from src.ingester import vector_store
from src.config import LLM
from src.utils import trimmer
from src.agent import run_agent  # Assuming run_agent can be called directly as a node
from src.global_queue import add_user_to_global_queue  # Import the global queue function


# Define prompt for messages-answering
rag_chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question based on the following context. "
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            "\n\nContext: {context}"
            "\n\nAnswer to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


tool_routing_prompt = PromptTemplate.from_template(
    """Decide whether the following user message requires using tools (e.g., updating user details, adding vendors)
or if it's a general question that can be answered based on existing documents.

Respond only with one word: "tool" or "retrieve".

User message:
{message}
"""
)

# Define application steps (nodes)
def retrieve(state: State) -> dict:
    """
    Retrieves relevant documents from the vector store based on the latest user query.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        user_query_text = last_message.content
    else:
        # This case should ideally not happen if the previous node ensures HumanMessage
        raise ValueError("Expected the last message in state['messages'] to be a HumanMessage for retrieval.")

    retrieved_docs = vector_store.similarity_search(user_query_text)
    return {"context": retrieved_docs}


def generate(state: State) -> dict:
    """
    Generates a response using the LLM based on retrieved context and chat history.
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    # Apply the trimmer to the messages before passing them to the prompt
    trimmed_messages = trimmer.invoke(state["messages"])

    full_messages_for_llm = rag_chat_prompt.invoke(
        {"messages": trimmed_messages, "context": docs_content, "language": state["language"]}
    )

    response = LLM.invoke(full_messages_for_llm)

    # Check if the LLM's response indicates it doesn't know the answer
    # Now checks for both English and Hebrew "don't know" phrases
    if "don't know" in response.content.lower() or "לא יודע" in response.content.lower():
        user_session_id = state["session_id"]  # Get the session ID from the state

        # Add user to the global service queue
        add_user_to_global_queue(user_session_id)  # Use the global queue function
        # Ensure the response is in Hebrew as per the user's language setting
        return {"messages": [AIMessage(content="אני לא בטוח לגבי התשובה, הפניתי אותך לנציג שירות. אנא המתן.")]}
    else:
        return {"messages": [response]}


def route_question(state: State) -> str:
    """
    Uses an LLM to decide whether to route the user's query to the tool agent or to the RAG pipeline.
    """
    last_message = state["messages"][-1]

    if not isinstance(last_message, HumanMessage):
        raise ValueError("Expected last message to be a HumanMessage.")

    routing_prompt = tool_routing_prompt.invoke({"message": last_message.content})
    decision = LLM.invoke(routing_prompt).content.strip().lower()

    if "tool" in decision:
        return "tool_agent"
    return "retrieve"



def build_and_compile_graph():
    graph_builder = StateGraph(State)

    # Define the nodes (these are the actual processing steps)
    # The 'entry_point_router' node will simply pass the state along and immediately
    # trigger the conditional edge based on route_question.
    graph_builder.add_node("entry_point_router", lambda x: x)  # A simple pass-through node
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("tool_agent", run_agent)  # The node for running LangChain tools

    # Set the 'entry_point_router' node as the initial entry point
    graph_builder.set_entry_point("entry_point_router")

    # Add conditional edges from 'entry_point_router'
    # The 'route_question' function is used here to define the routing logic.
    graph_builder.add_conditional_edges(
        "entry_point_router",  # FROM this node
        route_question,  # Use this function to decide WHERE to go
        {
            "tool_agent": "tool_agent",  # If route_question returns "tool_agent", go to tool_agent
            "retrieve": "retrieve"  # If route_question returns "retrieve", go to retrieve
        }
    )

    # Define the rest of the RAG flow (after retrieve)
    graph_builder.add_edge("retrieve", "generate")

    # Set finish points for both potential flows
    graph_builder.set_finish_point("generate")
    graph_builder.set_finish_point("tool_agent")

    # Checkpoint for state persistence
    checkpointer = MemorySaver()

    return graph_builder.compile(checkpointer=checkpointer)


# The compiled graph instance
compiled_rag_graph = build_and_compile_graph()
