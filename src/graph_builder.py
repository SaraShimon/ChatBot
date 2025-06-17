from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from src.models import State
from src.ingester import vector_store
from src.config import LLM
from src.utils import trimmer
from src.agent import run_agent


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


# Define application steps (nodes)
def retrieve(state: State) -> dict:
    """
    Retrieves relevant documents from the vector store based on the latest user query.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        user_query_text = last_message.content
    else:
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
    # LangGraph's add_messages will automatically handle adding this to the state's messages.
    return {"messages": [response]}  # Return the AI's message as a list for add_messages to process


def initial_router(state: State) -> str:
    last_message_content = state["messages"][-1].content.lower()
    if "update" in last_message_content or "change" in last_message_content:
        return "agent_node"
    else:
        return "retrieve"

def build_and_compile_graph():
    graph_builder = StateGraph(State)

    # Define the nodes
    graph_builder.add_node("initial_router_node", lambda x: x)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("agent_node", run_agent)

    # Set the initial_router_node as the entry point
    graph_builder.set_entry_point("initial_router_node")

    # Add conditional edges from the initial_router_node
    # Based on the initial_router function, decide where to go first
    graph_builder.add_conditional_edges(
        "initial_router_node",
        initial_router,
        {
            "agent_node": "agent_node",
            "retrieve": "retrieve"
        }
    )

    # Define the rest of the RAG flow (after retrieve)
    graph_builder.add_edge("retrieve", "generate")

    # (אופציונלי) אם אתה רוצה שהגרף יסתיים אחרי הסוכן או היצירה, הגדר את זה:
    graph_builder.set_finish_point("agent_node")
    graph_builder.set_finish_point("generate")

    # Checkpoint
    checkpointer = MemorySaver()

    return graph_builder.compile(checkpointer=checkpointer)

def router(state: State) -> str:
    last_message = state["messages"][-1].content.lower()
    if "update" in last_message or "change" in last_message:
        return "agent"
    else:
        return "generate"


# The compiled graph instance
compiled_rag_graph = build_and_compile_graph()
