from langchain_core.messages import HumanMessage, AIMessage
from src.graph_builder import compiled_rag_graph  # Import the compiled graph
from src.models import State  # Import State for type hinting


def ask_for_help(query: str, session_id: str = "default_thread", language: str = "Hebrew") -> str:
    """
    Main function to interact with the RAG chatbot.

    Args:
        query: The user's input query.
        session_id: A unique identifier for the conversation thread (e.g., Slack channel ID).
        language: The desired language for the chatbot's response.

    Returns:
        The content of the chatbot's response.
    """
    # The initial messages for the graph.invoke should contain only the current user's message.
    # LangGraph's checkpointer will load the previous messages for this thread_id.
    initial_input: State = {
        "messages": [HumanMessage(content=query)],
        "language": language,
        "context": [],  # Context is populated by the retrieve node
    }

    # Invoke the graph with the initial input and the session config
    output = compiled_rag_graph.invoke(
        initial_input,
        config={"configurable": {"thread_id": session_id}},
    )

    # Return only the content of the last message from the LLM's response
    # Ensure the output is a list of messages and get the content from the last one
    if output and "messages" in output and len(output["messages"]) > 0:
        last_message_from_graph = output["messages"][-1]
        if isinstance(last_message_from_graph, AIMessage):
            return last_message_from_graph.content
        else:
            # This case indicates the last message wasn't an AI message, which is unexpected for a final output.
            return f"Error: Unexpected last message type. Content: {last_message_from_graph.content}"
    else:
        return "Error: No response or invalid response structure from the model."


if __name__ == "__main__":
    # Example usage for testing
    print("Chatbot started. Type 'exit' to quit.")

    # For testing persistent memory, use a fixed session_id
    test_session_id = "test_user_123"

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = ask_for_help(user_input, session_id=test_session_id)
        print(f"Bot: {response}")
