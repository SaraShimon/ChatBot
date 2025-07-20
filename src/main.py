from langchain_core.messages import HumanMessage, AIMessage
from src.graph_builder import compiled_rag_graph  # Import the compiled graph
from src.models import State  # Import State for type hinting
from src.global_queue import get_current_global_queue_size, peek_global_queue # Import global queue functions for testing


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
    initial_input: State = {
        "messages": [HumanMessage(content=query)],
        "language": language,
        "context": [],  # Context is populated by the retrieve node
        "session_id": session_id, # Add session_id to the state for global queue access
    }

    output = compiled_rag_graph.invoke(
        initial_input,
        config={"configurable": {"thread_id": session_id}},
    )

    if output and "messages" in output and len(output["messages"]) > 0:
        last_message_from_graph = output["messages"][-1]
        if isinstance(last_message_from_graph, AIMessage):
            return last_message_from_graph.content
        else:
            return f"Error: Unexpected last message type. Content: {last_message_from_graph.content}"
    else:
        return "Error: No response or invalid response structure from the model."


if __name__ == "__main__":
    print("Chatbot started. Type 'exit' to quit.")

    # For testing, you can use different session_ids
    test_session_id_1 = "user_alice"
    test_session_id_2 = "user_bob"

    while True:
        user_input = input("You (User Alice): ")
        if user_input.lower() == 'exit':
            break

        response_alice = ask_for_help(user_input, session_id=test_session_id_1)
        print(f"Bot (Alice): {response_alice}")
        print(f"Global queue status: {peek_global_queue()} (Size: {get_current_global_queue_size()})")

        user_input_bob = input("You (User Bob): ")
        if user_input_bob.lower() == 'exit':
            break

        response_bob = ask_for_help(user_input_bob, session_id=test_session_id_2)
        print(f"Bot (Bob): {response_bob}")
        print(f"Global queue status: {peek_global_queue()} (Size: {get_current_global_queue_size()})")
