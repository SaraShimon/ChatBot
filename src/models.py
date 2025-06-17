from typing import Annotated, Sequence, List, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph import add_messages

# Define the state of our LangGraph application
class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A sequence of messages forming the chat history.
                  Annotated with `add_messages` to automatically manage
                  adding new messages to the state.
        language: The language to use for generating responses.
        context: A list of Document objects retrieved from the vector store,
                 providing relevant context for the LLM.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    context: List[Document]
