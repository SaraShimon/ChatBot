from collections import deque
import threading
import json
import os
import atexit
from src.config import QUEUE_FILE

def _load_queue_from_file() -> deque:
    """
    Loads the global service queue from a JSON file.
    If the file does not exist or is corrupted, it initializes an empty queue.

    Returns:
        A deque object populated with the queue data from the file, or an empty deque.
    """
    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE, "r") as f:
                data = json.load(f)
                # Ensure 'queue' key exists in the loaded data, default to empty list
                return deque(data.get("queue", []))
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {QUEUE_FILE}. Starting with empty queue.")
            return deque()
    # If file does not exist, return an empty deque
    return deque()

def _save_queue_to_file():
    """
    Saves the current state of the global service queue to a JSON file.
    This function is called internally after modifications to the queue.
    """
    with open(QUEUE_FILE, "w") as f:
        # Convert deque to a list for JSON serialization
        json.dump({"queue": list(global_service_queue)}, f)
    print(f"Global queue saved to {QUEUE_FILE}.")

# Initialize the global service queue by loading from the file
global_service_queue = _load_queue_from_file()
# Create a lock to ensure thread-safe operations on the queue
queue_lock = threading.Lock()

def add_user_to_global_queue(session_id: str):
    """
    Adds a user's session ID to the global service queue in a thread-safe manner.
    The queue is saved to a file after the addition.

    Args:
        session_id: The unique session ID of the user to be added.
    """
    with queue_lock:
        if session_id not in global_service_queue: # Prevent adding the same user multiple times
            global_service_queue.append(session_id)
            _save_queue_to_file() # Save the queue after modification
            print(f"User '{session_id}' added to global service queue. Current queue: {list(global_service_queue)}")
        else:
            print(f"User '{session_id}' is already in the global service queue.")

def get_next_user_from_global_queue() -> str | None:
    """
    Retrieves and removes the next user's session ID from the global service queue
    in a thread-safe manner. The queue is saved to a file after the removal.

    Returns:
        The session ID of the next user in the queue, or None if the queue is empty.
    """
    with queue_lock:
        if global_service_queue:
            user_id = global_service_queue.popleft() # Remove the user from the front of the queue
            _save_queue_to_file() # Save the queue after modification
            print(f"User '{user_id}' removed from global service queue. Remaining queue: {list(global_service_queue)}")
            return user_id
        print("DEBUG: Attempted to get user from empty queue.") # Added for debugging based on previous conversation
        return None

def get_current_global_queue_size() -> int:
    """
    Returns the current number of users waiting in the global service queue.

    Returns:
        The number of items in the queue.
    """
    with queue_lock:
        return len(global_service_queue)

def peek_global_queue() -> list[str]:
    """
    Returns a copy of the current global service queue without removing any elements.

    Returns:
        A list representing the current state of the queue.
    """
    with queue_lock:
        return list(global_service_queue)

# Register a cleanup function to save the queue state when the script exits normally.
# Note: This might not catch all exit scenarios (e.g., forced termination like kill -9).
atexit.register(_save_queue_to_file)
