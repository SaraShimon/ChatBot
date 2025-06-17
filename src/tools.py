from langchain.tools import tool
import json
from src.config import USERS_DATA_FILE_PATH

def update_user_status_tool(user_id: str, status_value: str, file_path: str = USERS_DATA_FILE_PATH) -> str:
    try:
        # Load existing data
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    # Ensure 'users' exists
    if "users" not in data:
        data["users"] = {}

    # Ensure the specific user exists
    if user_id not in data["users"]:
        data["users"][user_id] = {}

    # Update the key-value pair
    data['users'][user_id]['status'] = status_value

    # Save the updated data
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    return f"Updated user id: '{user_id}' to status '{status_value}' in {file_path}."

@tool("update_user_status",
      description=(
          "Use this tool to update the status of a user given the user ID and the new status value. "
          "Input should be two parameters: 'key' for user ID and 'value' for status. "
          "Examples: key='2', value='active'."
      ))
def update_user_status_json_entry(key: str, value: str) -> str:
    try:
        return update_user_status_tool(key, value)
    except Exception as e:
        return f"Failed to update: {e}"


if __name__ == '__main__':
    update_user_status_tool('1', 'old')
    update_user_status_tool('2', 'new')
