from src.config import USERS_DATA_FILE, USER_VENDORS_FILE
from pathlib import Path
from typing import Optional
import json

def _read_json_file(file_path: Path) -> dict:
    """
    Reads content from a JSON file and returns a dictionary.
    If the file does not exist, an empty dictionary is returned.
    If the file content is not valid JSON, a warning is printed and an empty dictionary is returned.

    :param file_path: The path to the JSON file.
    :return: A dictionary representing the JSON content, or an empty dictionary if the file is
             missing or invalid.
    """
    if not file_path.exists():
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: File {file_path} is not a valid JSON format. An empty dictionary will be returned.")
        return {}


def _write_json_file(file_path: Path, data: dict):
    """
    Writes a dictionary to a JSON file.

    :param file_path: The path to the JSON file where data will be written.
    :param data: The dictionary to write.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --- Tool Definition: Add Vendor to User ---

def add_vendor_to_user(user_id: int, vendor_name: str) -> str:
    """
    Adds a vendor to the list of vendors associated with a specific user in 'user_vendors.json'.
    The file stores a dictionary where keys are user IDs and values are lists of vendor names.
    If the user already exists, the vendor will be appended to their existing list.
    If the user does not exist, a new list will be created for them.

    :param user_id: The ID of the user to whom the vendor should be added.
    :param vendor_name: The name of the vendor to add.
    :return: A string confirming the action or an error message.
    """
    data = _read_json_file(USER_VENDORS_FILE)
    vendors = data.get("vendors", {})

    user_id_str = str(user_id)  # User keys are stored as strings in JSON

    if user_id_str not in vendors:
        vendors[user_id_str] = []

    if vendor_name not in vendors[user_id_str]:
        vendors[user_id_str].append(vendor_name)
        data["vendors"] = vendors
        _write_json_file(USER_VENDORS_FILE, data)
        return f"Vendor '{vendor_name}' successfully added for user {user_id}."
    else:
        return f"Vendor '{vendor_name}' already exists for user {user_id}."


# --- Tool Definition: Update User Details ---

def update_user_details(user_id: int, name: Optional[str] = None, phone: Optional[str] = None,
                        address: Optional[str] = None, email: Optional[str] = None) -> str:
    """
    Updates existing details of a user in 'users_data.json'.
    This tool allows updating any of the user's details (name, phone, address, and/or email) specifically.

    :param user_id: The ID of the user whose details need to be updated.
    :param name: (Optional) The updated name of the user.
    :param phone: (Optional) The updated phone number of the user.
    :param address: (Optional) The updated address of the user.
    :param email: (Optional) The updated email address of the user.
    :return: A string confirming the action or an error message.
    """
    data = _read_json_file(USERS_DATA_FILE)
    users = data.get("users", {})

    user_id_str = str(user_id)  # User keys are stored as strings in JSON

    if user_id_str not in users:
        return f"Error: User with ID {user_id} not found."

    updated_fields = []
    if name is not None:
        users[user_id_str]["name"] = name
        updated_fields.append("name")
    if phone is not None:
        users[user_id_str]["phone"] = phone
        updated_fields.append("phone")
    if address is not None:
        users[user_id_str]["address"] = address
        updated_fields.append("address")
    if email is not None:
        users[user_id_str]["email"] = email
        updated_fields.append("email")

    if not updated_fields:
        return "No details provided for update."

    data["users"] = users
    _write_json_file(USERS_DATA_FILE, data)
    return f"Details for user {user_id} successfully updated: {', '.join(updated_fields)}."

# --- Examples of Function Usage (for testing) ---
if __name__ == "__main__":
    # Create empty files if they don't exist
    if not USER_VENDORS_FILE.exists():
        _write_json_file(USER_VENDORS_FILE, {"vendors": {}})
    if not USERS_DATA_FILE.exists():
        _write_json_file(USERS_DATA_FILE, {"users": {}})

    print("--- Testing add_vendor_to_user ---")
    print(add_vendor_to_user(2, "X"))
    print(add_vendor_to_user(1, "Y"))  # Attempt to add the same vendor again
    print(add_vendor_to_user(3, "Z"))
    print("\nContent of user_vendors.json after additions:")
    print(json.dumps(_read_json_file(USER_VENDORS_FILE), indent=2, ensure_ascii=False))

    print("\n--- Testing update_user_details ---")
    # Create an example user if not already present
    initial_users_data = _read_json_file(USERS_DATA_FILE)
    if "users" not in initial_users_data or "1" not in initial_users_data["users"]:
        initial_users_data.setdefault("users", {})["1"] = {
            "name": "John Doe",
            "phone": "050-1234567",
            "address": "Main St 1",
            "email": "s0556781304@gmail.com"
        }
        _write_json_file(USERS_DATA_FILE, initial_users_data)
        print("Initial user (ID 1) created for testing.")

    print(update_user_details(1, email="john.doe@example.com"))
    print(update_user_details(1, phone="054-9876543", address="New St 2"))
    print(update_user_details(1, name="John D. Doe"))
    print(update_user_details(999, name="Non Existent"))  # Non-existent user
    print(update_user_details(1))  # No details provided for update
    print("\nContent of users_data.json after updates:")
    print(json.dumps(_read_json_file(USERS_DATA_FILE), indent=2, ensure_ascii=False))
