from typing import Optional
from src.tools_utils import add_vendor_to_user, update_user_details
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# --- Pydantic Schemas for Tool Arguments ---
class AddVendorSchema(BaseModel):
    """Input schema for add_vendor_to_user."""
    user_id: int = Field(description="The ID of the user to whom the vendor should be added.")
    vendor_name: str = Field(description="The name of the vendor to add.")

class UpdateUserSchema(BaseModel):
    """Input schema for update_user_details."""
    user_id: int = Field(description="The ID of the user whose details need to be updated.")
    name: Optional[str] = Field(None, description="The updated name of the user.")
    phone: Optional[str] = Field(None, description="The updated phone number of the user.")
    address: Optional[str] = Field(None, description="The updated address of the user.")
    email: Optional[str] = Field(None, description="The updated email address of the user.")


# --- LangChain Tool Definitions ---
# These are the actual tool objects you would pass to your agent
add_vendor_tool = StructuredTool.from_function(
    func=add_vendor_to_user,
    name="add_vendor_to_user",
    description="Adds a vendor to the list of vendors associated with a specific user. "
                "Useful when the user wants to add a new vendor. "
                "Example: 'הוסף את ספק 'ABC' למשתמש 123' (Add vendor 'ABC' to user 123)",
    args_schema=AddVendorSchema,
)

update_user_tool = StructuredTool.from_function(
    func=update_user_details,
    name="update_user_details",
    description="Updates existing personal details (name, phone, address, email) of a user. "
                "Useful when the user wants to change their personal information. "
                "Examples: "
                "'עדכן את כתובת המייל של משתמש 1 ל-new.email@example.com' (Update user 1's email to new.email@example.com). "
                "'שנה את מספר הטלפון של לקוח 3 ל-052-9876543 ואת השם ל-Jane Doe' (Change customer 3's phone to 052-9876543 and name to Jane Doe).",
    args_schema=UpdateUserSchema,
)
