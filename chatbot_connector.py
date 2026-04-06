import os
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Try to import Google Sheets logger (optional dependency)
try:
    from google_sheets_logger import ChatLogger
    SHEETS_LOGGING_AVAILABLE = True
except ImportError:
    SHEETS_LOGGING_AVAILABLE = False
    print("ℹ️  Google Sheets logging not available (google_sheets_logger.py not found)")

class ChatbotConnector:
    def __init__(self):
        """
        Initialize Supabase client and Google Sheets logger
        
        Args:
            google_sheet_name: Optional name of the specific sheet/tab to use for logging
                              Example: "ChatLogs", "Conversations", "FineTuning"
                              If not provided, uses the first sheet in the spreadsheet
        """
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        
        self.client: Client = create_client(supabase_url, supabase_key)
        
        # Initialize Google Sheets logger (optional)
        self.sheets_logger = None
        self._pending_user_message = None  # Store user message until we get AI response
        self._pending_student_id = None    # Store student ID
        
        if SHEETS_LOGGING_AVAILABLE:
            try:
                self.sheets_logger = ChatLogger()
                if self.sheets_logger.enabled:
                    print("✅ Conversation logging to Google Sheets enabled")
            except Exception as e:
                print(f"⚠️  Could not enable Google Sheets logging: {e}")
    
    def get_student_data(self, student_id: str):
        """
        Get complete student data including chat history and academic details
        
        Args:
            student_id: The unique student ID
            
        Returns:
            dict: Student data or None if not found
        """
        try:
            response = self.client.table("students").select("*").eq("student_id", student_id).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error fetching student: {e}")
            return None
    
    def get_chat_history(self, student_id: str):
        """
        Get chat history for a student
        
        Args:
            student_id: The unique student ID
            
        Returns:
            dict: Chat history in format:
                {
                    "conversation_id": "uuid",
                    "chat_history": [
                        {"role": "user", "content": "...", "timestamp": "..."},
                        ...
                    ]
                }
        """
        student = self.get_student_data(student_id)
        if student:
            chat_history = student.get("chat_history")
            
            # If no chat history exists, initialize it
            if not chat_history or not isinstance(chat_history, dict):
                return {
                    "conversation_id": str(uuid.uuid4()),
                    "chat_history": []
                }
            
            return chat_history
        return None
    
    def get_academic_details(self, student_id: str):
        """
        Get academic details for a student
        
        Args:
            student_id: The unique student ID
            
        Returns:
            dict: Academic details or None
        """
        student = self.get_student_data(student_id)
        if student:
            return student.get("academic_details")
        return None
    
    def add_message(self, student_id: str, role: str, content: str):
        """
        Add a message to student's chat history.
        Maintains only the last 3 user messages and 3 assistant messages.
        
        For Google Sheets logging: Stores user message, then logs both together
        when assistant message arrives (one row per conversation turn).
        
        Args:
            student_id: The unique student ID
            role: Either 'user' or 'assistant'
            content: The message content
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current chat history
            chat_history = self.get_chat_history(student_id)
            
            if chat_history is None:
                print(f"Student with ID {student_id} not found")
                return False
            
            conversation_id = chat_history.get("conversation_id")
            messages = chat_history.get("chat_history", [])
            
            # Create new message
            new_message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            }
            
            # Add new message
            messages.append(new_message)
            
            # Keep only last 3 messages from each role
            user_messages = [msg for msg in messages if msg["role"] == "user"][-3:]
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"][-3:]
            
            # Merge and sort by timestamp to maintain conversation order
            filtered_messages = user_messages + assistant_messages
            filtered_messages.sort(key=lambda x: x["timestamp"])
            
            # Update chat history
            updated_chat_history = {
                "conversation_id": conversation_id,
                "chat_history": filtered_messages
            }
            
            # Update in database
            self.client.table("students").update({
                "chat_history": updated_chat_history
            }).eq("student_id", student_id).execute()
            
            # Handle Google Sheets logging (one row per conversation turn)
            if self.sheets_logger and self.sheets_logger.enabled:
                try:
                    if role == "user":
                        # Store user message - wait for assistant response
                        self._pending_user_message = content
                        self._pending_student_id = student_id
                    elif role == "assistant":
                        # We have assistant response - log complete turn
                        if self._pending_user_message and self._pending_student_id == student_id:
                            self.sheets_logger.logger.log_conversation_turn(
                                student_id, 
                                self._pending_user_message, 
                                content
                            )
                            # Clear pending message
                            self._pending_user_message = None
                            self._pending_student_id = None
                except Exception as e:
                    # Don't fail if Google Sheets logging fails
                    print(f"⚠️  Failed to log to Google Sheets: {e}")
            
            return True
        except Exception as e:
            print(f"Error adding message: {e}")
            return False
    
    def get_or_initialize_academic_details(self, student_id: str):
        """
        Get academic details or initialize if empty
        
        Returns:
            dict: Academic details in format:
                {
                    "completed_courses": [],
                    "earned_credits": 0,
                    "gpa": 0.0
                }
        """
        academic_details = self.get_academic_details(student_id)
        
        if not academic_details or not isinstance(academic_details, dict):
            # Initialize empty structure
            return {
                "completed_courses": [],
                "earned_credits": 0,
                "gpa": 0.0
            }
        
        return academic_details
    
    def update_student_progress(self, student_id: str, completed_courses: list = None, 
                               earned_credits: int = None, gpa: float = None):
        """
        Update student's academic progress (only called from interactive_eligibility_check)
        
        Args:
            student_id: The unique student ID
            completed_courses: List of completed course names
            earned_credits: Total credit hours earned
            gpa: Student's GPA
            
        Returns:
            bool: True if successful
        """
        try:
            academic_details = self.get_or_initialize_academic_details(student_id)
            
            if completed_courses is not None:
                academic_details["completed_courses"] = completed_courses
            
            if earned_credits is not None:
                academic_details["earned_credits"] = earned_credits
            
            if gpa is not None:
                academic_details["gpa"] = gpa
            
            return self.update_academic_details(student_id, academic_details)
        except Exception as e:
            print(f"Error updating student progress: {e}")
            return False
    
    def update_academic_details(self, student_id: str, academic_details: dict):
        """
        Update academic details for a student
        
        Args:
            student_id: The unique student ID
            academic_details: Dictionary with academic information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.table("students").update({
                "academic_details": academic_details
            }).eq("student_id", student_id).execute()
            
            return True
        except Exception as e:
            print(f"Error updating academic details: {e}")
            return False
    
    def clear_chat_history(self, student_id: str):
        """
        Clear chat history for a student (start new conversation)
        
        Args:
            student_id: The unique student ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            new_conversation_id = str(uuid.uuid4())
            self.client.table("students").update({
                "chat_history": {
                    "conversation_id": new_conversation_id,
                    "chat_history": []
                }
            }).eq("student_id", student_id).execute()
            
            return True
        except Exception as e:
            print(f"Error clearing chat history: {e}")
            return False


# Example usage function
def handle_chat_message(student_id: str, user_message: str):
    """
    Example function showing how to handle a chat message
    This is what you'll call from your chatbot
    
    Args:
        student_id: The unique student ID
        user_message: The message from the student
        
    Returns:
        tuple: (chat_history, success)
    """
    connector = ChatbotConnector()
    
    # Get student's existing chat history
    chat_history = connector.get_chat_history(student_id)
    
    if chat_history is None:
        print(f"Student {student_id} not found")
        return None, False
    
    # Add user message to history
    success = connector.add_message(student_id, "user", user_message)
    
    if not success:
        return chat_history, False
    
    # Here you would generate your chatbot response
    bot_response = f"I received your message: '{user_message}'. How can I help you?"
    
    # Add bot response to history
    connector.add_message(student_id, "assistant", bot_response)
    
    # Get updated chat history
    updated_chat_history = connector.get_chat_history(student_id)
    
    return updated_chat_history, True
