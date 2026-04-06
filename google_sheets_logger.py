"""
Google Sheets Logger for Academic Advisor Chatbot

Logs all user and AI conversations to a Google Sheet using Service Account API.
All students' conversations go into one shared sheet.

NO OAUTH - Just uses a service account JSON file!
"""

import os
from datetime import datetime
from typing import Optional
import gspread
from google.oauth2.service_account import Credentials


class GoogleSheetsLogger:
    """Logs all conversations to Google Sheets using Service Account API."""
    
    # Scopes required for Google Sheets API
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self, spreadsheet_id: str = None, credentials_file: str = 'service-account.json'):
        """
        Initialize Google Sheets logger with Service Account.
        
        Args:
            spreadsheet_id: Google Sheet ID (from URL)
            credentials_file: Path to service account JSON file
        """
        self.spreadsheet_id = spreadsheet_id or os.getenv('GOOGLE_SHEET_ID')
        self.credentials_file = credentials_file
        self.client = None
        self.sheet = None
        
        if not self.spreadsheet_id:
            raise ValueError("GOOGLE_SHEET_ID must be set in .env or passed as parameter")
        
        if not os.path.exists(self.credentials_file):
            raise FileNotFoundError(
                f"Service account file '{self.credentials_file}' not found. "
                "Download it from Google Cloud Console."
            )
        
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google using Service Account."""
        try:
            # Load credentials from service account file
            creds = Credentials.from_service_account_file(
                self.credentials_file,
                scopes=self.SCOPES
            )
            
            # Create gspread client
            self.client = gspread.authorize(creds)
            
            # Open the spreadsheet
            self.sheet = self.client.open_by_key(self.spreadsheet_id).sheet1
            
        except Exception as e:
            raise Exception(f"Failed to authenticate with Google Sheets: {e}")
    
    def initialize_sheet(self):
        """
        Initialize the Google Sheet with headers if empty.
        Creates columns: Timestamp, Student ID, User Message, Assistant Message
        """
        try:
            # Check if sheet has headers
            first_row = self.sheet.row_values(1)
            
            # If no headers, add them
            if not first_row or first_row[0] != 'Timestamp':
                headers = ['Timestamp', 'Student ID', 'User Message', 'Assistant Message']
                self.sheet.insert_row(headers, 1)
                print("✓ Google Sheet initialized with headers")
            
            return True
            
        except Exception as error:
            print(f"Error initializing sheet: {error}")
            return False
    
    def log_message(self, student_id: str, role: str, message: str) -> bool:
        """
        DEPRECATED: Use log_conversation_turn instead.
        This method is kept for backwards compatibility but does nothing.
        
        Args:
            student_id: The student's ID
            role: Either 'user' or 'assistant'
            message: The message content
            
        Returns:
            bool: Always True (no-op)
        """
        # This method is deprecated - we now log complete turns only
        return True
    
    def log_conversation_turn(self, student_id: str, user_message: str, ai_response: str) -> bool:
        """
        Log a complete conversation turn (user + assistant) as ONE ROW.
        
        Args:
            student_id: The student's ID
            user_message: The user's message
            ai_response: The AI's response
            
        Returns:
            bool: True if logged successfully
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Prepare single row with both messages
            row = [timestamp, student_id, user_message, ai_response]
            
            # Append to sheet
            self.sheet.append_row(row, value_input_option='RAW')
            
            return True
            
        except Exception as error:
            print(f"Error logging conversation: {error}")
            return False
    
    def get_recent_conversations(self, limit: int = 100) -> list:
        """
        Get recent conversations from the sheet.
        
        Args:
            limit: Maximum number of rows to retrieve
            
        Returns:
            list: List of conversation dictionaries
        """
        try:
            # Get all values (skip header)
            all_values = self.sheet.get_all_values()[1:]
            
            # Take last N rows
            recent_values = all_values[-limit:] if len(all_values) > limit else all_values
            
            conversations = []
            for row in recent_values:
                if len(row) >= 4:
                    conversations.append({
                        'timestamp': row[0],
                        'student_id': row[1],
                        'role': row[2],
                        'message': row[3]
                    })
            
            return conversations
            
        except Exception as error:
            print(f"Error retrieving conversations: {error}")
            return []
    
    def get_student_conversations(self, student_id: str) -> list:
        """
        Get all conversations for a specific student.
        
        Args:
            student_id: The student's ID
            
        Returns:
            list: List of that student's conversations
        """
        all_conversations = self.get_recent_conversations(limit=10000)
        
        return [
            conv for conv in all_conversations
            if conv['student_id'] == student_id
        ]
    
    def clear_all_data(self):
        """
        Clear all data except headers (USE WITH CAUTION!)
        """
        try:
            # Get all values
            all_values = self.sheet.get_all_values()
            
            if len(all_values) > 1:
                # Delete all rows except header
                self.sheet.delete_rows(2, len(all_values))
                print("✓ All data cleared (headers preserved)")
                return True
            
            return True
            
        except Exception as error:
            print(f"Error clearing data: {error}")
            return False


# Integration wrapper for easy use with main.py
class ChatLogger:
    """Simple wrapper for logging chats to Google Sheets."""
    
    def __init__(self):
        """Initialize the logger if credentials are available."""
        self.logger = None
        self.enabled = False
        
        try:
            # Check if Google Sheets is configured
            sheet_id = os.getenv('GOOGLE_SHEET_ID')
            
            if sheet_id and os.path.exists('service-account.json'):
                self.logger = GoogleSheetsLogger(sheet_id)
                self.logger.initialize_sheet()
                self.enabled = True
                print("✓ Google Sheets logging enabled")
            else:
                if not sheet_id:
                    print("ℹ️  GOOGLE_SHEET_ID not set - Google Sheets logging disabled")
                elif not os.path.exists('service-account.json'):
                    print("ℹ️  service-account.json not found - Google Sheets logging disabled")
                
        except Exception as e:
            print(f"⚠️  Google Sheets logging unavailable: {e}")
            self.enabled = False
    
    def log(self, student_id: str, user_message: str, ai_response: str):
        """Log a conversation turn (if logging is enabled)."""
        if self.enabled and self.logger:
            try:
                self.logger.log_conversation_turn(student_id, user_message, ai_response)
            except:
                pass  # Silently fail - don't break the chatbot
