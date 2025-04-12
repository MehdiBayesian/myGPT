# src/session_manager.py
import os
import json
from datetime import datetime, timezone

SESSION_DIR = "sessions_history"
MAX_SESSIONS_DISPLAY = 10

def ensure_session_dir():
    """Ensures the session directory exists."""
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)

def get_session_filepath(session_id: str) -> str:
    """Constructs the full path for a session file."""
    session_id = os.path.basename(session_id) # Basic sanitization
    return os.path.join(SESSION_DIR, f"{session_id}.json")


def _get_session_count() -> int:
    """Counts the number of existing session files."""
    ensure_session_dir()
    try:
        files = [f for f in os.listdir(SESSION_DIR) if f.endswith(".json")]
        return len(files)
    except FileNotFoundError:
        return 0

def _generate_default_title() -> str:
    """Generates a default title like Session_1, Session_2, etc."""
    count = _get_session_count()
    return f"Session_{count + 1}" # Next available index

def list_sessions(limit: int = MAX_SESSIONS_DISPLAY) -> list[str]:
    """
    Lists the most recent session IDs (filenames without extension).
    Sorts by modification time (most recent first).
    """
    ensure_session_dir()
    try:
        files = [f for f in os.listdir(SESSION_DIR) if f.endswith(".json")]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(SESSION_DIR, f)), reverse=True)
        return [os.path.splitext(f)[0] for f in files[:limit]]
    except FileNotFoundError:
        return []


def create_new_session_id() -> str:
    """Generates a new session ID based on the current timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3] + "_UTC"


def load_history(session_id: str) -> list:
    """
    Loads chat history from a session file.
    Assumes the 'memory' field directly contains the list in 'messages' format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    Returns empty list if session doesn't exist or format is invalid.
    """
    ensure_session_dir()
    filepath = get_session_filepath(session_id)
    if not os.path.exists(filepath):
        print(f"Session file not found: {filepath}")
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        # --- Validation ---
        if not isinstance(session_data, dict) or "memory" not in session_data:
            print(f"Warning: Session file {filepath} missing 'memory' field or not a dictionary.")
            return []
        if not isinstance(session_data["memory"], list):
            print(f"Warning: Session file {filepath} 'memory' field is not a list.")
            return []

        # Directly return the 'memory' list, assuming it's in the correct format
        history_messages = session_data["memory"]

        # Optional: Add validation for each item in the list
        for i, item in enumerate(history_messages):
            if not isinstance(item, dict) or "role" not in item or "content" not in item:
                print(f"Warning: Invalid message format at index {i} in {filepath}: {item}. Returning partial history.")
                return history_messages[:i] # Return history up to the invalid item

        # print(f"Successfully loaded history (messages format) from {session_id}")
        return history_messages

    except (json.JSONDecodeError, IOError, TypeError) as e:
        print(f"Error loading session {session_id} from {filepath}: {e}")
        return []





def save_history(session_id: str, history_messages: list):
    """
    Saves chat history directly in the Gradio 'messages' format to a session file.
    The input `history_messages` list is saved directly into the 'memory' field.
    Manages 'created_at', 'updated_at', and 'title' fields.
    """
    ensure_session_dir()
    filepath = get_session_filepath(session_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    is_new_session = not os.path.exists(filepath)

    # --- Validation (Optional but recommended) ---
    if not isinstance(history_messages, list):
        print(f"Error saving session {session_id}: history_messages is not a list.")
        return
    for i, item in enumerate(history_messages):
        if not isinstance(item, dict) or "role" not in item or "content" not in item:
            print(f"Error saving session {session_id}: Invalid message format at index {i}: {item}")
            return
    # --- End Validation ---

    # 1. Prepare session data structure
    session_data = {
        "session_id": session_id,
        "title": None, # Placeholder
        "created_at": now_iso,
        "updated_at": now_iso,
        # Save the history_messages list directly into the 'memory' field
        "memory": history_messages
    }

    # 2. Determine title (using the new format)
    generated_title = f"Chat Session {session_id}" # Default fallback
    if history_messages and history_messages[0].get("role") == "user":
        first_content = history_messages[0].get("content", "")
        if isinstance(first_content, str): # Ensure content is string
             generated_title = f"{first_content[:30]}..." if first_content else generated_title
    elif history_messages: # If first message isn't user, use generic title
        generated_title = f"Chat ({history_messages[0].get('role', 'System')})"


    # 3. Handle metadata based on existence
    if is_new_session:
        session_data["title"] = generated_title if generated_title else _generate_default_title()
        print(f"Creating new session '{session_data['title']}' ({session_id})")
    else:
        # Try to load existing data only to preserve original title and created_at
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, dict):
                # Use existing title unless missing, then use generated
                session_data["title"] = existing_data.get("title") or generated_title
                session_data["created_at"] = existing_data.get("created_at", now_iso) # Preserve original creation time
            else:
                 print(f"Warning: Existing file {filepath} was not a dict. Resetting metadata.")
                 session_data["title"] = generated_title
                 # created_at already set to now_iso
        except (json.JSONDecodeError, IOError, KeyError, FileNotFoundError) as e:
             print(f"Warning: Could not read existing {filepath} for metadata: {e}. Using generated/defaults.")
             session_data["title"] = generated_title
             # created_at already set to now_iso

    # 4. Save the data
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Dump the session_data which now contains 'memory' in the correct format
            json.dump(session_data, f, ensure_ascii=False, indent=4)
        # print(f"Session {session_id} saved successfully (messages format).")
    except IOError as e:
        print(f"Error saving session {session_id} to {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during save for session {session_id}: {e}")

