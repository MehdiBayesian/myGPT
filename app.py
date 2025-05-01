import gradio as gr
import os
from dotenv import load_dotenv
# Import the central LLM client function
from src.llm_client import (
    get_llm_streaming_response, # Streaming
    get_backend_llm_info
)
    
from src.session_manager import (
    list_sessions,
    load_history, # Returns messages format: [{"role": ..., "content": ...}]
    save_history, # Expects messages format
    create_new_session_id,
    ensure_session_dir
)

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
MAX_SESSIONS_DISPLAY = 10

# --- Ensure Session Directory Exists ---
ensure_session_dir()

# --- Helper Functions ---
def get_initial_sessions():
    """Loads the initial list of sessions for the UI Radio choices."""
    # list_sessions returns list of (title, session_id) tuples
    return list_sessions(MAX_SESSIONS_DISPLAY)

def load_selected_session(session_id):
    """Loads history for the selected session_id."""
    if not session_id:
        print("No session ID provided for loading.")
        return [], None # Return empty list (correct format) and None ID
    print(f"Loading session: {session_id}")
    # load_history is expected to return the correct 'messages' format
    history_messages = load_history(session_id)
    # Defensive check (optional but good practice)
    if not isinstance(history_messages, list):
        print(f"Warning: load_history for {session_id} did not return a list. Returning empty list.")
        return [], session_id # Return empty list if format is wrong
    return history_messages, session_id


# --- Core Chat Logic ---
def add_message(session_id, current_history: list, message: str):
    """
    Handles adding messages, getting LLM response, saving, and updating UI.
    Strictly uses the 'messages' format internally for history.

    Args:
        session_id (str | None): The current session ID.
        current_history (list): The chat history in 'messages' format from chat_history_state.
        message (str): The new user message text.

    Yields:
        tuple: Updates for Gradio components.
    """
    if not message or not message.strip():
        # If input is empty, just return current state without changes
        # Refreshing radio might be desired, but let's keep it simple first
        current_sessions = get_initial_sessions()
        radio_update = gr.Radio(choices=current_sessions, value=session_id)
        # Yield current history back to chatbot and state
        yield current_history, current_history, session_id, radio_update
        return # Stop processing

    # --- Ensure `history_messages` is a valid list (using the input `current_history`) ---
    # It should already be in the correct format if loaded/reset correctly.
    # Create a mutable copy to work with if needed, or work directly if state handling is correct.
    history_messages = list(current_history) # Make a mutable copy from state

    new_session_created = False
    if session_id is None:
        session_id = create_new_session_id()
        print(f"Starting new session: {session_id}")
        history_messages = [] # Start with an empty list for the new session
        new_session_created = True

    # --- Append user message in the correct format ---
    history_messages.append({"role": "user", "content": message})
    print(f"DEBUG: Appended user message. History is now: {history_messages}") # Debug print

    # --- First Yield: Show user message ---
    # Yield the updated history_messages list. It MUST be in the correct format here.
    # Use gr.skip() for radio update on new session creation to prevent inconsistency.
    if new_session_created:
        print("DEBUG: First yield (new session) -> skip radio")
        yield history_messages, history_messages, session_id, gr.skip()
    else:
        # Update radio to ensure the current session remains selected
        current_sessions = get_initial_sessions()
        radio_update_same_list = gr.Radio(choices=current_sessions, value=session_id)
        print(f"DEBUG: First yield (existing session {session_id}) -> update radio value")
        yield history_messages, history_messages, session_id, radio_update_same_list

    # --- Get LLM Response with Streaming ---
    # The history_messages list is already in the format needed by llm_client
    print(f"DEBUG: Getting streaming LLM response for history: {history_messages}")
    
    # Initialize assistant's response in history_messages
    history_messages.append({"role": "assistant", "content": ""})
    
    # Use a temporary variable to accumulate streaming chunks
    full_response = ""
    
    # Process each chunk from the streaming response
    for chunk in get_llm_streaming_response(history_messages[:-1]):  # Send history without empty assistant message
        # Accumulate the full response
        full_response += chunk
        
        # Update the assistant's message in history
        history_messages[-1]["content"] = full_response
        
        # Yield the intermediate update
        yield history_messages, history_messages, session_id, gr.skip()
    
    print(f"LLM streaming response complete: {full_response[:100]}...")

    # --- Save the updated history (which is in 'messages' format) ---
    save_history(session_id, history_messages)
    print(f"History saved for session: {session_id}")

    # --- Final Yield: Update session list ---
    # Get the latest session list AFTER saving
    updated_sessions_after_save = get_initial_sessions()
    radio_update_after_save = gr.Radio(choices=updated_sessions_after_save, value=session_id)
    print("DEBUG: Final yield -> update radio list")
    # Yield the final history_messages list. It MUST be in the correct format here.
    yield history_messages, history_messages, session_id, radio_update_after_save


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan"), title="Gradio Chat") as demo:
    # State variables store the current session ID and the chat history in 'messages' format
    current_session_id = gr.State(None)
    chat_history_state = gr.State([]) # Initialize with empty list for messages format

    initial_sessions = get_initial_sessions() # Get initial list [(title, id), ...]

    gr.Markdown("# Gradio Chat Interface")
    with gr.Row():
        # Column 1: Session List
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("## Sessions")
            new_chat_button = gr.Button("➕ New Chat", variant="secondary")
            session_list_display = gr.Radio(
                label="Recent Sessions",
                choices=initial_sessions, # Expects list of (label, value) tuples
                value=None,
                interactive=True,
            )

        # Column 2: Chat Interface
        with gr.Column(scale=4):
            chatbot_display = gr.Chatbot(
                label="Conversation",
                type="messages", # Crucial: Use the 'messages' format
                height=600,
                show_copy_button=True,
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type your message here...",
                    scale=4,
                    show_label=False,
                )
                send_button = gr.Button("Send", variant="primary", scale=1, min_width=80)

    # --- Event Handlers ---

    # 1. Sending a message (Enter or Button)
    send_event = user_input.submit(
        fn=add_message,
        # Pass the current state values and the new message text
        inputs=[current_session_id, chat_history_state, user_input],
        # Expect yields, update chatbot, history state, session ID, and radio list
        outputs=[chatbot_display, chat_history_state, current_session_id, session_list_display],
    ).then(lambda: gr.Textbox(value=""), outputs=[user_input]) # Clear input

    send_button.click(
        fn=add_message,
        inputs=[current_session_id, chat_history_state, user_input],
        outputs=[chatbot_display, chat_history_state, current_session_id, session_list_display],
    ).then(lambda: gr.Textbox(value=""), outputs=[user_input]) # Clear input

    # 2. Selecting a session from the list
    def load_session_and_update_state(session_id_from_radio):
        """Loads history ('messages' format) and updates states."""
        print(f"UI: Radio selected session: {session_id_from_radio}")
        # load_selected_session handles None case and returns messages format
        history, session_id = load_selected_session(session_id_from_radio)
        # Update chatbot display, the history state, and the current session ID state
        return history, history, session_id

    session_list_display.select(
        fn=load_session_and_update_state,
        inputs=[session_list_display], # The value selected in the Radio is passed
        outputs=[chatbot_display, chat_history_state, current_session_id],
    )

    # 3. Starting a new chat
    def new_chat_action():
        """Resets the UI and state for a new chat session."""
        print("UI: Starting new chat action.")
        updated_sessions = get_initial_sessions() # Get fresh list for the radio
        # Reset chatbot display, session ID state, radio selection, history state, and input field
        return [], None, gr.Radio(choices=updated_sessions, value=None, label="Recent Sessions"), [], ""

    new_chat_button.click(
        fn=new_chat_action,
        inputs=None,
        outputs=[
            chatbot_display,         # Set to empty list []
            current_session_id,    # Set to None
            session_list_display,    # Update radio choices and clear selection
            chat_history_state,      # Set to empty list []
            user_input               # Set to empty string ""
        ],
    )

# --- Launch the Application ---
if __name__ == "__main__":
    # load .env file to setup LLM backend.
    # display information 
    print(get_backend_llm_info())

    ensure_session_dir()
    demo.launch(debug=True) # Launch in debug mode to see more details in console if errors occur