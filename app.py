import os
import argparse
import gradio as gr
from dotenv import load_dotenv
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

# --- Configuration ---
ADD_COPYRIGHT = True

# --- Constants ---
MAX_SESSIONS_DISPLAY = 10

# --- Global stop signal variable ---
# This is our "emergency brake" that can be accessed from any function
STOP_STREAMING = False

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
    global STOP_STREAMING
    # Reset stop signal at the beginning
    STOP_STREAMING = False
    
    if not message or not message.strip():
        # If input is empty, just return current state without changes
        current_sessions = get_initial_sessions()
        radio_update = gr.Radio(choices=current_sessions, value=session_id)
        # Yield current history back to chatbot and state
        yield current_history, current_history, session_id, radio_update, False, gr.Button(visible=True), gr.Button(visible=False), gr.Textbox(interactive=True)
        return # Stop processing

    # --- Ensure `history_messages` is a valid list (using the input `current_history`) ---
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

    # --- First Yield: Show user message and update button visibility ---
    if new_session_created:
        print("DEBUG: First yield (new session) -> skip radio")
        # Set is_streaming to True and update button visibility
        yield history_messages, history_messages, session_id, gr.skip(), True, gr.Button(visible=False), gr.Button(visible=True), gr.Textbox(interactive=False)
    else:
        # Update radio to ensure the current session remains selected
        current_sessions = get_initial_sessions()
        radio_update_same_list = gr.Radio(choices=current_sessions, value=session_id)
        print(f"DEBUG: First yield (existing session {session_id}) -> update radio value")
        # Set is_streaming to True and update button visibility
        yield history_messages, history_messages, session_id, radio_update_same_list, True, gr.Button(visible=False), gr.Button(visible=True), gr.Textbox(interactive=False)

    # --- Get LLM Response with Streaming ---
    print(f"DEBUG: Getting streaming LLM response for history: {history_messages}")
    
    # Initialize assistant's response in history_messages
    history_messages.append({"role": "assistant", "content": ""})
    
    # Use a temporary variable to accumulate streaming chunks
    full_response = ""
    
    # Process each chunk from the streaming response
    was_stopped = False
    try:
        for chunk in get_llm_streaming_response(history_messages[:-1]):  # Send history without empty assistant message
            # Check if stop signal is active - using global variable
            if STOP_STREAMING:
                print("Streaming stopped by user")
                was_stopped = True
                break
                
            # Accumulate the full response
            full_response += chunk
            
            # Update the assistant's message in history
            history_messages[-1]["content"] = full_response
            
            # Yield the intermediate update (keep button visibility during streaming)
            yield history_messages, history_messages, session_id, gr.skip(), True, gr.Button(visible=False), gr.Button(visible=True), gr.Textbox(interactive=False)
    
    except Exception as e:
        # Handle any exceptions during streaming
        print(f"Error during streaming: {e}")
        if not full_response:
            full_response = f"Sorry, an error occurred: {str(e)}"
            history_messages[-1]["content"] = full_response
    
    # Add message indicating if the response was stopped
    if was_stopped:
        full_response += "\n\n[Response was stopped early]"
        history_messages[-1]["content"] = full_response
    
    print(f"LLM streaming response complete: {full_response[:100]}...")

    # --- Save the updated history (which is in 'messages' format) ---
    save_history(session_id, history_messages)
    print(f"History saved for session: {session_id}")

    # --- Final Yield: Update session list and reset button visibility ---
    updated_sessions_after_save = get_initial_sessions()
    radio_update_after_save = gr.Radio(choices=updated_sessions_after_save, value=session_id)
    print("DEBUG: Final yield -> update radio list")
    # Yield the final history_messages list and update button visibility
    # IMPORTANT: Set input field to interactive=True at the end
    yield history_messages, history_messages, session_id, radio_update_after_save, False, gr.Button(visible=True), gr.Button(visible=False), gr.Textbox(interactive=True)


# --- Gradio Interface Definition ---

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan"), title="Gradio Chat") as demo:
    # State variables store the current session ID and the chat history in 'messages' format
    current_session_id = gr.State(None)
    chat_history_state = gr.State([]) # Initialize with empty list for messages format
    is_streaming = gr.State(False)    # Added state to track if currently streaming

    initial_sessions = get_initial_sessions() # Get initial list [(title, id), ...]

    gr.Markdown("# A Humble GPT")
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
                # Make the stop button more noticeable with red color
                stop_button = gr.Button("Stop", variant="stop", scale=1, min_width=80, visible=False, elem_classes="stop-button")

    # Add some CSS to make the stop button more noticeable
    gr.Markdown("""
    <style>
    .stop-button {
        background-color: #FF5252 !important;
        color: white !important;
        font-weight: bold !important;
    }
    </style>
    """)

    # --- Event Handlers ---

    # 1. Sending a message (Enter or Button)
    send_event = user_input.submit(
        fn=add_message,
        # Pass the current state values
        inputs=[current_session_id, chat_history_state, user_input],
        # Update all components including button visibility directly
        outputs=[chatbot_display, chat_history_state, current_session_id, session_list_display, is_streaming, send_button, stop_button, user_input]
    ).then(
        lambda: gr.Textbox(value=""), 
        outputs=[user_input]  # Clear input
    )

    send_button.click(
        fn=add_message,
        inputs=[current_session_id, chat_history_state, user_input],
        # Update all components including button visibility directly
        outputs=[chatbot_display, chat_history_state, current_session_id, session_list_display, is_streaming, send_button, stop_button, user_input]
    ).then(
        lambda: gr.Textbox(value=""), 
        outputs=[user_input]  # Clear input
    )

    # Stop button handler - set stop signal to True using a global variable
    def stop_streaming():
        global STOP_STREAMING
        STOP_STREAMING = True
        print("Stop button clicked! Setting global STOP_STREAMING to True.")
        # Re-enable input field immediately
        return gr.Textbox(interactive=True)

    stop_button.click(
        fn=stop_streaming,
        inputs=None,
        outputs=[user_input]
    )

    # React to streaming state changes
    is_streaming.change(
        fn=lambda s: gr.Textbox(interactive=not s),
        inputs=[is_streaming],
        outputs=[user_input]
    )

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
            current_session_id,      # Set to None
            session_list_display,    # Update radio choices and clear selection
            chat_history_state,      # Set to empty list []
            user_input               # Set to empty string ""
        ],
    )

    if ADD_COPYRIGHT is True:
        gr.HTML(f"""
            <div style="text-align: center; margin-top: 20px; padding: 10px;">
                © 2025 Impartial GradientZ, LLC. All rights reserved.
            </div>
        """)
# --- Launch the Application ---
if __name__ == "__main__":
    # load .env file to setup LLM backend.
    # display information 
    print(get_backend_llm_info())
    ensure_session_dir()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="myGPT App using local LLM models.")
    parser.add_argument("--share", action="store_true", help="Share the app publicly via Gradio https link.", default=False)
    args = parser.parse_args()
    
    
    demo.launch(debug=True, share=args.share) # Launch in debug mode to see more details in console if errors occur
