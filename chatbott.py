import streamlit as st
from groq import groq

# Page configuration for a sleek layout
st.set_page_config(
    page_title="Minimalist Chatbot",
    page_icon="💬",
    layout="centered"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    /* Hide top header clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean chat container padding */
    .stMain {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("💬 Groq AI Assistant")
st.caption("Powered by Groq LPU™ Inference Engine")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Secure API Key input
    api_key = st.text_input(
        "Groq API Key", 
        type="password", 
        placeholder="gsk_...",
        help="Get your key at https://console.groq.com/keys"
    )
    
    # Model selection (Exclusively OpenAI Models hosted on Groq)
    model = st.selectbox(
        "Model",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.05,
        help="Lower values make output focused and deterministic; higher values make output more creative."
    )
    
    # Clear conversation button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Chat State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Check if API Key is supplied
    if not api_key:
        st.info("Please enter your Groq API Key in the sidebar to start chatting.", icon="🔑")
        st.stop()
    
    # Append user message to state and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Send completion request with streaming enabled
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                temperature=temperature,
                stream=True
            )

            # Stream chunks directly to UI
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content is not None:
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
            
            # Remove cursor indicator at the end
            response_placeholder.markdown(full_response)
            
            # Save assistant response to state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")
