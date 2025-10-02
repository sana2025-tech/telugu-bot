import streamlit as st
from transformers import pipeline

# Load a text-generation pipeline (you can replace gpt2 with any Hugging Face model)
chatbot = pipeline("text-generation", model="gpt2")

st.title("Telugu Learning Chatbot")

# Keep chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You: ", "")

if st.button("Send") and user_input:
    # Generate response
    response = chatbot(user_input, max_length=80, do_sample=True, top_k=50, top_p=0.95)
    bot_reply = response[0]['generated_text']

    # Save history
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", bot_reply))

# Display conversation
for role, msg in st.session_state.history:
    if role == "You":
        st.markdown(f"**{role}:** {msg}")
    else:
        st.markdown(f"<span style='color:green'>**{role}:** {msg}</span>", unsafe_allow_html=True)
