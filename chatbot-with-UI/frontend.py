from langchain_core.messages import HumanMessage
import streamlit as st
from backend import app

CONFIG = {'configurable': {'thread_id':'thread_1'}}

# kind of storage 
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# previouse conversion
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])



user_input = st.chat_input('Type here')

# current conversion
if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
   
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
                message_chunk.content for message_chunk,metadata in app.stream(
                        {'messages':[HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode='messages'
                    )
                )  

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})