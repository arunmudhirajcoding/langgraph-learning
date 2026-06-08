from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import streamlit as st
from backend_with_tools import app, retrieve_all_threads
import uuid
import os

os.environ['LANGSMITH_PROJECT'] = 'langchain-chatbot'

# ----------------------------------------------utility functions------------------------------------------
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat(): # for new conversation 
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = [{'role':'assistant','content':app.invoke({'messages':'intro your self with your name in 5 words'}, {'configurable': {'thread_id':str(thread_id)}})['messages'][-1].content}]

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return app.get_state(config={'configurable':{'thread_id': thread_id}}).values['messages']

#  -------------------------------------------session store-----------------------------------------------
# kind of storage 
if 'message_history' not in st.session_state: # create message history
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state: # genrerate thread 
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state: # create chat threads
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

# CONFIG = {'configurable': {'thread_id':st.session_state['thread_id']}}
CONFIG = {
    'configurable': {'thread_id':st.session_state['thread_id']},
    "metadata": {
        "thread_id": st.session_state['thread_id'],
        "run_name": "chat_turn"
    }
}
# ---------------------------------------- ui ----------------------------------------
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(message,HumanMessage):
                role = 'user'
            else: 
                role = 'assistant'
            temp_messages.append({'role':role, 'content': message.content})
        st.session_state['message_history'] = temp_messages


# previouse conversion
for message in st.session_state['message_history']:
    # remove first user query from message history
    if message['role'] == 'user' and message['content'] == 'intro your self with your name in 5 words':
        continue
    with st.chat_message(message['role']):
        st.text(message['content'])



user_input = st.chat_input('Type here')

# current conversion
if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
    # this below code will be print tool call chat also as a stream .     
    # with st.chat_message('assistant'):
    #     # ---------- streaming chat -----------------------------------
    #     ai_message = st.write_stream(
    #             message_chunk.content for message_chunk,metadata in app.stream(
    #                     {'messages':[HumanMessage(content=user_input)]},
    #                     config=CONFIG,
    #                     stream_mode='messages'
    #                 )
    #             )  

    with st.chat_message('assistant'):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in app.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = CONFIG,
                stream_mode = "messages" 
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk,AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

         # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )
            
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})