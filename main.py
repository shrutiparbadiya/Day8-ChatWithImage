from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool

##############################
### initialize agent #########
##############################
tools = [ImageCaptionTool, ObjectDetectionTool]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k = 5,
    return_mesagges= True
)

llm = ChatOpenAI(
    openai_api_key = None,
    temperature = 0,
    model_name = 'gpt-3.5-turbo'
)

agent = initialize_agent(
    agent = "chat-conversational-react-description",
    tools = tools,
    llm = llm,
    max_iterations = 5,
    verbose = True,
    memory = conversational_memory,
    early_stoppy_method = 'generate'
)
# set title

st.title("Ask a question to an Image.")

# set header

st.header('Please upload an image')

# upload file

file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if file:
    # display image
    st.image(file, use_column_width=True)

    # text input
    user_question = st.text_input("Ask a question about your image.")

    ##############################
    ### Compute Agent Response ###
    ##############################
    with NamedTemporaryFile(dir = '.') as f:
        f.write(file.getbuffer())
        image_path = f.name
    response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
    # write agent response
    if user_question and user_question != "":
        st.write("Dummy response")