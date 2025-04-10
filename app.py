import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# API setup
api_key = st.secrets["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=api_key
)

# Page setup
st.set_page_config(page_title="🤖Hey There!!", layout="centered")
st.title("🤖 Welcome to Bot Server!!")
st.markdown("Upload a PDF and start chatting !")

# Session state init
for key in ["messages", "qa_chain", "pdf_uploaded", "pdf_processed", "form_submitted"]:
    if key not in st.session_state:
        st.session_state[key] = False if "messages" != key else []

# Prompt template
relevance_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an intelligent assistant helping users with the Tata Code of Conduct.
Question: "{question}"
Is this question strictly related to the Tata Code of Conduct?
Respond with only "YES" or "NO".
"""
)

def is_query_relevant(question):
    prompt = relevance_prompt.format(question=question)
    response = llm.invoke(prompt)
    return "yes" in response.content.lower()

# PDF Upload
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file and not st.session_state["pdf_uploaded"]:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    st.session_state["pdf_uploaded"] = True
    st.success("📄 PDF uploaded! Click below to process.")

# Manual button to process only once
if st.session_state["pdf_uploaded"] and not st.session_state["pdf_processed"]:
    if st.button("🔍 Process PDF"):
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        st.session_state["pdf_processed"] = True
        st.success("✅ PDF Processed! Ask away below.")

# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Prompt", key="input", placeholder="Ask from the uploaded PDF...", height=70, label_visibility="collapsed")
    submit = st.form_submit_button("Send")

if submit and user_input.strip():
    st.session_state.messages.append({'role': 'user', 'text': user_input})
    st.session_state["form_submitted"] = True

# Handle new message
if st.session_state["form_submitted"]:
    if st.session_state.qa_chain and is_query_relevant(user_input):
        response = st.session_state.qa_chain.invoke(user_input)
        bot_msg = response["result"]
    else:
        bot_msg = "❌ Out of my scope, Please ask relevant questions."

    st.session_state.messages.append({'role': 'bot', 'text': bot_msg})
    st.session_state["form_submitted"] = False  # Reset for next form
    st.rerun()

# Chatbox style

with st.container(border=True, height=400):
    for message in reversed(st.session_state.messages):
        if message['role'] == 'bot':
            st.markdown(
                f"<div style='color:#ffffff; background-color:#000000; padding:5px;margin: 5px 0; display: inline-block; max-width: 80%;border-radius:10px'>🤖 {message['text']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='color:#000000; background-color:#ffffff; padding:5px;margin: 5px 0; display: inline-block; max-width: 80%;border-radius:10px'>👤 {message['text']}</div>",
                unsafe_allow_html=True
            )
            st.divider()
