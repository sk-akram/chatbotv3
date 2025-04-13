# API setup
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- API setup ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.5
)

# --- Page setup ---
st.set_page_config(page_title="🤖 with 🧠", layout="centered")
st.title("🤖 Welcome to Sk's Bot Server !!")
st.markdown("Upload a PDF and start chatting..")

# --- Session State Init ---
for key in ["messages", "conversation_chain", "pdf_uploaded", "pdf_processed"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else False

# --- Prompt Template ---
answer_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an intelligent assistant helping users with document-based queries.

Context:
{context}

User's Question: {question}

Based on the context above, provide a relevant, factual, and concise answer. If no relevant information is available, respond with: ❌ Out of scope. Please ask a question related to the content of the document.
"""
)

# --- PDF Upload ---
pdf_file = st.file_uploader("📄 Upload PDF", type="pdf")
if pdf_file and not st.session_state["pdf_uploaded"]:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    st.session_state["pdf_uploaded"] = True
    st.success("📄 PDF uploaded! Now process it.")

# --- Process Button ---
if st.session_state["pdf_uploaded"] and not st.session_state["pdf_processed"]:
    if st.button("🔍 Process PDF"):
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": answer_prompt}
        )
        st.session_state["pdf_processed"] = True
        st.success("✅ PDF processed! Ask your questions below.")

# --- Chat UI ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Ask something...", height=70, label_visibility="collapsed")
    send = st.form_submit_button("Send")

if send and user_input.strip():
    chain = st.session_state.conversation_chain
    if chain:
        result = chain.invoke({"question": user_input})
        bot_reply = result["answer"]

        st.session_state.messages.append({"role": "user", "text": user_input})
        st.session_state.messages.append({"role": "bot", "text": bot_reply})

        st.rerun()

# --- Chat Display ---
with st.container(border=True, height=400):
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "bot":
            st.markdown(f"<div style='color:white; background:#000; padding:8px; border-radius:10px;'>🤖 {msg['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:black; background:#eee; padding:8px; border-radius:10px;'>👤 {msg['text']}</div>", unsafe_allow_html=True)
            st.divider()
