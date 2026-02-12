import streamlit as st
#-----
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


st.set_page_config(page_title="RAG Web Demo", layout="wide")
st.title("Simple RAG Demo")

#grab api key
load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY2')
if not api_key:
  st.error('No API key')
  st.stop()

if 'store' not in st.session_state:
  st.session_state.store = None

#upload file
uploaded_file = st.file_uploader(
  "Choose a file", type="pdf"
)

if st.button("Initialize RAG System"):
  with st.spinner("Loading and processing docs..."):
    if uploaded_file is not None:
      # Clear old vector store first
      if st.session_state.store is not None:
        st.session_state.store.delete_collection()
        st.session_state.store = None
      
      # Clear old responses
      if 'last_response' in st.session_state:
        del st.session_state.last_response
      if 'last_context' in st.session_state:
        del st.session_state.last_context
      
      with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
      #document loader
      loader = PyPDFLoader("temp.pdf")
      documents= loader.load()

      #text_split and chunking
      text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=100, 
        separators=["\n\n", "\n", " ", ""]
        )

      chunks = text_splitter.split_documents(documents)

      #embedding and store
      embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': True}
        )

      st.session_state.store = Chroma.from_documents(documents=chunks, embedding=embeddings)
      st.success("RAG initialized Succesfully!")

      os.remove('temp.pdf')
    else:
      st.warning("Please upload PDF File")

if st.session_state.store is not None:
  #load model
  model = init_chat_model("google_genai:gemini-2.5-flash")

  #prompt template
  prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer questions ONLY using the provided context. If the answer isn't in the context or the question asks for tasks like counting, coding, or creative writing, respond: 'I don't know'. Never use external knowledge."),
      ("user", "Context:\n{context}\n\nQuestion: {question}")
  ])

  chain = prompt | model |StrOutputParser()

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Ask a Question")
    question =st.text_area("Enter your Question:")

    if st.button("Get Answer"):
      if question:
        with st.spinner("Generating answer.."):
          retriever = st.session_state.store.as_retriever()
          docs = retriever.invoke(question)
          docs_content = "\n\n".join(doc.page_content for doc in docs)

          response = chain.invoke({
                "question":question,
                "context": docs_content
              })
          
          st.session_state.last_response = response
          st.session_state.last_context = docs
      else:
        st.warning("Please input a question")
    
    with col2:
      st.subheader("Answer will be displayed here!")
      if 'last_response' in st.session_state:
        st.write(st.session_state.last_response)

        # with st.expander("Show Retrieved Content"):
        #   for i, doc in enumerate(st.session_state.last_context, 1):
        #     st.markdown(f"**Document {i}**")
        #     st.markdown(doc.page_content)
        #     st.markdown("---")




      
      

