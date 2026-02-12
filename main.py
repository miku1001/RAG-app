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



#grab api key
load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
  raise ValueError('No API key')

#load model
model = init_chat_model("google_genai:gemini-2.5-flash")

#document loader
loader = PyPDFLoader("files\\phil_const.pdf")
documents= loader.load()

#text_split and chunking
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000, 
  chunk_overlap=200, 
  separators=["\n\n", "\n", " ", ""]
  )

chunks = text_splitter.split_documents(documents)

#embedding and store
embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2",
  encode_kwargs={'normalize_embeddings': True}
  )

store = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = store.as_retriever()

#prompt template
prompt = ChatPromptTemplate.from_messages([
   ("system", """You are a legal assistant.
      Answer the question strictly using ONLY the provided context.
      If the answer is not found in the context, say:
      "The answer is not found in the provided document."
      Be accurate and concise."""),
    ("user", "Question: {question}\nContext: {context}")
])

question = input('Enter question: ')

docs = retriever.invoke(question)
docs_content = "\n\n".join(doc.page_content for doc in docs)

chain = prompt | model |StrOutputParser()

print(chain.invoke({
   "question":question,
   "context": docs_content
}))




