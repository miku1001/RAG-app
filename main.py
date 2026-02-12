import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


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
  model_name="sentence-transformers/all-MiniLM-L6-v2"
  )

store = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = store.as_retriever()

#prompt template
prompt = PromptTemplate(
          template="""Answer the question based only on the following context: {context}. 
          Reject questions unrelated to the context: {context}.
          Note: Be accurate and concise. Just answer the user's chat.

          Question: {question}
          """)

question = input('Enter question: ')

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = {"context": retriever | format_docs, "question": RunnablePassthrough() }| prompt | model

print(chain.invoke(question).content)




