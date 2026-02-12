import os
from dotenv import load_dotenv

#grab api key
load_dotenv()

api_key = os.environ.get('GEMINI_API_KEY')

if not api_key:
  raise ValueError('No API key')

#load model
from langchain.chat_models import init_chat_model
model = init_chat_model("google_genai:gemini-2.5-flash")


#document loader
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('files\phil_const.pdf')
documents= loader.load()


#text_split and chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000, 
  chunk_overlap=200, 
  separators=["\n\n", "\n", " ", ""]
  )

chunks = text_splitter.split_documents(documents)

#embedding and store
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-mpnet-base-v2"
  )

store = Chroma.from_documents(documents=chunks, embedding=embeddings)





 
#prompt template
from langchain_core.prompts import PromptTemplate

context = "Manila is the capital of philippines"
prompt = PromptTemplate(
          template="""Answer the question based only on the following context: {context}. 
          Reject questions unrelated to the context: {context}.
          Note: Be accurate and concise. Don't create your own question. Just answer the user's chat.

          Question: {question}
          """)

chain =  prompt | model 



