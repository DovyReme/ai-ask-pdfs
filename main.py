import dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
import os
import getpass
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from numpy import dot
from dotenv import load_dotenv

dotenv.load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", stop_sequences=["\n\n"])
# llm = ChatOpenAI(model="gpt-4o-mini")

# convert multiple pdf in data folder into text chunks  and store them in vectorstore
docs = []
for pdf in os.listdir("data"):
    pdf_path = f"data/{pdf}"
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())

# split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

question = input("Enter your question: ")
results = rag_chain.invoke({"input": question})

print(results["answer"])
print(results["context"][0].metadata)
