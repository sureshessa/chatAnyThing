
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.language_models.chat_models import LangSmithParams


#Load pdf documets from data loader
def get_PDF_data():
    loader=DirectoryLoader('./data',
                           glob="**/*.pdf",
                           loader_cls=PyPDFLoader)
    docs=loader.load()
    return docs

#Split Text into Chunks
def get_textchunks(documents):
    text_splitter=RecursiveCharacterTextSplitter(
                                                chunk_size=1000,
                                                chunk_overlap=200)
    text_chunks=text_splitter.split_documents(documents)
    return text_chunks

#Convert text chuncks into embeddings and store them into Chroma db
def get_vector_store(text_chunks):
    persist_directory = 'db'

    # create the embedding function
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

    vectordb = Chroma.from_documents(documents=text_chunks,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None

    vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)
    return vectordb


def get_RAG_Prompt_template():
    template = (
        'Answer the following question as an Expert in the given ' +
        'context, If you are unable to find an answer to the question ' +
        'in the given context or if there is no context respond with ' +
        '"Not found in the context!"\nProvide step by step answer(s) ' +
        'You must include relevant information from given documets.\n' +
        'History:{chat_history}'+
        'Question: {question}\n\n' +
        'Context: {context}'
        )

    prompt = PromptTemplate(
                input_variables=["chat_history", "question","context"], template=template
                )

    return prompt

def get_chain(llm, retriever):
    # create a conversation buffer memory
    memory = ConversationBufferMemory(memory_key='chat_history',input_key="question")  
    prompt = get_RAG_Prompt_template() 

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    verbose=False,
    chain_type_kwargs={
        "verbose": False,
        "prompt": prompt,
        "memory": memory
    } )
    
    return qa_chain

