from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def extract_text_from_document(file):
    pass

def extract_images_from_documents(file):
    pass

def load_text_file(file):
    loader = TextLoader(file, encoding='utf-8')
    document = loader.load()
    print(f"Loaded {len(document)} documents")
    return document

def create_chunks(document):
    splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    documents = splitter.split_documents(document)
    print(f"Splited {len(document)} documents into {len(documents)} chunks")
    return documents

def storing_embeddings(documents, embeddings):
    vector_db = PineconeVectorStore.from_documents(documents, embeddings, index_name='rag-project')
    return vector_db

def generate_questions(llm, vector_store):
    query = "Give 10 important questions to practice regarding the topics provided"
    prompt_template = PromptTemplate(input_variables=[], template=query)
    # vectorstore = PineconeVectorStore(index_name="rag-project", embedding=embeddings)
    prompt = hub.pull('langchain-ai/retrieval-qa-chat')
    combined_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=combined_docs_chain)
    result = retriever_chain.invoke({ "input": query })
    return result
    

if __name__ == "__main__":

    load_dotenv()  

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    document = load_text_file(file)

    chunked_documents = create_chunks(document)

    vectorstore = storing_embeddings(chunked_documents)

    questions = generate_questions(llm, vectorstore)