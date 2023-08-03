from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# file paths and data store paths
DATA_PATH = "data/"
FAISS_DB_PATH = "vectorstores/faiss_db"


# create store
def create_vector_store():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # split texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=10)
    texts = text_splitter.split_documents(documents=documents)

    # create embeddings: with cuda, you can use cpu if you want
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})

    # create store
    db_store = FAISS.from_documents(embedding=embeddings, documents=texts)

    # create persistant version locally
    db_store.save_local(FAISS_DB_PATH)


if __name__ == "__main__":
    create_vector_store()
