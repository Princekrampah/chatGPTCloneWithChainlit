from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl


FAISS_DB_PATH = "vectorstores/faiss_db"
MODEL_PATH = "/home/prince/Downloads/llama-2-7b-chat.ggmlv3.q8_0.bin"

prompt_template_str: str = """Use the provided information to answer the user questions.\
if you do not know the answer, kindly inform the user you do not know the answer and avoid\ 
making up your own answers

Context: {context}
Question: {question}

only return the most correct answer below and do not include any other text
Correct answer here:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template_str)

    return prompt_template


def load_llm_model():
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_token=512,
        temperature=0.3
    )

    return llm


def retrieval_chain(llm, prompt, db_store):
    """
    Retrieval chain function
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db_store.as_retriever(
            search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


def qa_chatbot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    db_store = FAISS.load_local(
        folder_path=FAISS_DB_PATH, embeddings=embeddings)
    llm = load_llm_model()
    prompt = set_custom_prompt()
    qa = retrieval_chain(llm=llm, prompt=prompt, db_store=db_store)

    return qa


def final_result(query):
    result = qa_chatbot()
    response = result({"query": query})
    return response

# Chainlit

#  # Runs when the app is started


@cl.on_chat_start
async def start():
    chain = qa_chatbot()
    msg = cl.Message(content="Loading, Please wait...")
    await msg.send()
    # display this once the page loads successfully.
    msg.content = "Hello there, welcome to Book Reviewer"
    # update the loading msg
    await msg.update()
    cl.user_session.set("chain", chain)


# # Runs on message(user input)
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    # make an async call(acall)
    response = await chain.acall(message, callbacks=[cb])
    answer = response["result"]
    source_docs = response["source_documents"]

    if source_docs:
        answer += f"\nReferences: {str(source_docs)}"
    else:
        answer += f"\nNo source found"

    await cl.Message(content=answer).send()
