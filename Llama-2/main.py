from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory


def create_prompt() -> str:
    """
    Generates prompt template

    :param: Takes in no parameters
    :return: a prompt template
    """
    # Prompt obtained from langchain docs
    _DEFAULT_TEMPLATE = """Assistant is a large language model trained by Meta.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    {history}
    Human: {human_input}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["history", "human_input"], template=_DEFAULT_TEMPLATE)

    return prompt


def load_model() -> CTransformers:
    """
    This function is used to load the LLM(Llama)

    :param: Takes in no prompt template
    :return: CTranformer object, the Llama model
    """
    MODEL_PATH = "/home/prince/Downloads/llama-2-7b-chat.ggmlv3.q8_0.bin"

    Llama_llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_token=512,
        temperature=0.3
    )

    return Llama_llm


def create_llm_chain() -> LLMChain:
    """
    Create an LLMChain and return to caller

    :params: Takes in no parameters
    :return: LLChain object
    """

    llm = load_model()
    prompt = create_prompt()
    memory = ConversationBufferWindowMemory(k=6, return_messages=True)

    llm_chain: LLMChain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    return llm_chain

print("LLMChain created")
llm_chain: LLMChain = create_llm_chain()

print("Making prediction")
response: str = llm_chain.predict(
    human_input="What is the largest country on earth")

print(response)
