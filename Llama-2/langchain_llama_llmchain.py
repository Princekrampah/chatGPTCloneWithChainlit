from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_PATH = "/home/prince/Downloads/llama-2-7b-chat.ggmlv3.q8_0.bin"

# TODO:
# install necesarry libraries. I already have langchain
# installed, make sure you are running the latest
# version of python 3.8.1+. I have tried with the CPU only and the GPU
# 1. Create a function to generate prompt
# 2. Create a function to load Llama-2


def create_prompt() -> PromptTemplate:
    """
    Generates prompt template

    :param: Takes in no parameters
    :return: a prompt template
    """
    # Prompt obtained from langchain docs
    _DEFAULT_TEMPLATE: str = """Assistant is a large language model trained by Meta.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    Human: {human_input}
    Assistant:"""

    prompt: PromptTemplate = PromptTemplate(
        input_variables=["human_input"], template=_DEFAULT_TEMPLATE)

    return prompt


def load_model() -> LLMChain:
    # Callbacks support token-wise streaming
    # Verbose is required to pass to the callback manager
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    Llama_llm = LlamaCpp(
        model_path=MODEL_PATH,
        callback_manager=callback_manager,
        verbose=True,
    )
    prompt = create_prompt()

    llm_chain = LLMChain(llm=Llama_llm, prompt=prompt)

    return llm_chain


llm_chain: LLMChain = load_model()

question: str = """What is the biggest country on Earth?"""

response: str = llm_chain.run(question)

print(response)
