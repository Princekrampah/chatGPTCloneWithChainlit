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


def load_model() -> LlamaCpp:
    # Callbacks support token-wise streaming
    # Verbose is required to pass to the callback manager
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # n_gpu_layers - determines how many layers of the model are offloaded to your GPU.
    # n_batch - how many tokens are processed in parallel.

    # Change this value based on your model and your GPU VRAM pool.
    n_gpu_layers = 40
    # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_batch = 512

    Llama_llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.75,
        max_tokens=2000,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,
    )

    return Llama_llm


llm = load_model()

model_prompt: str = """
Question: What is the biggest country on Earth?
"""

response: str = llm(prompt=model_prompt)

print(response)
