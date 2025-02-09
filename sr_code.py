# import chainlit as cl #support for chatbot API
import torch

# from chainlit.types import AskFileResponse

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub



text_splitter= RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
embedding = HuggingFaceEmbeddings()


def read_split_pdf(file_path):
    Loader = PyPDFLoader
    loader = Loader(file_path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    return docs


def create_vector_db(docs,embedding):
    vector_db = Chroma.from_documents(documents=docs,
                                    embedding=embedding)
    retriever = vector_db.as_retriever()

    return retriever


MODEL_NAME = "lmsys/vicuna-7b-v1.5"
def llm_model(model_name, tokenizer, max_new_tokens):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )
    tokenizer = tokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
    )

    return llm


docs = read_split_pdf()
retriever = create_vector_db(docs,embedding)
llm = llm_model(model_name = MODEL_NAME, tokenizer=AutoTokenizer(),max_new_tokens = 1024)


prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


USER_QUESTION = "list out all prompt methods in the file"
output = rag_chain.invoke(USER_QUESTION)
answer = output.split('Answer:')[1].strip()
answer