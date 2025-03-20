import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import logging
import time

# get conf
import configparser

config = configparser.ConfigParser()
config.read("/webSearch/config.ini")

google_api_key = config["google"]["api_key"]
cse_id = config["google"]["cse_id"]
openai_api_key = config["openai"]["api_key"]

from openai import OpenAI

client = OpenAI(api_key=openai_api_key)

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dataclasses import dataclass


# model config
@dataclass
class ModelConfig:
    max_input: int
    max_output: int


model_info = {
    "gpt-3.5-turbo": ModelConfig(max_input=16385, max_output=4096),
    "gpt-4o-mini": ModelConfig(max_input=128000, max_output=16384),
    "gpt-4-turbo": ModelConfig(max_input=128000, max_output=4096),
    "gpt-4o": ModelConfig(max_input=128000, max_output=16384),
}
############################################################################################################

# token calculator
import tiktoken

encoders = {
    "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
    "gpt-4o-mini": tiktoken.encoding_for_model("gpt-4o-mini"),
    "gpt-4-turbo": tiktoken.encoding_for_model("gpt-4-turbo"),
    "gpt-4o": tiktoken.encoding_for_model("gpt-4o"),
}


class TokenLengthCalculator:
    def __init__(self, model_name):
        if model_name not in encoders:
            raise ValueError(f"Model {model_name} not supported.")
        self.encoder = encoders[model_name]

    def tiktoken_len(self, text):
        tokens = self.encoder.encode(text)
        return len(tokens)


############################################################################################################

model_name = "gpt-4o"


def add_messages_to_history(new: list, old: list):
    new.extend(old)
    return new


class State(TypedDict):
    current_user_input: str
    history: Annotated[list, add_messages_to_history]
    need_web_search: bool
    keyword: str
    urls: list[str]
    web_documents: list[Document]
    chunked_documents: list[str]
    context: str
    total_tokens: int
    openai_responses_api_answer: str
    display_message: str
    manual_web_search_answer: str


logging.basicConfig(level=logging.INFO)

def log_node_execution(node_name, print_state=False, state_keys_to_print=None):
    def decorator(func):
        def wrapper(state, *args, **kwargs):
            start_time = time.time()
            logging.info(f"Executing node: {node_name}")
            if print_state:
                if state_keys_to_print:
                    filtered_state = {
                        key: state[key] for key in state_keys_to_print if key in state
                    }
                    logging.debug(f"Filtered state: {filtered_state}")
                else:
                    logging.debug(f"State before execution: {state}")
            result = func(state, *args, **kwargs)
            end_time = time.time()
            logging.info(f"Node {node_name} executed in {end_time - start_time:.2f}s")
            return result
        return wrapper
    return decorator

def log_node_execution_async(node_name, print_state=False, state_keys_to_print=None):
    def decorator(func):
        async def wrapper(state, *args, **kwargs):
            start_time = time.time()
            logging.info(f"Executing node: {node_name}")
            if print_state:
                if state_keys_to_print:
                    filtered_state = {
                        key: state[key] for key in state_keys_to_print if key in state
                    }
                    logging.debug(f"Filtered state: {filtered_state}")
                else:
                    logging.debug(f"State before execution: {state}")
            result = await func(state, *args, **kwargs)
            end_time = time.time()
            logging.info(f"Node {node_name} executed in {end_time - start_time:.2f}s")
            return result
        return wrapper
    return decorator


@log_node_execution("get_user_input")
def get_user_input(state: State):
    user_input = input("User: ")
    return {"current_user_input": user_input}


@log_node_execution("check_use_web_search")
def check_use_web_search(state: State):
    need_web_search = input("Do you want to search the web? (y/n): ") == "y"
    return {"need_web_search": need_web_search}


@log_node_execution("extract_web_search_keyword")
def extract_web_search_keyword(state: State):
    prompt = (
        f'Extract the search query from the user input: "{state["current_user_input"]}"'
    )
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "user", "content": prompt},
        ],
        tool_choice="required",
        tools=[
            {
                "type": "function",
                "name": "extract_web_search_keyword",
                "description": "Extract the search query from the user's input.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "The keyword for web search.",
                        }
                    },
                    "required": ["keyword"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
    )
    return {"keyword": json.loads(response.output[0].arguments).get("keyword")}


@log_node_execution("google_search", True, ["keyword"])
def google_search(state: State):
    google_api_key = config["google"]["api_key"]
    cse_id = config["google"]["cse_id"]
    filterPDFtemplate = f'{state["keyword"]} -filetype:pdf -filetype:doc'
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": filterPDFtemplate, "key": google_api_key, "cx": cse_id, "num": 3}
    web_pages = requests.get(url, params=params)
    return {"urls": [web.get("link") for web in web_pages.json().get("items", [])]}


# @log_node_execution_async("fetch_web_pages_to_documents", True)
# async def playwright_fetch_web_pages_to_documents(state: State):
#     from langchain_community.tools.playwright.utils import (
#         create_async_playwright_browser,
#     )
#     from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
#     import nest_asyncio
#     nest_asyncio.apply()
#     async_browser = create_async_playwright_browser()
#     toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
#     tools = toolkit.get_tools()
#     tools_by_name = {tool.name: tool for tool in tools}
#     navigate_tool = tools_by_name["navigate_browser"]
#     extract_text_tool = tools_by_name["extract_text"]
#     extracted_texts = []
#     for url in state["urls"]:
#         await navigate_tool.arun({"url": url})
#         page_content = await extract_text_tool.arun({})
#         extracted_texts.append(page_content)


@log_node_execution("fetch_web_pages_to_documents")
def fetch_web_pages_to_documents(state: State):
    loader = WebBaseLoader(web_path=state["urls"])
    web_documents = loader.load()

    for i, doc in enumerate(web_documents):
        file_path = f"/webSearch/ref{i + 1}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc.page_content)
        print(f"Document saved to {file_path}")

    return {"web_documents": web_documents}


@log_node_execution("chunking_documents")
def chunking_documents(state: State):
    # 10000 only for testing, actual value should be prompt tokens
    token_calculator = TokenLengthCalculator(model_name)
    splitter = RecursiveCharacterTextSplitter(
        # chunk_size=(model_info[model_name].max_input - 10000),
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
        length_function=token_calculator.tiktoken_len,
    )
    chunked_documents = []
    for doc in state["web_documents"]:
        chunks = splitter.split_text(doc.page_content)
        chunked_documents.extend(chunks)
    return {"chunked_documents": chunked_documents}


@log_node_execution("retrieval_documents")
def retrieval_documents(state: State):
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    chunk_embeddings = embeddings.embed_documents(state["chunked_documents"])

    embeddings_df = pd.DataFrame(chunk_embeddings)
    embeddings_csv_path = "/webSearch/embeddings.csv"
    embeddings_df.to_csv(embeddings_csv_path, index=False)
    print(f"Embeddings saved to {embeddings_csv_path}")

    user_input_embedding = embeddings.embed_query(state["current_user_input"])

    def calculate_similarity(index):
        return index, cosine_similarity(user_input_embedding, chunk_embeddings[index])

    with ThreadPoolExecutor() as executor:
        similarities = list(
            executor.map(calculate_similarity, range(len(chunk_embeddings)))
        )

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarities[:5]]
    top_chunks = [state["chunked_documents"][i] for i in top_indices]
    return {"context": top_chunks}


@log_node_execution("combine_documents", True, ["context"])
def combine_documents(state: State):
    combined_text = ""
    for doc in state["chunked_documents"]:
        combined_text += doc.replace("\n", " ").strip()
    return {"context": combined_text}


@log_node_execution("generate_ai_message")
def generate_ai_message(state: State):
    response = ask_llm(
        state["current_user_input"], state["context"], None, None, state["history"]
    )
    return {"manual_web_search_answer": response.output_text}


@log_node_execution("chatbot")
def chatbot(state: State):
    response = ask_llm(state["current_user_input"], "", None, None, state["history"])
    return {"display_message": response.output_text}


@log_node_execution("web_search_by_openai_built_in_tool")
def web_search_by_openai_built_in_tool(state: State):
    response = ask_llm(
        state["current_user_input"],
        "",
        [{"type": "web_search_preview"}],
        "required",
        state["history"],
    )
    return {"openai_responses_api_answer": response.output_text}


def ask_llm(question, context, tools=None, tool_choice=None, history=[]):
    messages = []
    messages.append(
        {"role": "developer", "content": "使用者國籍為台灣, 請嚴格執行輸出只用繁體中文"}
    )
    if len(history) > 0:
        messages.extend(history)
    if len(context) > 0:
        messages.append({"role": "developer", "content": context})
    if len(question) > 0:
        messages.append({"role": "user", "content": question})
    response = client.responses.create(
        model="gpt-4o",
        input=messages,
        tool_choice=tool_choice,
        tools=tools,
    )
    return response


@log_node_execution("gen_graph_png")
def gen_graph_png(state: State):
    try:
        graph_image_path = "/webSearch/graph.png"
        if os.path.exists(graph_image_path):
            os.remove(graph_image_path)  # Delete the existing file if it exists
        with open(graph_image_path, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        print(f"Graph image saved to {graph_image_path}")
    except Exception as e:
        print(f"Failed to save graph image: {e}")


@log_node_execution("print_answer")
def print_answer(state: State):
    if state["need_web_search"]:
        print("---------------------------------------------------------------")
        print(f'google_search_resp.output_text: {state["manual_web_search_answer"]}')
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        print(
            f'openai_responses_resp.output_text: {state["openai_responses_api_answer"]}'
        )
        print("---------------------------------------------------------------")
        return {
            "history": [
                {"role": "user", "content": state["current_user_input"]},
                {"role": "assistant", "content": state["openai_responses_api_answer"]},
            ]
        }
    print("---------------------------------------------------------------")
    print(f'chatbot_resp.output_text: {state["display_message"]}')
    print("---------------------------------------------------------------")


@log_node_execution("need_web_search")
def need_web_search(state: State) -> str:
    if state["need_web_search"]:
        return "Y"
    else:
        return "N"


@log_node_execution("should_quit")
def should_quit(state: State) -> str:
    quit_keywords = {"q", "quit", "exit"}
    if state["current_user_input"].strip().lower() in quit_keywords:
        return "Y"
    return "N"


graph_builder = StateGraph(State)
graph_builder.add_node("get_user_input", get_user_input)
graph_builder.add_node("check_use_web_search", check_use_web_search)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("extract_web_search_keyword", extract_web_search_keyword)
graph_builder.add_node("google_search", google_search)
graph_builder.add_node("fetch_web_pages_to_documents", fetch_web_pages_to_documents)
graph_builder.add_node("chunking_documents", chunking_documents)
graph_builder.add_node("retrieval_documents", retrieval_documents)
graph_builder.add_node("combine_documents", combine_documents)
graph_builder.add_node("generate_ai_message", generate_ai_message)
graph_builder.add_node(
    "web_search_by_openai_built_in_tool", web_search_by_openai_built_in_tool
)
graph_builder.add_node("print_answer", print_answer)
graph_builder.add_node("gen_graph_png", gen_graph_png)
graph_builder.add_edge(START, "gen_graph_png")
graph_builder.add_edge("gen_graph_png", "get_user_input")
graph_builder.add_conditional_edges(
    "get_user_input",
    should_quit,
    {"Y": END, "N": "check_use_web_search"},
)
graph_builder.add_conditional_edges(
    "check_use_web_search",
    need_web_search,
    {"Y": "extract_web_search_keyword", "N": "chatbot"},
)
graph_builder.add_edge(
    "extract_web_search_keyword", "web_search_by_openai_built_in_tool"
)
graph_builder.add_edge("extract_web_search_keyword", "google_search")
graph_builder.add_edge("google_search", "fetch_web_pages_to_documents")
graph_builder.add_edge("fetch_web_pages_to_documents", "chunking_documents")
graph_builder.add_edge("chunking_documents", "retrieval_documents")
graph_builder.add_edge("retrieval_documents", "combine_documents")
graph_builder.add_edge("combine_documents", "generate_ai_message")
graph_builder.add_edge("chatbot", "print_answer")
graph_builder.add_edge("print_answer", "get_user_input")
graph_builder.add_edge(
    ["generate_ai_message", "web_search_by_openai_built_in_tool"], "print_answer"
)
graph = graph_builder.compile()


async def main():
    await graph.ainvoke(input={})


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
