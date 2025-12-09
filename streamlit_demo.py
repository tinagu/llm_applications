from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
import cohere
from langchain_community.vectorstores.utils import DistanceStrategy
import streamlit as st
from typing import List, Dict
import json
import glob
import numpy as np
import re
import os

st.title("Demo support chatbot")

def load_embedding(questions: List[str], config: Dict) -> List[List[float]]:
    """
    Create or load embedding
    """

    question_embedding_files = glob.glob("embedding_cache/*.npy")
    if not question_embedding_files:
        question_embedding_lst = config["embedding_func"].embed(questions, input_type="search_document")
        for i, question_embed in enumerate(question_embedding_lst):
            np.save(f"question_embed_{i}.npy", question_embed)
    else:
        question_embedding_lst = []
        for embedding_file in question_embedding_files:
            question_embedding_lst.append(np.load(embedding_file))

    return question_embedding_lst

def create_store(config):
    """
    Create the question-answer store
    """
    with open("ground_truth.json", "r") as i_f:
        content = json.load(i_f)

    doc_store = {pair["question"]: pair["answer"] for pair in content.get("questions", [])}
    if config["enable_llm"]:
        questions = list(doc_store.keys())
        questions_embedding = load_embedding(questions, config)
        question_embedding_pairs = zip(questions, questions_embedding)
        question_store = FAISS.from_embeddings(question_embedding_pairs,
                                               metadatas=[{'doc_id': i} for i in range(len(questions))],
                                               embedding=config["embedding_func"],
                                               distance_strategy=DistanceStrategy.COSINE)
    else:
        question_store = None


    return question_store, doc_store

@st.cache_data
def get_exist_questions(doc_store):
    return ["Ask me a question."] + list(doc_store.keys())

def check_relevance(question1, question2, config):
    response = config["llm"].chat(
        model=config["LLM_MODEL"],
        message=json.dumps([
            ('system', "Reply only Yes or No if the two questions below are similar or not."),
            ('user', f'Question1:{question1}\n\nQuestion2:{question2}')
        ])
    )

    print(response)
    return "yes" in response.text.lower()

def response_generator(messages: List[str], config):
    """
    Generate the answer
    """
    print(messages[-1])
    question = messages[-1]["content"]
    print(f"{question=}")
    question_store = config["question_store"]
    doc_store = config["doc_store"]
    print(config)

    answer = "no answer"
    if question in doc_store:
        print("question found in doc_store")
        # Exist question
        answer = doc_store[question]
    elif config["enable_llm"]:
        # Search similar exist question
        print("search similar question")
        similar_exist_question, score = question_store.similarity_search_with_relevance_scores(question, k=1)[0]
        print(f"{similar_exist_question=}, {score=}")
        similar_exist_question = similar_exist_question.page_content
        if check_relevance(question, similar_exist_question, config):
            answer = doc_store[similar_exist_question]

    print("answer: ", answer)
    return answer

def generate_llm_response(messages, config):
    stream = config["llm"].chat_stream(
        model=config["LLM_MODEL"],
        message=json.dumps([
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ])
    )
    for data in stream:
        if data.event_type == "text-generation":
            yield data.text

def clean_question(question):
    """
    Clean the question
    """
    question = re.sub("[\d]+", "", question)
    question = re.sub("^select", "", question, flags=re.IGNORECASE)
    return question

def init_config():
    config = {
        "cohere_api_key": os.environ.get("COHERE_API_KEY", ""),
        "enable_llm": True if len(os.environ.get("COHERE_API_KEY", "")) > 0 else False
    }
    if config["enable_llm"]:
        config["embedding_func"] = CohereEmbeddings(model="embed-english-light-v3.0",
                                                    cohere_api_key=config["cohere_api_key"])
        config["llm"] = cohere.Client(api_key=config["cohere_api_key"])
        config["LLM_MODEL"] = "command-a-03-2025"

    return config

def main():
    if "config" not in st.session_state:
        st.session_state.config = init_config()

    print(st.session_state.config)

    question_store, doc_store = create_store(st.session_state.config)
    if "doc_store" not in st.session_state:
        st.session_state.config["question_store"] = question_store
        st.session_state.config["doc_store"] = doc_store

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.selectbox("Select question", [""] + get_exist_questions(doc_store))
    if not question:
        return

    if "Ask" in question:
        question = st.text_input(
            "Questions: ", placeholder="Ask me anything ...", key="input"
        )
    question = clean_question(question)
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    response = response_generator(st.session_state.messages, st.session_state.config)
    print("from response_generate:", type(response))
    if response != "no answer":
        st.chat_message("assistant").markdown(response)
    else:
        print(st.session_state.config)
        print("to llm response")
        if st.session_state.config["enable_llm"]:
            response = st.chat_message("assistant").write_stream(generate_llm_response(st.session_state.messages, st.session_state.config))
        else:
            st.chat_message("assistant").markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

main()