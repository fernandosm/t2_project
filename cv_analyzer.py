import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

st.set_page_config(
    page_title="CV Analyser",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

def load_document(file):
    import os

    name, ext = os.path.splitext(file)

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        print(f"Loading pdf document..{file}")
        loader = PyPDFLoader(file)

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        print(f"Loading docx document..{file}")
        loader = Docx2txtLoader(file)
    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader

        print(f"Loading txt document..{file}")
        loader = TextLoader(file)
    else:
        raise ValueError("Unsupported file format")
        return None
    data = loader.load()

    return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, question, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    answer = chain.run(question)
    return answer


def calculate_embeddings_cost(text):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in text])

    return total_tokens / 1000 * 0.0004

def clear_history():
    if "history" in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv("config/.env"), override=True)

    st.subheader("Ask questions about the _Resume_ 🤖")
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        uploaded_file = st.file_uploader(
            "Upload the CV", type=["pdf", "docx", "txt"]
        )
        chunk_size = st.number_input(
            "Chunk size", value=256, min_value=128, max_value=2048,
            on_change=clear_history
        )
        k = st.number_input("k", value=3, min_value=1, max_value=20, on_change=clear_history)
        add_data = st.button("Add data", on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner("Loading document, chunking and embedding file ..."):
                bytes_data = uploaded_file.read()
                file_name = os.path.join("data/st/docs/", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f"Number of chunks: {len(chunks)} of chunk size: {chunk_size}")
                embeddings_cost = calculate_embeddings_cost(chunks)
                st.write(f"Token cost: {embeddings_cost:.4f} ADA")

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success("Data loaded and embedded")

    q = st.text_input("Your question, please:")
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area("LLM Answer", value=answer)

            st.divider()

            if "history" not in st.session_state:
                st.session_state.history = ""
            
            value = f"Q: {q} \nA: {answer}"
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area("Chat History", value=h, key="history", height=400)
