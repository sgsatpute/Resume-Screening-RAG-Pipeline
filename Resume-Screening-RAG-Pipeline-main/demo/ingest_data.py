import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader


def ingest(df: pd.DataFrame, content_column: str, embedding_model):
    loader = DataFrameLoader(df, page_content_column=content_column)

    # Bug 5 fix — old chunk_overlap=500 was 49% of chunk_size=1024,
    # flooding FAISS with near-duplicate chunks and slowing retrieval.
    # 150/1000 = 15% overlap, which is the standard recommended ratio.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    documents       = loader.load()
    document_chunks = text_splitter.split_documents(documents)

    vectorstore_db = FAISS.from_documents(
        document_chunks,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return vectorstore_db