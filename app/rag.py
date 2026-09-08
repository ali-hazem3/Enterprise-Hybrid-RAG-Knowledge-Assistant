from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)


VECTOR_STORE_PATH = Path("storage/vector_store")


embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


def get_vector_store():
    return Chroma(
        collection_name="pulsefit_handbook",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_PATH),
    )


def search_documents(question: str, k: int = 3):
    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        question,
        k=k,
    )

    return results

if __name__ == "__main__":
    results = search_documents(
        "What are the PulseFit opening hours?"
    )

    for index, document in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult {index}")
        print("Source:", document.metadata.get("source"))
        print("Version:", document.metadata.get("version"))
        print(document.page_content)