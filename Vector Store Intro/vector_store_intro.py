import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Document, bir metin parçasını (page_content) ve onunla ilgili ek bilgiyi (metadata)
# bir arada tutan LangChain'in temel veri yapısıdır. Vector store'a eklenecek her
# "bilgi parçası" bir Document olarak temsil edilir.
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# Embedding, bir metni anlamını temsil eden sayısal bir vektöre çevirme işlemidir.
# Anlamca yakın metinler, vektör uzayında birbirine yakın konumlanır — "benzerlik
# arama"nın (similarity search) temeli budur.
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL_NAME"))

# Chroma, embedding'leri diskte/bellekte saklayan ve benzerlik araması yapabilen açık
# kaynak bir vector database. from_documents(), tüm dokümanları otomatik olarak
# embed'leyip Chroma'ya kaydeder.
vectorstore = Chroma.from_documents(
    documents,
    embedding=embeddings,
)

if __name__ == "__main__":
    # similarity_search: sorguyu embed'ler, en anlamca yakın dokümanları döner.
    #print(vectorstore.similarity_search("cat"))

    # similarity_search_with_score: aynısı ama her sonucun yanında bir benzerlik
    # skoru da döner — sonuçların ne kadar "yakın" olduğunu görmek için kullanışlı.
    #print(vectorstore.similarity_search_with_score("cat"))

    # Burada sorguyu elle embed'leyip (embed_query), hazır vektörle arama yapıyoruz
    # (similarity_search_by_vector) — similarity_search'ün arka planda yaptığı
    # iki adımı (embed + ara) elle ayrıştırmış oluyoruz.
    embedding = embeddings.embed_query("dog")
    print(vectorstore.similarity_search_by_vector(embedding))