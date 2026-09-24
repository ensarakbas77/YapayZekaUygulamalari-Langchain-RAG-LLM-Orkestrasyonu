import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

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

vectorstore = Chroma.from_documents(
    documents,
    embedding=GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL_NAME")),
)

# RunnableLambda, sıradan bir Python fonksiyonunu (burada vectorstore.similarity_search)
# bir LCEL chain'inin parçası olabilecek bir Runnable'a çevirir. .bind(k=1), her çağrıda
# k parametresini sabitler (yani her sorguda sadece en alakalı 1 sonucu getir).
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

#print(retriever.batch(["cat", "shark"]))

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL_NAME"))

# Modele SADECE bu context'teki bilgiyi kullanarak cevap vermesini söylüyoruz — bu,
# RAG'in (Retrieval-Augmented Generation) temel fikri: model kendi genel bilgisi
# yerine, retriever'ın bulduğu dokümanlara dayanarak cevap üretir.
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

# Bu dict, LCEL'de "paralel çalıştırma" anlamına gelir: chain invoke edildiğinde,
# gelen girdi (örn. "tell me about cats") hem retriever'a gidip {context}'i doldurur,
# hem de RunnablePassthrough sayesinde değişmeden {question}'ı doldurur. İkisi birleşip
# tek bir dict olarak prompt'a, oradan da llm'e aktarılır.
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

if __name__ == "__main__":
    response = rag_chain.invoke("tell me about cats")
    # StrOutputParser eklemediğimiz için response bir AIMessage; response.content de
    # Gemini'de düz string değil blok listesi olabilir (önceki dosyalarda gördüğümüz gibi).
    print(response.content)