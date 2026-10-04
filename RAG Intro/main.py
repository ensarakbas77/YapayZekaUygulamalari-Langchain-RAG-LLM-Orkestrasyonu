import os
import bs4
from dotenv import load_dotenv

# USER_AGENT kontrolü langchain_community import edilirken yapılıyor; bu yüzden .env'i
# WebBaseLoader'ı import etmeden ÖNCE yüklememiz gerekiyor.
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL_NAME"))


# --- 1. YÜKLE (Load) ---
# RAG'in ilk adımı bilgi kaynağını okumaktır. WebBaseLoader bir web sayfasını indirip
# Document nesnelerine çevirir. SoupStrainer ile sayfanın sadece başlık ve yazı
# içeriğini alıyoruz; menü, footer gibi gereksiz HTML kısımları baştan elenmiş oluyor.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# --- 2. BÖL (Split) ---
# Uzun bir yazıyı tek parça embed'lemek yerine küçük parçalara (chunk) böleriz; böylece
# arama sırasında sadece soruyla ilgili kısımlar bulunur. chunk_overlap=200, ardışık
# parçaların 200 karakter örtüşmesini sağlar; bir fikir parça sınırında ikiye bölünüp
# anlamını kaybetmesin diye. RecursiveCharacterTextSplitter önce paragraf, sonra cümle,
# en son kelime sınırlarından bölmeyi dener.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 3. İNDEKSLE (Index) ---
# Her parça embedding modeliyle vektöre çevrilip Chroma'ya kaydedilir.
# (persist_directory vermediğimiz için vector store bellekte tutulur, script bitince silinir.)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL_NAME")),
)

# --- 4. GETİR (Retrieve) ---
# as_retriever(), vector store'u bir Runnable'a çevirir: soruyu alır, anlamca en yakın
# parçaları (varsayılan olarak 4 tane) Document listesi olarak döner.
retriever = vectorstore.as_retriever()

# RAG prompt'u: modele sadece verilen context'e dayanarak, kısa cevap vermesini söylüyoruz.
# Burada prompt'u elle yazdık. Aynısı LangChain Hub'da hazır duruyor ("rlm/rag-prompt");
# Hub'dan çekmek istersek güncel yol LangSmith SDK'sı:
#
#   from langsmith import Client
#   prompt = Client().pull_prompt("rlm/rag-prompt")
#
# Eski yol (`from langchain import hub` + `hub.pull(...)`) artık önerilmiyor: langchainhub
# deprecated, `hub` modülü de `langchain_classic` altına taşınıp deprecated oldu.
# Hub çağrısı LangSmith'e gittiği için LANGCHAIN_API_KEY (ve EU hesaplarında doğru
# LANGCHAIN_ENDPOINT) gerekebilir. Hazır prompt da {context} ve {question} değişkenlerini
# kullandığı için zincirin geri kalanı değişmeden çalışır.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question}\n"
            "Context: {context}\n"
            "Answer:",
        )
    ]
)


# Retriever bir Document listesi döner, ama prompt'a düz metin gerekir.
# Bu fonksiyon parçaların metinlerini aralarına boş satır koyarak tek string'e birleştirir.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- 5. ÜRET (Generate) ---
# Zincirin başındaki dict paralel çalışır: gelen soru hem retriever'a gidip {context}'i
# (retriever | format_docs: bul, sonra metne çevir) doldurur, hem de RunnablePassthrough
# sayesinde değişmeden {question}'ı doldurur. Sonra sırayla prompt -> model -> parser akar.
# (format_docs sıradan bir fonksiyon, "|" ile zincire girince otomatik Runnable'a çevrilir.)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    # .stream() cevabı parça parça döner; end="" ve flush=True ile parçalar
    # satır atlamadan, anında ekrana yazılır ("yazıyor" efekti).
    for chunk in rag_chain.stream("What is Task Decomposition?"):
        print(chunk, end="", flush=True)
