import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

load_dotenv()
model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL_NAME"))

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

parser = StrOutputParser()

# Bir önceki dosyayla (messageWithTemplates.py) aynı chain; farkı burada sonucu
# terminale tek seferlik yazdırmak yerine bir web servisi olarak dışarı açacağız.
chain = prompt_template | model | parser

# FastAPI uygulaması: langserve, bu app üzerine LangChain chain'lerini
# otomatik olarak REST endpoint'lerine dönüştürür.
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# add_routes, chain'i "/chain" altında bir API'ye dönüştürür:
# /chain/invoke, /chain/stream ve /chain/playground gibi endpoint'ler otomatik oluşur.
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn
    # uvicorn, FastAPI uygulamasını çalıştıran ASGI sunucusudur.
    # Script çalıştığı sürece sunucu ayakta kalır ve gelen istekleri karşılar.
    uvicorn.run(app, host="localhost", port=8000)