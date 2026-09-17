import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL_NAME"))

# Önceki dosyalarda mesajlar sabitti (hep Türkçe'ye çeviriyordu).
# ChatPromptTemplate ile mesajları şablon haline getirip {language} ve {text} gibi
# değişkenleri çalışma zamanında (chain.invoke ile) doldurabiliriz.
system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

parser = StrOutputParser()

# Zincire artık prompt_template de eklendi: prompt_template -> model -> parser.
# invoke'a verilen dict önce şablondaki boşlukları doldurup mesaj listesine dönüşür,
# sonra modele gider, en sonda parser çıktıyı sade string'e çevirir.
chain = prompt_template | model | parser


if __name__ == "__main__":
    print(chain.invoke({"language": "Italian", "text": "Hi"}))