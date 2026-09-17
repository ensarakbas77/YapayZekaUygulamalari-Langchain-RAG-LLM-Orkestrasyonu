import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL_NAME"))

messages = [
    SystemMessage(content="Translate the following from English to Turkish"),
    HumanMessage(content="hi!"),
]

parser = StrOutputParser()

# LCEL (LangChain Expression Language) "|" operatörüyle model ve parser'ı zincirliyoruz.
# chain.invoke(...) çağrıldığında sırayla: önce model.invoke() çalışır (AIMessage döner),
# sonra o çıktı otomatik olarak parser.invoke()'a geçer (düz string'e çevrilir).
# Yani bir önceki dosyadaki "result = model.invoke(...)" + "parser.invoke(result)"
# iki adımını tek satıra indirger.
chain = model | parser

if __name__ == "__main__":
    print(chain.invoke(messages))