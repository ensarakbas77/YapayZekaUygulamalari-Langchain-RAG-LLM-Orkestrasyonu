import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ChatGoogleGenerativeAI, LangChain'in Gemini modelleriyle konuşmak için kullandığı sınıf.
# Model adını .env'den okuyoruz, böylece kodu değiştirmeden farklı modeller denenebilir.
model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL_NAME"))

# LangChain'de bir konuşma, farklı rollerdeki mesajlardan oluşan bir listeyle temsil edilir:
# SystemMessage -> modele nasıl davranması gerektiğini söyler (talimat/persona)
# HumanMessage  -> kullanıcının gönderdiği asıl mesaj
messages = [
    SystemMessage(content="Translate the following from English to Turkish"),
    HumanMessage(content="hi!"),
]

if __name__ == "__main__":
    # invoke(), mesajları modele gönderip cevabı bekler; dönen nesne bir AIMessage'dır.
    response = model.invoke(messages)

    # response.content ham model çıktısıdır (bazı modellerde düz string,
    # Gemini gibi bazılarında ekstra metadata içeren blok listesi olabilir).
    print(response.content)