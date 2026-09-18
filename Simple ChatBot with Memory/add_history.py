import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL_NAME"))

# Bir önceki dosyada (message.py) konuşma geçmişini elle bir listeye yazıyorduk.
# Burada bu işi otomatikleştiriyoruz: her session_id için ayrı bir mesaj geçmişi
# tutan bir "hafıza deposu". Gerçek bir projede bu dict yerine Redis/Postgres gibi
# kalıcı bir depo kullanılır; burada basit tutmak için bellekte (in-memory) tutuyoruz.
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Bu session_id'yi ilk kez görüyorsak boş bir geçmiş oluştur, görmüşsek olanı döndür.
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# MessagesPlaceholder, sistem mesajından sonra değişken uzunlukta bir mesaj listesinin
# (yani konuşma geçmişi + yeni kullanıcı mesajının) prompt'a enjekte edileceği yeri işaretler.
# "messages" ismi, aşağıda invoke()'a verdiğimiz dict'teki anahtarla birebir eşleşmeli.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

parser = StrOutputParser()

chain = prompt | model | parser

# session_id, hangi konuşmanın geçmişini kullanacağımızı belirler — farklı session_id'ler
# birbirinden tamamen bağımsız, ayrı hafızalara sahip olur.
config = {"configurable": {"session_id": "firstChat"}}

# RunnableWithMessageHistory, chain'i sarmalayıp her invoke() çağrısında otomatik olarak:
# 1) get_session_history ile o session'ın geçmişini yükler,
# 2) yeni kullanıcı mesajını geçmişe ekleyip chain'e gönderir,
# 3) modelin cevabını da geçmişe kaydeder.
# input_messages_key="messages", yukarıdaki MessagesPlaceholder'ın beklediği anahtarla
# aynı olmak zorunda — aksi halde prompt, mesajları nereye yerleştireceğini bulamaz.
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

if __name__ == "__main__":
    # Artık her invoke() çağrısında sadece YENİ mesajı gönderiyoruz; geçmiş konuşmayı
    # elle listeye eklemek zorunda değiliz, RunnableWithMessageHistory bunu arka planda
    # otomatik hallediyor.
    while True:
        user_input = input("> ")
        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        print(response)