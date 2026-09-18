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

load_dotenv()

model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL_NAME"))

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | model
config = {"configurable": {"session_id": "firstChat"}}

# input_messages_key="messages": prompt'taki MessagesPlaceholder'ın beklediği anahtarla
# aynı olmalı — bir önceki dosyada (add_history.py) çözdüğümüz aynı kural burada da geçerli.
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

if __name__ == "__main__":
    while True:
        user_input = input("> ")

        # .invoke() yerine .stream() kullanıyoruz: cevabın tamamı gelene kadar beklemek
        # yerine, model ürettikçe parça parça (chunk chunk) AIMessageChunk nesneleri alırız.
        # Bu, ChatGPT tarzı "kelime kelime yazılıyor" hissini vermek için kullanılır.
        for r in with_message_history.stream(
                {
                    "messages": [HumanMessage(content=user_input)]
                },
                config=config,
        ):
            # r.content yerine r.text kullanıyoruz: Gemini'de content düz string değil,
            # blok listesi olabiliyor (önceki dosyalarda gördüğümüz gibi). .text,
            # hem string hem blok-listesi içeriği otomatik sade metne çeviren,
            # langchain-core'un sağladığı standart bir property (r.content kullansaydık
            # her chunk'ta çirkin bir liste/dict çıktısı görürdün).
            print(r.text, end="|")