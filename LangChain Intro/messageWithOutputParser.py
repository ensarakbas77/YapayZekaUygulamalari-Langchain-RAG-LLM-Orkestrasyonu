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

# StrOutputParser, modelin döndürdüğü AIMessage'ı sade bir metne (string) çevirir.
# response.content'i elle işlemek yerine bu standart aracı kullanmak, farklı
# modellerin farklı çıktı formatlarıyla tek tek uğraşmaktan kurtarır.
parser = StrOutputParser()

if __name__ == "__main__":
    result = model.invoke(messages)
    print(parser.invoke(result))