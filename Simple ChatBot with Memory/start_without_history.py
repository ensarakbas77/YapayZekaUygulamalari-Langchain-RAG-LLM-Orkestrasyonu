import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL_NAME"))


if __name__ == "__main__":

    # Modelin GERÇEK bir hafızası yok; her invoke() çağrısı bağımsız ve önceki
    # çağrılardan habersizdir. "Konuşma geçmişi" hissi vermek için, önceki
    # mesajları (hem kullanıcının hem modelin) elle bir liste halinde her
    # seferinde yeniden gönderiyoruz.
    response = model.invoke(
        [
            HumanMessage(content="Hi!, I'm Ensar"),
            AIMessage(content="Hello Ensar! How can I assist you today?"),  # modelin gerçek eski cevabı değil, elle yazdığımız sahte geçmiş
            HumanMessage(content="What is my name?")  # model, yukarıdaki geçmiş sayesinde "Ensar" cevabını verebilir
        ]
    )
    print(response)