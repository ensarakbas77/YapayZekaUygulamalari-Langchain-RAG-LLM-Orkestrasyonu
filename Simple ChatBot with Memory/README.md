# LangChain Chatbot — Message History & Streaming

Konuşma geçmişi (chat history) yönetimini adım adım gösteren üç script.

## Dosyalar

### `start_without_history.py`
Modelin hafızası olmadığını gösterir. Önceki mesajlar (kullanıcı + model) elle bir listeye yazılıp her `invoke()` çağrısında yeniden gönderilir.

### `add_history.py`
`RunnableWithMessageHistory` ile konuşma geçmişi otomatikleştirilir. Her `session_id` için ayrı bir hafıza tutulur, önceki mesajları elle eklemeye gerek kalmaz.

### `history_with_streaming.py`
Aynı yapı, `.invoke()` yerine `.stream()` kullanılarak cevabın parça parça (chunk chunk) alınmasını sağlar.

## Örnek Çalıştırma (`add_history.py`)

```
sys:1: LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.
> my name is ensar
Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
Hello, Ensar! It's a pleasure to meet you. How are you doing today? Is there anything I can help you with?
> what is my name
Your name is Ensar.
> 456 * 3123 = ?
456 * 3123 = 1,424,088
> what day is it?
Today is Tuesday, May 21, 2024.
```