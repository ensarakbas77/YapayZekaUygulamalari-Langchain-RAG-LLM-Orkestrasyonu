# LangChain Vector Store & RAG (Gemini)

Bu proje, LangChain ile bir **vector store** (vektör veritabanı) kurmayı ve bunun üzerine basit bir **RAG** (Retrieval-Augmented Generation) sistemi inşa etmeyi gösteriyor. Embedding ve chat modeli olarak Google Gemini kullanılıyor.

## Kavramlar (kısaca)

- **Embedding**: Bir metni, anlamını temsil eden sayısal bir vektöre çevirme işlemi. Anlamca yakın metinler, vektör uzayında birbirine yakın konumlanır.
- **Vector Store**: Bu vektörleri saklayıp "bu sorguya en yakın olanları bul" diyebileceğin bir veritabanı (burada [Chroma](https://www.trychroma.com/) kullanılıyor).
- **RAG (Retrieval-Augmented Generation)**: Modelin cevap üretirken kendi genel bilgisi yerine, vector store'dan çekilen ilgili dokümanları context olarak kullanmasını sağlayan yöntem — model, verilen bilginin dışına çıkmadan cevap verir.

## Dosyalar

### `vector_store_intro.py`
5 örnek doküman (hayvanlarla ilgili kısa açıklamalar) Chroma vector store'a embed'lenip kaydediliyor. Ardından bir sorgu embed'lenip, anlamca en yakın dokümanlar aranıyor (similarity search).

### `main.py`
Bir öncekinin üzerine RAG chain'i ekleniyor: **retriever** (vector store'dan en alakalı sonucu getiren adım) + **prompt** (modele sadece verilen context'i kullanmasını söyleyen talimat) + Gemini modeli, LCEL ile tek bir chain'de birleştiriliyor.

## Örnek Çalıştırma

### `vector_store_intro.py` — "dog" sorgusu

```
[Document(id='4d524b43-...', metadata={'source': 'mammal-pets-doc'}, page_content='Dogs are great companions, known for their loyalty and friendliness.'),
 Document(id='c2cc21cb-...', metadata={'source': 'mammal-pets-doc'}, page_content='Rabbits are social animals that need plenty of space to hop around.'),
 Document(id='e3eaca7f-...', metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.'),
 Document(id='0f905e83-...', metadata={'source': 'fish-pets-doc'}, page_content='Goldfish are popular pets for beginners, requiring relatively simple care.')]
```

Sonuçların sırasına dikkat: en alakalı doküman (`Dogs are great companions...`) ilk sırada, ama "Rabbits" de "Cats"ten önce geliyor — çünkü embedding modeli sadece kelime eşleşmesine değil, **anlamsal yakınlığa** bakıyor; dog, rabbit ve cat aynı kategoride (evcil memeli) yer alıyor.

### `main.py` — "tell me about cats" sorgusu

```
[{'type': 'text', 'text': 'Cats are independent pets that often enjoy their own space.', 'extras': {'signature': '...'}}]
```

Model kendi genel "kedi" bilgisini kullanmadı — cevap, retriever'ın bulduğu tek dokümanın içeriğiyle birebir sınırlı kaldı. Bu, RAG'in amaçladığı davranış: model, verilen context'in dışına çıkmıyor.

## Notlar

- `response.content`'in düz string değil blok listesi olarak dönmesi, önceki projelerde de gördüğümüz Gemini davranışı — chain'e `StrOutputParser` eklenirse sade string olarak alınabilir.