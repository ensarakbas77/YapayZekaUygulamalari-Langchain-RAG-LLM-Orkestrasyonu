# RAG Intro — Bir Web Sayfası Üzerinde Soru-Cevap

Bu proje, LangChain ile baştan sona çalışan basit bir **RAG** (Retrieval-Augmented Generation) uygulaması kuruyor: bir blog yazısını internetten indiriyor, parçalara bölüp vector store'a kaydediyor ve kullanıcının sorusunu **sadece o yazıdaki bilgiye dayanarak** cevaplıyor. Embedding ve chat modeli olarak Google Gemini kullanılıyor.

## RAG neden var?

Bir dil modeli yalnızca eğitildiği verilerden haberdardır. Kendi dokümanların, şirket içi bilgiler ya da eğitim tarihinden sonra yazılmış içerikler hakkında soru sorduğunda ya bilmediğini söyler ya da uydurur. RAG bu sorunu şöyle çözer:

1. Bilgi kaynağını (blog yazısı, PDF, doküman…) parçalara bölüp bir vector store'a koyarız.
2. Kullanıcı soru sorduğunda, soruya **anlamca en yakın** parçaları vector store'dan buluruz.
3. Bu parçaları soruyla birlikte modele "context" olarak veririz ve sadece bunlara dayanarak cevaplamasını isteriz.

Model kendi hafızasına güvenmek yerine, önüne konan kaynaklardan okuyarak cevap verir.

## Pipeline: 5 adım

```
Web sayfası ──► [1. Load] ──► [2. Split] ──► [3. Index] ──► Vector Store (Chroma)
                                                                    │
Soru ──────────────────────────────────────────► [4. Retrieve] ◄────┘
                                                      │ (en yakın parçalar)
                                                      ▼
                                      Prompt (soru + context) ──► [5. Generate] ──► Cevap
```

### 1. Load — Kaynağı yükle
`WebBaseLoader` bir web sayfasını indirip LangChain'in `Document` nesnelerine çevirir. `bs4.SoupStrainer` ile sayfanın sadece başlık ve yazı içeriğini (`post-title`, `post-header`, `post-content`) alıyoruz; menü, footer gibi gereksiz kısımlar baştan elenir.

### 2. Split — Parçalara böl
Uzun bir yazıyı tek parça olarak embed'lemek yerine küçük parçalara (chunk) böleriz; böylece arama sırasında yalnızca soruyla ilgili kısımlar bulunur.

```python
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

- `chunk_size=1000`: her parça en fazla yaklaşık 1000 karakter.
- `chunk_overlap=200`: ardışık parçalar 200 karakter örtüşür; böylece bir fikir parça sınırında ikiye bölünüp anlamını kaybetmez.
- "Recursive" olması: önce paragraf, sonra cümle, en son kelime sınırlarından bölmeyi dener; yani metni mümkün olduğunca anlamlı yerlerden keser.

### 3. Index — Embed'le ve kaydet
Her parça `GoogleGenerativeAIEmbeddings` ile bir vektöre çevrilip Chroma'ya kaydedilir. Anlamca yakın metinler vektör uzayında birbirine yakın durur; arama bu yakınlığa dayanır.

> Chroma burada **bellekte (in-memory)** çalışıyor, ayrı bir sunucu gerekmiyor. `persist_directory` verilmediği için script bitince veri silinir ve her çalıştırmada sayfa yeniden indirilip yeniden embed'lenir.

### 4. Retrieve — İlgili parçaları getir
```python
retriever = vectorstore.as_retriever()
```
Retriever, soruyu alır ve anlamca en yakın parçaları (varsayılan olarak 4 tane) `Document` listesi olarak döner. Prompt'a düz metin gerektiği için bu listeyi `format_docs` fonksiyonuyla tek bir string'e çeviriyoruz.

### 5. Generate — Cevabı üret
Prompt, modele sadece verilen context'i kullanmasını, bilmiyorsa "bilmiyorum" demesini ve en fazla üç cümleyle cevap vermesini söyler. Tüm akış LCEL (LangChain Expression Language: bileşenleri `|` operatörüyle birbirine bağlayıp bir zincir kurmamızı sağlayan LangChain sözdizimi) ile tek bir zincirde birleşir:

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Zincirin başındaki dict **paralel** çalışır: gelen soru bir yandan retriever'a gidip `{context}`'i doldurur, bir yandan `RunnablePassthrough` sayesinde değişmeden `{question}`'a yazılır. Sonra sırayla prompt → model → parser akar. `StrOutputParser` Gemini'nin blok listesi formatındaki cevabını sade metne çevirir.

### Zincir nasıl akar?

Zincir **soldan sağa** akar, sağdan sola değil. `|` operatörü Unix/Linux terminalindeki pipe'a benzer: soldaki bileşenin çıktısı, sağdaki bileşenin girdisi olur. Yani `a | b | c` yazdığımızda önce `a`, sonra `b`, en son `c` çalışır.

`rag_chain.stream("What is Task Decomposition?")` çağrıldığında sırayla şunlar olur:

1. **Dict (iki kol paralel):** Soru hem retriever'a gider (bulunan parçalar `format_docs` ile metne çevrilir, yani `retriever | format_docs` kendi içinde küçük bir zincirdir) hem de `RunnablePassthrough` ile değişmeden geçer. Çıktı: `{"context": "<bulunan parçalar>", "question": "What is Task Decomposition?"}`
2. **`prompt`:** Bu dict'i alıp `{context}` ve `{question}` boşluklarını doldurur, modele gidecek mesajı hazırlar.
3. **`llm`:** Hazır prompt'u Gemini'ye gönderir, cevap bir `AIMessage` olarak döner.
4. **`StrOutputParser`:** `AIMessage`'ı düz metne çevirir.

`.stream()` kullandığımız için son adımdaki cevap tek seferde değil, model ürettikçe parça parça ekrana akar.

## Kurulum

`requirements.txt`:

```
langchain
langchain-community
langchain-core
langchain-google-genai
langchain-text-splitters
langchain-chroma
beautifulsoup4
faiss-cpu
python-dotenv
tqdm
```

`.env` dosyası (`.env.example` şablon olarak kullanılabilir):

```env
# Google Gemini Yapılandırması
GEMINI_API_KEY="your-gemini-api-key-here"
GEMINI_MODEL_NAME="gemini-3.1-flash-lite"
GEMINI_EMBEDDING_MODEL_NAME="models/gemini-embedding-001"

# LangChain / LangSmith İzleme (opsiyonel; kapatmak için TRACING_V2=false)
LANGCHAIN_API_KEY="your-langsmith-api-key-here"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="your-project-name"

# LangSmith hesabın hangi bölgedeyse ona göre ayarla:
# US (varsayılan): https://api.smith.langchain.com
# EU:              https://eu.api.smith.langchain.com
LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com

# WebBaseLoader'ın web sitelerine kendini tanıtmak için kullandığı kimlik
USER_AGENT="RAGIntro/1.0"
```

Çalıştırma:

```bash
python main.py
```

## Örnek Çalıştırma

Soru: `"What is Task Decomposition?"` (kaynak: Lilian Weng'in AI ajanları üzerine blog yazısı)

```
Task decomposition is a technique where complex tasks are broken down into smaller, more manageable subgoals or steps to enhance model performance. This can be achieved through LLM prompting, human input, or specialized techniques like Chain of Thought and Tree of Thoughts. Additionally, systems like HuggingGPT utilize task planning to parse user requests into a series of logical, dependent tasks.
```

Cevap, modelin genel bilgisinden değil, retriever'ın yazıdan bulduğu parçalardan geliyor. Cevap `.stream()` ile parça parça ekrana yazılır. Model her seferinde birebir aynı cümleleri üretmediği için çıktı çalıştırmadan çalıştırmaya küçük farklılıklar gösterebilir.

## Notlar

- **`USER_AGENT` uyarısı**: `langchain_community` bu değişkeni import edildiği anda kontrol ediyor. Bu yüzden `main.py`'de `load_dotenv()` çağrısı, `langchain_community` importundan **önce** yapılıyor.
- **`langchain-community` sunset uyarısı**: Paket artık aktif olarak geliştirilmiyor ve çalıştırınca bir `DeprecationWarning` görünüyor. Paket kurulu olduğu sürece çalışmaya devam ediyor, bu projede sorun çıkarmıyor.
- **Prompt**: Orijinal örneklerde hazır RAG prompt'u LangChain Hub'dan (`hub.pull`) çekilir. `langchainhub` paketi deprecated olduğu için burada aynı yapıdaki prompt elle yazıldı.
- **AFC mesajı**: Çıktıdaki `Direct use of automatic function calling...` satırı Google SDK'sının bilgi mesajıdır, çalışmayı etkilemez.
