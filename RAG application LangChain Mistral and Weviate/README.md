# English
# RAG application LangChain Mistral and Weviate

## Purpose
This project aims to create a process that leverages the LangChain library to work with a Weaviate-based vector database, supporting retrieval-augmented generation (RAG) scenarios. The project includes loading data from PDF documents, converting text into vectors, and generating responses tailored to queries.

## What Does the Project Do?
- **Vector Database Integration:** Interacts with a vector database using Weaviate.
- **Embedding Generation:** Converts texts into embedding format using the Hugging Face Sentence Transformers model.
- **Text Loading and Splitting:** Loads and chunks PDF documents for efficient processing.
- **RAG Chain:** Searches for relevant content using queries, merges the results, and generates accurate and concise responses.

## Use Cases
- Rapid access to information within academic papers or long documents.
- Internal information management and document content querying.
- Customized knowledge-based assistants similar to ChatGPT.

## How It Works
1. **Weaviate Connection:** Establishes a Weaviate client using an API key.
2. **Embedding Modeling:** Converts texts into embeddings using the "sentence-transformers/all-mpnet-base-v2" model.
3. **Text Loading and Splitting:** Loads text from PDFs using PyPDFLoader and splits it with RecursiveCharacterTextSplitter.
4. **Query and Answering:** Retrieves the most relevant content from the database, constructs a response chain, and generates accurate answers.

## Notes
- **API Key:** External services such as Weaviate and Hugging Face require API keys.
- **Model:** By default, the "sentence-transformers/all-mpnet-base-v2" embedding model is used, but other models can be substituted.
- **Performance Optimization:** GPU support can be enabled if needed.

# Türkçe
# RAG application LangChain Mistral and Weviate

## Amaç
Bu proje, LangChain kütüphanesi kullanılarak Weaviate tabanlı bir vektör veritabanı üzerinde çalışan ve bilgi alma (retrieval-augmented generation, RAG) senaryolarını destekleyen bir süreç oluşturmayı hedeflemektedir. Proje, PDF dokümanlarından veri yükleme, metinleri vektör haline getirme, ve sorgulara uygun şekilde yanıt oluşturma gibi işlemleri içerir.

## Proje Neler Yapar?
- **Vektör Veritabanı Bağlantısı:** Weaviate kullanılarak bir vektör veritabanı ile etkileşim kurar.
- **Embedding Oluşturma:** Hugging Face Sentence Transformers modeli ile metinleri embedding formatına dönüştürür.
- **Metin Yükleme ve Ayrıştırma:** PDF dokümanlarını parçalar halinde böler ve şekillendirir.
- **RAG Zinciri:** Sorguları kullanıp ilgili içerikleri arar, bunları birleştirir ve doğru ve özet bir yanıt oluşturur.

## Kullanım Senaryoları
- Akademik makaleler veya uzun dokümanlar içindeki bilgiye hızlı erişim.
- Şirket içi bilgi yönetimi ve doküman içeriği sorgulama.
- ChatGPT benzeri, bilgi tabanlı özel yapılandırılmış yardımcılar.

## Nasıl İşler?
1. **Weaviate Bağlantısı:** API anahtarla bir Weaviate istemcisi oluşturulur.
2. **Embedding Modellemesi:** Metinler "sentence-transformers/all-mpnet-base-v2" modeline göre embedding olarak dönüştürülür.
3. **Metin Yükleme ve Parçalama:** PyPDFLoader kullanılarak PDF'den metinler yüklenir ve RecursiveCharacterTextSplitter ile parçalanır.
4. **Sorgu ve Yanıtlama:** Veritabanından en ilgili içerik parçalarını getirerek, bir yanıt zinciri oluşturulur ve doğru yanıt üretilir.

## Notlar
- **API Anahtarı:** Weaviate ve Hugging Face gibi harici hizmetlere erişim için API anahtarları gereklidir.
- **Model:** Varsayılan olarak, "sentence-transformers/all-mpnet-base-v2" embedding modeli kullanılmaktadır ancak farklı modeller tercih edilebilir.
- **Performans Optimizasyonu:** Gerekirse GPU desteği etkinleştirilebilir.