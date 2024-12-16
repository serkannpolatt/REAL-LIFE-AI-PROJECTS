# English
# AI Model Query System

## Overview
This project integrates a large language model (LLM) with a retrieval-augmented generation (RAG) system to provide answers to user queries based on documents. It uses state-of-the-art transformer models and advanced natural language processing techniques for text generation, question-answering, and document retrieval. The system is optimized for efficiency and scalability, allowing it to process complex queries quickly, either through GPU or CPU.

## Key Features
- **Text Generation**: The system generates detailed, contextually relevant responses to queries based on input prompts.
- **Document Retrieval**: It retrieves relevant information from a collection of documents, such as PDFs, using vector databases and embeddings.
- **Efficient Model Usage**: The model is configured for optimized memory usage, utilizing quantization techniques for large models.
- **Real-Time Query Handling**: The system can handle queries in real-time, processing the text generation and retrieval operations within seconds.
- **Pipeline Integration**: The model is set up within a pipeline that is ready to generate responses and integrate into larger systems or applications.

## Use Cases
- **Legal Documents Querying**: The system can be used to extract information from legal texts, such as the EU AI Act, by querying the document with specific questions.
- **Research Assistance**: Researchers can use the system to answer domain-specific questions, extract data from academic papers, and synthesize information from multiple sources.
- **Customer Support**: Businesses can deploy this system for automatic query responses, based on the knowledge embedded in their document repositories.

## Setup
- Download the pretrained models from HuggingFace or your preferred model hosting service.
- Set up the necessary environment variables for model configuration and document paths.
- Load the models using the provided initialization scripts and adjust configurations (e.g., memory management, quantization settings) according to your hardware.

## Usage
- Once the system is set up, run the provided query function to ask questions based on a document.
- Input your queries, and the system will return answers derived from relevant document sections.
- For more complex workflows, integrate the model within your application to handle multiple queries or operate continuously in a production environment.

## Performance
- **Speed**: The system is optimized to handle queries with minimal latency, using efficient model loading and query execution methods.
- **Scalability**: It supports large document collections by using vector-based retrieval techniques and is capable of scaling with the size of the input data.

# Türkçe 
# Türkçe
# AI Model Sorgulama Sistemi

## Genel Bakış
Bu proje, kullanıcı sorgularına belge tabanlı yanıtlar sağlamak için büyük bir dil modeli (LLM) ile bilgi geri getirme ve artırılmış üretim (RAG) sistemini entegre eder. Metin üretimi, soru-cevaplama ve belge geri alımı için en son teknolojiye sahip transformer modelleri ve gelişmiş doğal dil işleme teknikleri kullanır. Sistem, verimlilik ve ölçeklenebilirlik için optimize edilmiştir, bu sayede karmaşık sorguları hızlı bir şekilde işleyebilir, ister GPU ister CPU kullanılarak.

## Ana Özellikler
- **Metin Üretimi**: Sistem, girilen istemlere dayalı olarak sorgulara ayrıntılı ve bağlamsal olarak ilgili yanıtlar üretir.
- **Belge Geri Alımı**: PDF gibi belge koleksiyonlarından, vektör veritabanları ve gömme (embedding) teknikleri kullanarak ilgili bilgileri geri alır.
- **Verimli Model Kullanımı**: Model, büyük modeller için nicemleme (quantization) tekniklerini kullanarak optimize edilmiş bellek kullanımıyla yapılandırılmıştır.
- **Gerçek Zamanlı Sorgu İşleme**: Sistem, gerçek zamanlı sorguları işleyebilir, metin üretimi ve geri alma işlemlerini saniyeler içinde gerçekleştirir.
- **Pipeline Entegrasyonu**: Model, yanıtlar üretmek ve daha büyük sistemlere veya uygulamalara entegre etmek için hazır bir pipeline içerisinde kuruludur.

## Kullanım Durumları
- **Hukuki Belgeler Sorgulama**: Sistem, EU AI Yasası gibi hukuki metinlerden belirli sorularla bilgi çıkarabilir.
- **Araştırma Yardımcısı**: Araştırmacılar, alanla ilgili soruları yanıtlamak, akademik makalelerden veri çıkarmak ve birden fazla kaynaktan bilgi sentezlemek için sistemi kullanabilir.
- **Müşteri Destek**: İşletmeler, belge depolarındaki bilgiye dayalı otomatik sorgu yanıtları sağlamak için bu sistemi kullanabilir.

# Kurulum
- Pretrained modelleri HuggingFace veya tercih ettiğiniz model barındırma hizmetinden indirin.
- Model yapılandırması ve belge yolları için gerekli ortam değişkenlerini ayarlayın.
- Modelleri, sağlanan başlatma betikleriyle yükleyin ve donanımınıza göre yapılandırmaları (örneğin, bellek yönetimi, nicemleme ayarları) ayarlayın.

# Kullanım
- Sistem kurulduktan sonra, belgeler üzerinden soru sormak için sağlanan sorgu işlevini çalıştırın.
- Sorgularınızı girin ve sistem, ilgili belge bölümlerinden türetilen yanıtları döndürecektir.
- Daha karmaşık iş akışları için, modeli uygulamanıza entegre ederek birden fazla sorguyu işleyebilir veya üretim ortamında sürekli çalıştırabilirsiniz.

# Performans
- **Hız**: Sistem, verimli model yükleme ve sorgu yürütme yöntemleri kullanarak sorguları minimum gecikmeyle işlemeye optimize edilmiştir.
- **Ölçeklenebilirlik**: Vektör tabanlı geri alma tekniklerini kullanarak büyük belge koleksiyonlarını destekler ve giriş verisinin boyutuyla ölçeklenebilir.

