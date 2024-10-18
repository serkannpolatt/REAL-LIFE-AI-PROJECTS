# English
# AI Agent for RAG (Retrieval-Augmented Generation)

## Overview

This project provides an end-to-end RAG (Retrieval-Augmented Generation) pipeline that allows users to ask questions based on data extracted from a specific website and receive contextual answers. The application utilizes a series of modern machine learning and natural language processing (NLP) techniques to load, process, and answer questions based on textual documents.

## Technical Details

### Architecture

- **Web Data Loading**: Uses `WebBaseLoader` to load data from a user-provided URL, filtering the page content by selecting specific HTML elements.
  
- **Embedding Creation**: Loaded text documents are transformed into numerical representations using OpenAI’s embedding models, maintaining the meaning and context of the documents.

- **Vector Store Management**: Pinecone is used to store the vector representations of the documents, enabling faster and more efficient answering of queries.

- **Retrieval-Augmented Generation**: User queries are answered by combining relevant documents retrieved from the vector store. Answers are generated using Groq LLM and finally presented to the user in a coherent manner.

### Features

- **Document Loading**: Ability to load text content from a specified web address.
- **Question Answering**: Provides contextual and meaningful answers to user queries.
- **Dynamic Vector Store Management**: Dynamic management and updating of data as needed.
- **Modular Integration**: The application is integrated with various NLP and machine learning libraries.

### Technologies Used

- **Streamlit**: For the user interface.
- **Langchain**: For NLP processes.
- **Pinecone**: For vector store management.
- **OpenAI Embeddings**: For text embedding operations.
- **Groq LLM**: For answer generation.

# Türkçe
# RAG (Retrieval-Augmented Generation) için AI Ajanı

## Genel Bakış

Bu proje, belirli bir web sitesinden veri çıkararak kullanıcıların soru sormasına ve bağlamsal yanıtlar almasına olanak tanıyan uçtan uca bir RAG (Retrieval-Augmented Generation) pipeline'ı sunmaktadır. Uygulama, metin belgelerini yüklemek, işlemek ve soruları yanıtlamak için modern makine öğrenimi ve doğal dil işleme (NLP) tekniklerinden oluşan bir dizi kullanmaktadır.

## Teknik Detaylar

### Mimari

- **Web Veri Yükleme**: Kullanıcı tarafından sağlanan bir URL'den veri yüklemek için `WebBaseLoader` kullanır ve sayfa içeriğini belirli HTML öğelerini seçerek filtreler.

- **Gömülü Temsil Oluşturma**: Yüklenen metin belgeleri, OpenAI'nin gömme modellerini kullanarak sayısal temsilere dönüştürülür; bu sayede belgelerin anlamı ve bağlamı korunur.

- **Vektör Depo Yönetimi**: Vektör temsilleri Pinecone kullanılarak depolanır; bu da sorguların daha hızlı ve daha verimli bir şekilde yanıtlanmasını sağlar.

- **Retrieval-Augmented Generation**: Kullanıcı sorguları, vektör deposundan alınan ilgili belgelerle birleştirilerek yanıtlanır. Yanıtlar Groq LLM kullanılarak üretilir ve son olarak kullanıcıya uyumlu bir şekilde sunulur.

### Özellikler

- **Belge Yükleme**: Belirli bir web adresinden metin içeriği yükleyebilme.
- **Soru Yanıtlama**: Kullanıcı sorgularına bağlamsal ve anlamlı yanıtlar sağlar.
- **Dinamik Vektör Depo Yönetimi**: Gerekli olduğunda verilerin dinamik yönetimi ve güncellenmesi.
- **Modüler Entegrasyon**: Uygulama, çeşitli NLP ve makine öğrenimi kütüphaneleri ile entegre edilmiştir.

### Kullanılan Teknolojiler

- **Streamlit**: Kullanıcı arayüzü için.
- **Langchain**: NLP süreçleri için.
- **Pinecone**: Vektör depo yönetimi için.
- **OpenAI Gömülü Temeller**: Metin gömme işlemleri için.
- **Groq LLM**: Yanıt üretimi için.