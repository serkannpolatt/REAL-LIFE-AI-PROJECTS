# Deepseek Document AI

DocuMind AI is a Streamlit application that analyzes PDF documents and provides intelligent responses using Natural Language Processing (NLP). It extracts information from uploaded PDF files and delivers relevant answers to user queries.

## ✨ Features
- **PDF Upload:** Users can upload PDF documents.
- **Text Splitting:** Documents are processed by splitting them into meaningful text chunks.
- **Vector Database:** Processed text chunks are stored in a vector-based database.
- **Natural Language Processing (NLP):** Provides the most relevant answers to user queries.
- **Customized UI:** Dark theme and user-friendly interface.

## 🔧 Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## 📁 Usage

1. **Upload a PDF:** Select a PDF file from the "Upload Research Document (PDF)" section on the main page.
2. **Wait for document processing:** The uploaded document will be automatically split and indexed.
3. **Ask a question:** Enter any question related to the document and press "Enter."
4. **Receive an answer:** The AI assistant will provide the best response based on the extracted document information.

## 🔄 How It Works

1. **PDF Upload & Save**
   - The uploaded PDF file is saved.
2. **Document Processing**
   - The PDF file is read and divided into text chunks for processing.
3. **Vector Database Update**
   - Processed text chunks are converted into vector format and added to a database.
4. **Question-Answer Process**
   - The user's question is answered based on the indexed documents.

## ✨ Technologies Used
- **Python**: Primary development language
- **Streamlit**: User interface
- **LangChain**: Natural language processing and document analysis
- **Ollama LLM**: Large language model
- **PDFPlumber**: PDF reading




# Türkçe
# Deepseek Document AI

DocuMind AI, PDF belgelerini analiz eden ve doğal dil işleme (NLP) kullanarak akıllı yanıtlar veren bir Streamlit uygulamasıdır. Kullanıcıların yüklediği PDF dosyalarından bilgi çıkarır ve kullanıcıların sorularına uygun yanıtlar sunar.

## ✨ Özellikler
- **PDF Yükleme:** Kullanıcılar PDF belgelerini yükleyebilir.
- **Metin Bölme:** Belgeler anlamlı metin parçalarına bölünerek işlenir.
- **Vektör Veri Tabanı:** İşlenen metin parçaları, vektör tabanlı bir veritabanında saklanır.
- **Doğal Dil İşleme (NLP):** Kullanıcının sorduğu sorulara en alakalı cevaplar verilir.
- **Özelleştirilmiş UI:** Koyu tema ve kullanıcı dostu arayüz.

## 🔧 Kurulum

1. Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

## 📁 Kullanım

1. **PDF yükleyin:** Ana sayfada "Upload Research Document (PDF)" bölümünden bir PDF dosyası seçin.
2. **Belgenin işlenmesini bekleyin:** Yüklenen belge otomatik olarak parçalanır ve indekslenir.
3. **Soru sorun:** Belge ile ilgili herhangi bir soru yazın ve "Enter" tuşuna basın.
4. **Yanıt alın:** AI asistanı, belgeden elde ettiği bilgilerle en iyi cevabı verecektir.

## 🔄 Çalışma Mantığı

1. **PDF Yükleme & Kaydetme**
   - Kullanıcıdan gelen PDF dosyası kaydedilir.
2. **Belge İşleme**
   - PDF dosyası okunur ve metin parçalarına bölünerek işlenir.
3. **Vektör Veri Tabanı Güncelleme**
   - İşlenen metin parçaları vektör formatına çevrilir ve bir veritabanına eklenir.
4. **Soru-Cevap İşlemi**
   - Kullanıcının sorusu, indekslenen belgelere göre en alakalı yanıt ile cevaplanır.

## ✨ Kullanılan Teknolojiler
- **Python**: Ana geliştirme dili
- **Streamlit**: Kullanıcı arayüzü
- **LangChain**: Doğal dil işleme ve belge analizi
- **Ollama LLM**: Büyük dil modeli
- **PDFPlumber**: PDF okuma
