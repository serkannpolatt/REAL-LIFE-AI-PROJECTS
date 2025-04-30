import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import time

st.set_page_config(
    page_title="GDSC RAG Demo", layout="wide"
)  # Streamlit sayfa ayarlarını yapılandır

# Logo ekleme (en üstte ve ortada)
logo = Image.open("gdsc.jpg")  # Logo dosyasını aç
col1, col2, col3 = st.columns(
    [1, 1.5, 1]
)  # Sütunları oluştur, orta sütunu daha geniş yap
with col2:
    st.image(logo, width=1000)  # Logoyu orta sütunda görüntüle

st.markdown("""
## GDSC Demo - Belgelerinizden Anında İçgörüler! 🚀

Merhaba Google Developer Student Club Üyeleri! 👋

Bu demo, Google'ın Üretken Yapay Zeka modellerini kullanarak belgelerinizden nasıl anında içgörüler elde edebileceğinizi gösteriyor. 
Retrieval-Augmented Generation (RAG) çerçevesi ile güçlendirilmiş bu sohbet robotu, yüklediğiniz PDF belgelerini analiz ederek sorularınıza doğru ve hızlı yanıtlar veriyor.

### Nasıl Çalışır? 🤔

Bu demoyu kullanmak çok kolay:

1. **API Anahtarınızı Girin**: Google AI Studio'dan edindiğiniz API anahtarınızı girin. (https://makersuite.google.com/app/apikey)
2. **Belgelerinizi Yükleyin**: PDF formatındaki belgelerinizi yükleyin. Birden fazla belge yükleyebilirsiniz!
3. **Sorunuzu Sorun**: Belgelerinizle ilgili herhangi bir soruyu sorun ve anında yanıt alın!

**İpucu:** Bu demo, Google'ın Gemini modelini kullanıyor. Farklı modelleri deneyerek sonuçları karşılaştırabilirsiniz!
""")

# API anahtarı girişi
api_key = st.text_input(
    "Google API Anahtarınızı Girin:", type="password", key="api_key_input"
)  # API anahtarı girişi için bir metin kutusu oluştur


def get_pdf_text(pdf_docs):
    # PDF belgelerinden metin çıkarır
    text = ""  # Metni saklamak için boş bir dize oluştur
    for pdf in pdf_docs:  # Yüklenen her PDF belgesi için döngü
        pdf_reader = PdfReader(pdf)  # PDF okuyucuyu oluştur
        for page in pdf_reader.pages:  # Her sayfa için döngü
            text += page.extract_text()  # Sayfa içeriğini metne ekle
    return text  # Birleştirilmiş metni döndür


def get_text_chunks(text):
    # Metni daha küçük parçalara böler
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )  # Metin bölücüyü oluştur
    chunks = text_splitter.split_text(text)  # Metni parçalara böl
    return chunks  # Metin parçalarını döndür


def get_vector_store(text_chunks, api_key):
    # Metin parçalarını vektörlere dönüştürür ve FAISS indeksine kaydeder
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )  # Gömme modelini oluştur
    vector_store = FAISS.from_texts(
        text_chunks, embedding=embeddings
    )  # Vektör veritabanını oluştur
    vector_store.save_local("faiss_index")  # Vektör veritabanını kaydet


def get_conversational_chain():
    # Sohbet zincirini oluşturur
    prompt_template = """
    Sağlanan bağlamdan mümkün olduğunca ayrıntılı olarak soruyu yanıtlayın, tüm ayrıntıları sağladığınızdan emin olun, eğer cevap
    sağlanan bağlamda değilse, sadece "cevap bağlamda mevcut değil" deyin, yanlış cevap vermeyin\n\n
    Bağlam:\n {context}?\n
    Soru: \n{question}\n

    Cevap:
    """  # Prompt şablonunu tanımla
    # "gemini-pro" yerine kullanılabilecek başka bir model deneyin
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest", temperature=0.3, google_api_key=api_key
    )  # Sohbet modelini oluştur
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )  # Prompt şablonunu oluştur
    chain = load_qa_chain(
        model, chain_type="stuff", prompt=prompt
    )  # Soru cevaplama zincirini oluştur
    return chain  # Sohbet zincirini döndür


def user_input(user_question, api_key):
    # Kullanıcının sorusunu alır, ilgili belgeleri arar ve yanıt üretir
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )  # Gömme modelini oluştur
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )  # Vektör veritabanını yükle
    docs = new_db.similarity_search(user_question)  # Benzer belgeleri ara
    chain = get_conversational_chain()  # Sohbet zincirini oluştur
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )  # Soruyu sor ve yanıtı al
    st.write("Yanıt: ", response["output_text"])  # Yanıtı ekrana yazdır
    time.sleep(15)  # Saniyede 1 istek ile sınırlamak için 15 saniye bekleyin


def main():
    # st.image(logo, width=100)

    st.header("AI Sohbet Robotu - GDSC Demo 🤖")

    user_question = st.text_input(
        "PDF Dosyalarınızdan Bir Soru Sorun:", key="user_question"
    )

    if user_question and api_key:
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menü")
        pdf_docs = st.file_uploader(
            "PDF Dosyalarınızı Yükleyin:",
            accept_multiple_files=True,
            key="pdf_uploader",
        )
        if st.button("Gönder ve İşle", key="process_button") and api_key:
            with st.spinner("İşleniyor..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Tamamlandı!")


if __name__ == "__main__":
    main()
