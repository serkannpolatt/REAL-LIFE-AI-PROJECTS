# Google Developer Students için AI Destekli Bilgi Üreticisi 🚀

## Proje Tanımı

Document Genie, Google'ın Üretken Yapay Zekası, özellikle Gemini-PRO modelinin gelişmiş yeteneklerinden yararlanarak PDF belgelerinden metin çıkarmak ve analiz etmek için tasarlanmış güçlü bir Streamlit uygulamasıdır. Bu araç, yüklenen belgelerin içeriğine dayalı olarak kullanıcı sorgularına kesin, bağlam duyarlı yanıtlar sunmak için Retrieval-Augmented Generation (RAG) çerçevesini kullanır. 

## Amaç

Bu projenin amacı, Google Developer Students topluluğuna yapay zeka ve doğal dil işleme (NLP) teknolojilerini kullanarak belge analizini nasıl otomatikleştirebileceklerini göstermektir. Document Genie, öğrencilere RAG çerçevesinin pratik bir uygulamasını sunarak, belgelerden bilgi çıkarma, soru-cevap sistemleri oluşturma ve daha fazlası gibi çeşitli alanlarda kullanılabilecek bir araç sunar.

## Özellikler

- **Anında Bilgiler**: Yüklenen PDF belgelerinden metinleri çıkarır ve analiz ederek anında bilgiler sağlar.
- **Retrieval-Augmented Generation (RAG)**: Yüksek kaliteli, bağlamsal olarak alakalı yanıtlar için Google'ın Üretken Yapay Zeka modeli Gemini-PRO'yu kullanır.
- **Güvenli API Anahtarı Girişi**: Üretken yapay zeka modellerine erişim için Google API anahtarlarının güvenli bir şekilde girilmesini sağlar.
- **GDSC Etkinlikleri İçin Harika**: Bu proje, Google Developer Student Clubs etkinliklerinde yapay zeka ve doğal dil işleme konularını tanıtmak için mükemmel bir başlangıç noktasıdır.
- **Çoklu Belge Desteği**: Aynı anda birden fazla PDF belgesi yükleyebilir ve analiz edebilirsiniz.
- **Kullanıcı Dostu Arayüz**: Streamlit ile oluşturulmuş basit ve sezgisel bir kullanıcı arayüzü sunar.
- **Özelleştirilebilir Parametreler**: Metin parçalama ve vektör oluşturma süreçlerini özelleştirmek için çeşitli parametreler sunar.

## Başlarken

### Ön Koşullar

- Google API Anahtarı: Google'ın Üretken Yapay Zeka modelleriyle etkileşim kurmak için bir Google API anahtarı edinin. Anahtarınızı almak için [Google API Anahtarı Kurulumu](https://makersuite.google.com/app/apikey) adresini ziyaret edin.
- Streamlit: Bu uygulama Streamlit ile oluşturulmuştur. Ortamınızda Streamlit'in kurulu olduğundan emin olun.
- Python 3.7 veya üzeri: Uygulama Python 3.7 veya üzeri bir sürüm gerektirir.

### Kurulum

Bu depoyu klonlayın veya kaynak kodunu yerel makinenize indirin. Uygulama dizinine gidin ve gerekli Python paketlerini yükleyin:

```bash
git clone <repository_url>
cd <application_directory>
pip install -r requirements.txt
streamlit run "main.py"
```

