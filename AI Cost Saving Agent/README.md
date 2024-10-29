# English
# Asana Task and Project Management Assistant

## Project Overview
This project is a Streamlit-based application designed for managing tasks and projects on Asana using AI. By leveraging models such as **Groq** and **OpenAI**, this tool aims to streamline task management by intelligently routing complex queries to the most suitable AI model. The project integrates with Asana’s API, enabling users to create, update, delete, and retrieve tasks and projects within their workspace.

## Key Features
- **Task Management**: Create, update, and delete tasks within specified Asana projects, directly from the application.
- **Project Management**: Fetch all available projects and create new ones with optional due dates.
- **AI Routing**: Intelligent decision-making for selecting models based on query complexity, enabling cost-effective and efficient responses.
- **Tool Integration**: Utilizes several helper functions, or "tools," for interacting with the Asana API, making task and project management seamless.
- **Dynamic UI**: Built with Streamlit to provide a user-friendly interface for task and project handling.

## Target Audience
This tool is designed for:
- **Project Managers** and **Team Leads**: Simplify task management in Asana with quick creation, updates, and retrievals.
- **AI Enthusiasts and Developers**: Demonstrates an integration of task management with AI-powered decision-making for routing tasks based on complexity.
- **Organizations Using Asana**: Improve task handling efficiency using AI-powered automation and Asana’s extensive project management capabilities.

## Setup Instructions
1. **Environment Variables**: Set up a `.env` file with the following:
   - `GROQ_MODEL`: Model name for Groq (e.g., `llama3-groq-70b-8192-tool-use-preview`).
   - `OPENAI_MODEL`: Model name for OpenAI (e.g., `gpt-4o`).
   - `ASANA_ACCESS_TOKEN`: Access token for the Asana API.
   - `ASANA_WORKPLACE_ID`: Workspace ID for the Asana account.
2. **Install Dependencies**: Install required packages, including `asana`, `dotenv`, `streamlit`, and other necessary libraries.
3. **Run the Application**: Launch the app using Streamlit by running `streamlit run <filename>.py`.

## How It Works
1. **Task and Project Operations**: Functions like `create_asana_task`, `get_asana_projects`, and `delete_task` provide operations on Asana tasks and projects.
2. **AI Model Routing**: The function `decide_model_from_prompt` determines whether a complex or simple AI model should handle a request based on user input.
3. **Tool Invocation**: If an AI model decides that a tool (API function) should be called, it performs the function and updates the conversation history.
4. **Dynamic Prompting**: User messages are processed with a focus on task simplicity, providing routing instructions for AI selection.

## Usage Scenarios
- **Creating a Task**: Users can input a task name and due date to create tasks directly in Asana.
- **Fetching Projects**: Retrieve a list of projects in the Asana workspace for task organization.
- **AI-Driven Task Suggestions**: The AI models suggest next steps based on the complexity of the request, with simpler requests routed to cheaper models.

This project serves as an automation tool for Asana users, integrating AI models to enhance productivity by simplifying task management in a collaborative environment.


# Türkçe
# Asana Görev ve Proje Yönetim Asistanı

## Proje Özeti
Bu proje, Asana'da görev ve proje yönetimini yapay zeka kullanarak kolaylaştırmak için tasarlanmış Streamlit tabanlı bir uygulamadır. **Groq** ve **OpenAI** gibi modellerin kullanımı sayesinde, karmaşık sorguları en uygun AI modeline yönlendirerek görev yönetimini daha verimli hale getirir. Proje, kullanıcıların Asana çalışma alanlarındaki görev ve projeleri oluşturmasına, güncellemesine, silmesine ve geri çağırmasına olanak tanıyan Asana API’si ile entegredir.

## Temel Özellikler
- **Görev Yönetimi**: Belirlenen Asana projeleri içerisinde doğrudan görev oluşturma, güncelleme ve silme işlemleri.
- **Proje Yönetimi**: Mevcut projeleri çağırma ve isteğe bağlı son teslim tarihleriyle yeni projeler oluşturma.
- **AI Yönlendirmesi**: Sorgu karmaşıklığına göre en uygun modelin seçilmesini sağlayan akıllı karar mekanizması, maliyet etkin ve verimli yanıtlar sunar.
- **Araç Entegrasyonu**: Asana API ile etkileşimde bulunmak için "araçlar" olarak tanımlanan yardımcı işlevleri kullanarak görev ve proje yönetimini kolaylaştırır.
- **Dinamik Kullanıcı Arayüzü**: Görev ve proje işlemleri için kullanıcı dostu bir arayüz sağlamak amacıyla Streamlit kullanılarak inşa edilmiştir.

## Hedef Kitle
Bu araç aşağıdaki kullanıcılar için tasarlanmıştır:
- **Proje Yöneticileri** ve **Ekip Liderleri**: Asana'da görev oluşturma, güncelleme ve geri çağırma işlemlerini kolaylaştırır.
- **AI Meraklıları ve Geliştiriciler**: Sorgu karmaşıklığına göre görev yönlendirme yapan bir AI uygulamasının entegrasyonunu sergiler.
- **Asana Kullanan Organizasyonlar**: AI destekli otomasyon ve Asana’nın kapsamlı proje yönetimi özellikleriyle görev yönetim verimliliğini artırır.

## Kurulum Talimatları
1. **Ortam Değişkenleri**: Aşağıdaki değişkenleri içeren bir `.env` dosyası oluşturun:
   - `GROQ_MODEL`: Groq modeli adı (örn. `llama3-groq-70b-8192-tool-use-preview`).
   - `OPENAI_MODEL`: OpenAI modeli adı (örn. `gpt-4o`).
   - `ASANA_ACCESS_TOKEN`: Asana API erişim anahtarı.
   - `ASANA_WORKPLACE_ID`: Asana hesabı için çalışma alanı ID'si.
2. **Bağımlılıkları Kurun**: `asana`, `dotenv`, `streamlit` ve diğer gerekli kütüphaneleri içeren paketleri yükleyin.
3. **Uygulamayı Çalıştırın**: Uygulamayı Streamlit ile başlatmak için `streamlit run <dosya_adi>.py` komutunu çalıştırın.

## Nasıl Çalışır
1. **Görev ve Proje İşlemleri**: `create_asana_task`, `get_asana_projects` ve `delete_task` gibi işlevler, Asana görev ve projeleri üzerinde işlemler sağlar.
2. **AI Model Yönlendirmesi**: `decide_model_from_prompt` işlevi, kullanıcının girdisine göre basit veya karmaşık bir AI modelinin talebi işlemesine karar verir.
3. **Araç Çağırma**: Eğer bir AI modeli bir aracın (API işlevi) çağrılmasına karar verirse, bu işlevi yürütür ve konuşma geçmişini günceller.
4. **Dinamik Promptlama**: Kullanıcı mesajları basitlik odaklı işlenir ve AI seçiminde yönlendirme talimatları verilir.

## Kullanım Senaryoları
- **Görev Oluşturma**: Kullanıcılar bir görev adı ve teslim tarihi girerek Asana’da doğrudan görev oluşturabilirler.
- **Projeleri Çağırma**: Asana çalışma alanındaki projeler listelenerek görev organizasyonu yapılabilir.
- **AI Destekli Görev Önerileri**: AI modelleri, talebin karmaşıklığına göre sonraki adımları önerir ve basit talepleri uygun maliyetli modellere yönlendirir.

Bu proje, Asana kullanıcıları için görev yönetimini kolaylaştıran ve yapay zeka modellerini entegre ederek iş verimliliğini artıran bir otomasyon aracı olarak hizmet vermektedir.
