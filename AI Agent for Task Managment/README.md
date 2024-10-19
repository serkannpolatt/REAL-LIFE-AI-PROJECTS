# AI Agent for Asana Task Management

## Project Overview
This project creates a chat bot interface to manage tasks and projects in Asana using the Asana API and AI tools. Users can perform tasks such as creating tasks, retrieving projects, and updating tasks.

## Features
- **Task Creation**: Add new tasks to specified projects.
- **Retrieve Projects**: View all projects in your Asana workspace.
- **Retrieve Tasks**: List all tasks within a specific project.
- **Update Tasks**: Update the status or due date of existing tasks.
- **Delete Tasks**: Remove unwanted tasks from Asana.

## Requirements
- Python 3.x
- Required libraries:
  - Asana
  - OpenAI
  - dotenv
  - Streamlit
  - Langchain

## Installation
1. Clone or download this repository.
2. Install the required libraries by running the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Asana API key and OpenAI model:
   ```makefile
   ASANA_ACCESS_TOKEN=<your_asana_access_token>
   LLM_MODEL=<your_model>
   ASANA_WORKPLACE_ID=<your_asana_workspace_id>
   ```

## Usage
To start the project, run the following command:
```bash
streamlit run main.py
```
Then, open the application in your browser and start managing tasks.


# Türkçe
# AI Agent for Asana Task Management

## Proje Hakkında
Bu proje, Asana'nın API'sini ve AI araçlarını kullanarak Asana'daki görevleri ve projeleri yönetmek için bir sohbet botu arayüzü oluşturur. Kullanıcılar, görev oluşturma, projeleri alma ve görev güncelleme gibi işlemleri yapabilirler.

## Özellikler
- **Görev Oluşturma**: Belirtilen projeye yeni görevler ekleyin.
- **Projeleri Alma**: Asana çalışma alanınızdaki tüm projeleri görüntüleyin.
- **Görevleri Alma**: Belirli bir projedeki tüm görevleri listeleyin.
- **Görev Güncelleme**: Mevcut görevlerin durumunu veya son tarihini güncelleyin.
- **Görev Silme**: İstenmeyen görevleri Asana'dan silin.

## Gereksinimler
- Python 3.x
- Gerekli kütüphaneler:
  - Asana
  - OpenAI
  - dotenv
  - Streamlit
  - Langchain

## Kurulum
1. Bu depoyu klonlayın veya indirin.
2. Gerekli kütüphaneleri yüklemek için terminalde şu komutu çalıştırın:
   ```bash
   pip install -r requirements.txt
   ```
3. `.env` dosyasını oluşturun ve Asana API anahtarınızı ve OpenAI modelinizi ekleyin:
   ```makefile
   ASANA_ACCESS_TOKEN=<your_asana_access_token>
   LLM_MODEL=<your_model>
   ASANA_WORKPLACE_ID=<your_asana_workspace_id>
   ```

## Kullanım
Projeyi başlatmak için aşağıdaki komutu çalıştırın:
```bash
streamlit run main.py
```
Ardından, tarayıcınızda uygulamayı açın ve görev yönetimine başlayın.