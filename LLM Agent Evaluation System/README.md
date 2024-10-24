# English
# LLM Eval Chatbot

This project creates a chatbot using a Streamlit application that manages tasks in Asana and Google Drive. Users can interact with the assistant by entering commands in natural language.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Descriptions](#code-descriptions)
- [License](#license)

## Features

- A chatbot that responds to user inputs.
- Integration with Asana and Google Drive.
- Storage of message history.
- Asynchronous response flow to enhance user experience.

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.7 or higher
- Streamlit
- LangChain
- Other related libraries (e.g., `dotenv`, `uuid`)

## Installation

1. Clone or download this project.
2. Install the required dependencies.
3. Create a `.env` file containing environment variables.

## Usage

1. Launch the application.
2. Interact with the chatbot in the opened browser window.

## Code Descriptions

### Main Components

- **Streamlit**: Used to create the user interface of the application.
- **LangChain**: Utilized for natural language processing and model integration.
- **Asynchronous**: Asynchronous programming is used for smoother and faster response retrieval.

### Functions

- **create_chatbot_instance**: This function creates and reuses the chatbot instance.
- **get_thread_id**: Generates a unique thread ID for each user.
- **prompt_ai**: Used to communicate user messages with the model.

### Application Flow

1. **Initialization**: When the application starts, the user interface is created, and existing message history is displayed.
2. **User Input**: When a user enters a command, it is added to the message history, and the assistant's response is retrieved.
3. **Response Display**: The assistant's response is updated and displayed to provide a better user experience.

## License

This project is licensed under the MIT License.




# Türkçe
# LLM Eval Chatbot

Bu proje, Asana ve Google Drive'daki görevleri yönetmek için bir chatbot oluşturan bir Streamlit uygulamasıdır. Kullanıcı, doğal dilde komutlar girerek asistanla etkileşime geçebilir.

## İçindekiler

- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Kod Açıklamaları](#kod-açıklamaları)
- [Lisans](#lisans)

## Özellikler

- Kullanıcıdan gelen girdilere yanıt veren bir chatbot.
- Asana ve Google Drive ile entegre çalışma.
- Mesaj geçmişinin saklanması.
- Asenkron yanıt akışı ile kullanıcı deneyimini iyileştirme.

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki bağımlılıkların yüklü olması gerekmektedir:

- Python 3.7 veya üzeri
- Streamlit
- LangChain
- Diğer ilgili kütüphaneler (örneğin, `dotenv`, `uuid`)

## Kurulum

1. Bu projeyi klonlayın veya indirin.
2. Gerekli bağımlılıkları yükleyin.
3. Ortam değişkenlerini içeren bir `.env` dosyası oluşturun.

## Kullanım

1. Uygulamayı başlatın.
2. Tarayıcınızda açılan pencerede chatbot ile etkileşime geçin.

## Kod Açıklamaları

### Ana Bileşenler

- **Streamlit**: Uygulamanın kullanıcı arayüzünü oluşturmak için kullanılır.
- **LangChain**: Doğal dil işleme ve model entegrasyonu için kullanılır.
- **Asynchronous**: Yanıtların daha akıcı ve hızlı bir şekilde alınabilmesi için asenkron programlama kullanılmıştır.

### Fonksiyonlar

- **create_chatbot_instance**: Bu fonksiyon, chatbot örneğini oluşturur ve yeniden kullanılabilir hale getirir.
- **get_thread_id**: Her kullanıcı için benzersiz bir iş parçacığı kimliği oluşturur.
- **prompt_ai**: Kullanıcıdan alınan mesajları model ile iletişim kurmak için kullanılır.

### Uygulama Akışı

1. **Başlatma**: Uygulama başlatıldığında, kullanıcı arayüzü oluşturulur ve mevcut mesaj geçmişi gösterilir.
2. **Kullanıcı Girdisi**: Kullanıcı bir komut girdiğinde, bu komut mesaj geçmişine eklenir ve asistanın yanıtı alınır.
3. **Yanıt Gösterimi**: Asistanın yanıtı, kullanıcıya daha iyi bir deneyim sunmak için güncellenerek gösterilir.

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.
