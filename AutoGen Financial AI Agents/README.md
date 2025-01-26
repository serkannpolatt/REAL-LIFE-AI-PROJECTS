### EN
### Learn How to Configure an AutoGen Financial AI Agent

In this notebook, you'll learn step-by-step how to create your own financial AI agent using AutoGen, with clear and practical examples to guide you through the entire process.

#### Setup and Instructions:
🔹 **Agent Setup**:  
I created two key agents: AssistantAgent and UserProxyAgent (more on that below).

🔹 **Fetch and Analyze Data**:  
We start by fetching NVIDIA's historical stock prices, then plot them for visual analysis.

🔹 **Develop a Momentum Trading Strategy**:  
The agents generate a momentum trading strategy, implement it in Python, compute it using NVIDIA's historical data, and then save the results as both a plot and a CSV file.

Here’s a quick overview of the workflow:

1- Propose a Python code implementation of a momentum trading strategy.

2- Apply the strategy to NVIDIA's historical prices.

3- Save the results as 'nvidia_trading_strategy.csv'.

4- Plot and save the strategy in 'nvidia_trading_strategy.png'.

#### Meet the Agents:

🔹 **AssistantAgent**  
Acts like a typical AI assistant powered by a large language model (LLM). It can generate and refine Python code or summarize texts without needing human input or code execution.

🔹 **UserProxyAgent**  
This agent is more advanced. It can:
- Interact with human inputs,
- Execute code directly,
- Call functions or tools, 
- And either rely on LLMs for answers or execute code, depending on the task.

More explanation in the notebook.

#### Key Takeaways:

1. **Easy to Implement**: The setup process is straightforward and user-friendly.

2. **Simpler and More Powerful**: In my experience, this approach is even easier and more effective than using the standard OpenAI Assistant AI.

3. **Automatic and Local File Generation**: Files are generated automatically and saved locally for future use, unlike OpenAI Assistant AI where I had to include multiple checks to confirm file creation and re-request if necessary.

4. **Highly Customizable**: You can create multiple agents and integrate them into a GroupChat manager (which I’ll cover in future notebooks), offering a high degree of customization and flexibility.

### TR
### NASIL YAPILIR: AutoGen Finansal AI Ajanı Yapılandırma

Bu not defterinde, AutoGen kullanarak kendi finansal AI ajanınızı nasıl oluşturacağınızı, sizi adım adım yönlendirecek net ve pratik örneklerle öğreneceksiniz.

#### Kurulum ve Talimatlar:
🔹 **Ajan Kurulumu**:  
İki ana ajan oluşturdum: AssistantAgent ve UserProxyAgent (aşağıda daha fazla bilgi bulabilirsiniz).

🔹 **Veri Çekme ve Analiz Etme**:  
İlk olarak, NVIDIA'nın geçmiş hisse senedi fiyatlarını çekiyoruz ve ardından görsel analiz için bunları grafikle gösteriyoruz.

🔹 **Bir Momentum Ticaret Stratejisi Geliştirme**:  
Ajanlar, bir momentum ticaret stratejisi oluşturur, Python ile uygular, NVIDIA'nın geçmiş verilerini kullanarak hesaplar ve ardından sonuçları hem grafik hem de CSV dosyası olarak kaydeder.

İşte iş akışının hızlı bir özeti:

1- Bir momentum ticaret stratejisinin Python kodu uygulamasını önerin.

2- Stratejiyi NVIDIA'nın geçmiş fiyatlarına uygulayın.

3- Sonuçları 'nvidia_trading_strategy.csv' olarak kaydedin.

4- Stratejiyi 'nvidia_trading_strategy.png' olarak çizip kaydedin.

#### Ajanlarla Tanışın:

🔹 **AssistantAgent**  
Büyük bir dil modeli (LLM) tarafından güçlendirilen tipik bir AI asistanı gibi çalışır. Python kodu üretebilir, düzenleyebilir veya metinleri özetleyebilir, insan girişi veya kod çalıştırılmasına gerek olmadan.

🔹 **UserProxyAgent**  
Bu ajan daha gelişmiştir. Şunları yapabilir:
- İnsan girdileriyle etkileşime girebilir,
- Kodu doğrudan çalıştırabilir,
- Fonksiyonları veya araçları çağırabilir,
- Ve göreve bağlı olarak, yanıtlar için LLM'lere başvurabilir veya kodu çalıştırabilir.

Daha fazla açıklama not defterinde mevcuttur.

#### Önemli Notlar:

1. **Kolay Kurulum**: Kurulum süreci basit ve kullanıcı dostudur.

2. **Daha Basit ve Daha Güçlü**: Deneyimlerime göre, bu yaklaşım standart OpenAI Assistant AI'ye kıyasla çok daha kolay ve etkili.

3. **Otomatik ve Yerel Dosya Üretimi**: Dosyalar otomatik olarak üretilir ve gelecekteki kullanım için yerel olarak kaydedilir, OpenAI Assistant AI'de olduğu gibi dosya oluşturma işlemini doğrulamak ve gerekiyorsa tekrar istemek zorunda kalmazsınız.

4. **Yüksek Derecede Özelleştirilebilir**: Birden fazla ajan oluşturabilir ve bunları bir GrupChat yöneticisine entegre edebilirsiniz (bunu gelecekteki not defterlerinde ele alacağım), bu da yüksek özelleştirme ve esneklik sunar.
