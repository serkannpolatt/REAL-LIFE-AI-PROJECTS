# English
# Quantization
---
## Content
- **[Types of Quantization](#types-of-quantization)**
- **[GPTQ: Post-training quantization on generative models](#gptq-post-training-quantization-on-generative-models)**
- **[AutoGPTQ Integration](#autogptq-integration)**
- **[GGML](#GGML-General-Graphical-Model-Library)**
- **[NF4 vs. GGML vs. GPTQ](#nf4-vs-ggml-vs-gptq)**
- **[Quantization vs. Pruning vs. Knowledge Distillation](#quantization-vs-pruning-vs-knowledge-distillation)**
- **[Summary](#Summary)**



---
## Types of Quantization
1. **Post-Training Quantization (PTQ)**
   Is a straightforward technique where the weights of an already trained model are converted to lower precision without necessitating any retraining.
   Although easy to implement, PTQ is associated with potential performance degradation.
2. **Quantization-Aware Training (QAT)**
   incorporates the weight conversion process during the pre-training or fine-tuning stage, resulting in enhanced model performance. However, QAT is computationally expensive and demands representative training data.

* **We will focus on PTQ only.**



---
## GPTQ: Post-training quantization on generative models
* GPTQ is not only efficient enough to be applied to models boasting hundreds of billions of parameters, but it can also achieve remarkable precision by compressing these models to a mere 2, 3, or 4 bits per parameter without sacrificing significant accuracy.
* What sets GPTQ apart is its adoption of a mixed int4/fp16 quantization scheme. Here, model weights are quantized as int4, while activations are retained in float16. During inference, weights are dynamically dequantized, and actual computations are performed in float16.
* GPTQ has the ability to quantize models without the need to load the entire model into memory. Instead, it quantizes the model module by module, significantly reducing memory requirements during the quantization process.
* GPTQ first applies scalar quantization to the weights, followed by vector quantization of the residuals.

### **When you should use GPTQ?**
   * An approach that is being applied to numerous models and that is indicated by HuggingFace, is the following:
      - Fine-tune the original LLM model with bitsandbytes in 4-bit, nf4, and QLoRa for efficient fine-tuning.
      - Merge the adapter into the original model
      - Quantize the resulting model with GPTQ 4-bit


---
## AutoGPTQ Integration
> The AutoGPTQ library emerges as a powerful tool for quantizing Transformer models, employing the efficient GPTQ method.

* AutoGPTQ advantages : 
  - Quantized models are serializable and can be shared on the Hub.
  - GPTQ drastically reduces the memory requirements to run LLMs, while the inference latency is on par with FP16 inference. 
  - AutoGPTQ supports Exllama kernels for a wide range of architectures.
  - The integration comes with native RoCm support for AMD GPUs.
  - Finetuning with PEFT is available.



## AutoGPTQ Requirements
* Installing dependencies
```
!pip install -q -U transformers peft accelerate optimum
!pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
```

* In order to quantize your model, you need to provide a few arguemnts:
  - **bits:** The number of bits u need to quantize to.
  - **dataset:** dataset used to calibrate the quantization
  - **model_seqlen:** the model sequence length used to process the dataset
  - **block_name_to_quantize**
```
from optimum.gptq import GPTQQuantizer
quantizer = GPTQQuantizer(
      bits=4,
      dataset="c4",
      block_name_to_quantize = "model.decoder.layers",
      model_seqlen = 2048
)
```

* Save the model
```
save_folder = "/path/to/save_folder/"
quantizer.save(model,save_folder)
```

* Load quantized weights
```
from accelerate import init_empty_weights
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
empty_model.tie_weights()
quantized_model = load_quantized_model(empty_model, save_folder=save_folder, device_map="auto")
```

* Exllama kernels for faster inference
> For 4-bit model, you can use the exllama kernels in order to a faster inference speed. It is activated by default. If you want to change its value, you just need to pass disable_exllama in load_quantized_model(). In order to use these kernels, you need to have the entire model on gpus.
```
quantized_model = load_quantized_model(
      empty_model,
      save_folder=save_folder,
      device_map="auto",
      disable_exllama=False
)
```



---
## GGML (General Graphical Model Library)
GGML is a C library focused on machine learning. It was designed to be used in conjunction with the llama.cpp library.
* The library is written in C/C++ for efficient inference of Llama models. It can load GGML models and run them on a CPU. Originally, this was the main difference with GPTQ models, which are loaded and run on a GPU.

### Quantization with GGML
The way GGML quantizes weights is not as sophisticated as GPTQ’s. Basically, it groups blocks of values and rounds them to a lower precision. Some techniques, like Q4_K_M and Q5_K_M, implement a higher precision for critical layers. In this case, every weight is stored in 4-bit precision, with the exception of half of the attention.wv and feed_forward.w2 tensors.

* Experimentally, this mixed precision proves to be a good tradeoff between accuracy and resource usage.
* weights are processed in blocks, each consisting of 32 values. For each block, a scale factor (delta) is derived from the largest weight value. All weights in the block are then scaled, quantized, and packed efficiently for storage (nibbles).
* This approach significantly reduces the storage requirements while allowing for a relatively simple and deterministic conversion between the original and quantized weights.



---
## NF4 vs. GGML vs. GPTQ
<kbd>
   <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*yz7rSvjKtukVXdVHwxGfAQ.png">
</kbd>

* Based on these results, we can say that GGML models have a slight advantage in terms of perplexity. The difference is not particularly significant, which is why it is better to focus on the generation speed in terms of tokens/second.
* The best technique depends on your GPU: if you have enough VRAM to fit the entire quantized model, GPTQ with ExLlama will be the fastest. If that’s not the case, you can offload some layers and use GGML models with llama.cpp to run your LLM.



--- 
## Quantization vs. Pruning vs. Knowledge Distillation
## Pruning
> Removing some of the connections in our neural network, which result in **sparse network**.
<kbd>
   <img src="https://miro.medium.com/v2/resize:fit:559/1*0mGJU7eNWgrqU5sgk-7RoQ.png">
</kbd>

* Procedure:
  * Pick pruning factor X.
  * In each layer, set the lowest X% of weights to zero.
  * Optional: retrain the model to recover accuracy.

### General MatMul vs. Sparse MatMul
<kbd>
   <img src="https://github.com/ElDokmak/LLMs/assets/85394315/c7a3526e-995b-43e7-8a99-6b1e43883826">
</kbd>

* General MatMul: we multiply every row and colum by each other even they were zeros.
* Sparse MatMul: we skip rows and columns with zeros, we multiply only ones with values in it.

## Distillation
> We train a **Student model** on the output labels of the **Teacher model**
<kbd>
   <img src="https://github.com/ElDokmak/LLMs/assets/85394315/160351cd-1a97-450d-b8f8-ab9a9dac65f0">
</kbd>



---
## Summary
<img src="https://github.com/ElDokmak/LLMs/assets/85394315/15cf3903-455c-4811-8f7e-14b6eaf30237">

---

# Türkçe
# Niceleme

---

## İçerik
- **[Niceleme Türleri](#Niceleme-Türleri)**
- **[GPTQ: Generatif Modellerde Eğitim Sonrası Kuantizasyon](#GPTQ-Üretici-Modellerde-Eğitim-Sonrası-Niceleme)**
- **[AutoGPTQ Entegrasyonu](#AutoGPTQ-Entegrasyonu)**
- **[GGML](#GGML-Genelleştirilmiş-Grafiksel-Model-Kitaplığı)**
- **[NF4 vs. GGML vs. GPTQ](#NF4-GGML-GPTQ)**
- **[Kuantizasyon vs. Budama vs. Bilgi Aktarımı](#Kuantizasyon-vs-Pruning-vs-Bilgi-Distilasyonu)**
- **[Özet](#Özet)**


---

## Niceleme Türleri

1. **Eğitim Sonrası Kuantizasyon (PTQ)**  
   Halihazırda eğitilmiş bir modelin ağırlıklarının, herhangi bir yeniden eğitim gerektirmeksizin daha düşük bir hassasiyete dönüştürüldüğü basit bir tekniktir.  
   Uygulaması kolay olmasına rağmen, PTQ performans düşüşü riski taşır.

2. **Kuantizasyon Farkındalıklı Eğitim (QAT)**  
   Ağırlık dönüştürme işlemini ön eğitim veya ince ayar aşamasında entegre eder, bu da model performansını artırır. Ancak, QAT hesaplama açısından maliyetlidir ve temsili eğitim verileri gerektirir.

* **Sadece PTQ'ye odaklanacağız.**


---

## GPTQ Üretici Modellerde Eğitim Sonrası Niceleme

* **GPTQ**, yüz milyarlarca parametreye sahip modellerde bile uygulanabilecek kadar verimlidir ve bu modelleri parametre başına yalnızca 2, 3 veya 4 bite sıkıştırarak kayda değer bir doğruluk kaybı olmaksızın olağanüstü hassasiyet elde edebilir.
* **GPTQ'nun farkı**, karışık bir int4/fp16 kuantizasyon şeması benimsemesidir. Bu şemada, model ağırlıkları int4 olarak kuantize edilirken, aktivasyonlar float16 olarak korunur. Çıkarım sırasında ağırlıklar dinamik olarak dekuantize edilir ve gerçek hesaplamalar float16 olarak gerçekleştirilir.
* **GPTQ**, modelin tamamını belleğe yüklemeye gerek kalmadan modelleri kuantize edebilir. Bunun yerine, model modül modül kuantize edilerek kuantizasyon işlemi sırasında bellek gereksinimleri önemli ölçüde azaltılır.
* GPTQ, önce ağırlıklara skaler kuantizasyon uygular, ardından kalanlar üzerinde vektör kuantizasyonu gerçekleştirir.

### **GPTQ'yu Ne Zaman Kullanmalısınız?**

HuggingFace tarafından önerilen ve birçok modele uygulanan bir yaklaşım şudur:
1. Orijinal LLM modelini bitsandbytes ile 4-bit, nf4 ve QLoRa kullanarak verimli bir şekilde ince ayar yapın.
2. Adapter'ı orijinal modele entegre edin.
3. Ortaya çıkan modeli GPTQ 4-bit ile kuantize edin.


---

## AutoGPTQ Entegrasyonu
> **AutoGPTQ** kütüphanesi, verimli GPTQ yöntemini kullanarak Transformer modellerini kuantize etmek için güçlü bir araç olarak öne çıkmaktadır.

* **AutoGPTQ Avantajları:**
  - Kuantize edilmiş modeller seri hale getirilebilir ve Hub'da paylaşılabilir.
  - **GPTQ**, LLM'leri çalıştırmak için gereken bellek gereksinimlerini önemli ölçüde azaltırken, çıkarım gecikmesi FP16 çıkarımıyla eşdeğerdir.
  - **AutoGPTQ**, geniş bir mimari yelpazesi için **Exllama çekirdeklerini** destekler.
  - Entegrasyon, AMD GPU'lar için doğal **RoCm desteği** ile birlikte gelir.
  - **PEFT** ile ince ayar yapma imkanı sunar.

## AutoGPTQ Gereksinimleri
* Bağımlılıkları yükleme
```
!pip install -q -U transformers peft accelerate optimum
!pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
```
* Modelinizi kuantize etmek için birkaç argüman sağlamanız gerekir:
  - **bits:** Kuantize etmek istediğiniz bit sayısı.
  - **dataset:** Kuantizasyonu kalibre etmek için kullanılan veri seti.
  - **model_seqlen:** Veri setini işlemek için kullanılan modelin dizi uzunluğu.
  - **block_name_to_quantize:** Kuantize edilecek blok adı.
```
from optimum.gptq import GPTQQuantizer
quantizer = GPTQQuantizer(
      bits=4,
      dataset="c4",
      block_name_to_quantize = "model.decoder.layers",
      model_seqlen = 2048
)
```

* Modeli kaydet
```
save_folder = "/path/to/save_folder/"
quantizer.save(model,save_folder)
```

* Nicelenmiş ağırlıkları yükleyin
```
from accelerate import init_empty_weights
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
empty_model.tie_weights()
quantized_model = load_quantized_model(empty_model, save_folder=save_folder, device_map="auto")
```

* Daha hızlı çıkarım için Exllama çekirdekleri
> 4-bit model için daha hızlı çıkarım yapmak amacıyla exllama çekirdeklerini kullanabilirsiniz. Varsayılan olarak etkindir. Değerini değiştirmek isterseniz, load_quantized_model() işlevinde devre dışı_exllama'yı iletmeniz yeterlidir. Bu çekirdekleri kullanabilmek için modelin tamamının GPU'da olması gerekir.
```
quantized_model = load_quantized_model(
      empty_model,
      save_folder=save_folder,
      device_map="auto",
      disable_exllama=False
)
```




---
## GGML (Genelleştirilmiş Grafiksel Model Kitaplığı)
GGML, makine öğrenimi üzerine odaklanmış bir C kütüphanesidir. llama.cpp kütüphanesiyle birlikte kullanılmak üzere tasarlanmıştır.
* Kütüphane, Llama modellerinin verimli bir şekilde çıkarım yapılabilmesi için C/C++ dilinde yazılmıştır. GGML modellerini yükleyebilir ve bunları bir CPU üzerinde çalıştırabilir. Başlangıçta bu, GPTQ modelleriyle olan ana farkıydı, çünkü GPTQ modelleri bir GPU üzerinde yüklenip çalıştırılır.

### GGML ile Kuantizasyon
GGML'in ağırlıkları kuantize etme yöntemi, GPTQ'nun yöntemine göre daha karmaşık değildir. Temelde, değer blokları gruplandırılır ve daha düşük bir hassasiyete yuvarlanır. Q4_K_M ve Q5_K_M gibi bazı teknikler, kritik katmanlar için daha yüksek bir hassasiyet uygular. Bu durumda, her ağırlık 4-bit hassasiyetle saklanır, ancak dikkat.wv ve feed_forward.w2 tensörlerinin yarısı dışında.

* Deneysel olarak, bu karışık hassasiyet, doğruluk ve kaynak kullanımı arasında iyi bir denge sunduğu kanıtlanmıştır.
* Ağırlıklar, her biri 32 değer içeren bloklar halinde işlenir. Her blok için, en büyük ağırlık değerinden türetilen bir ölçek faktörü (delta) hesaplanır. Bloktaki tüm ağırlıklar ardından ölçeklendirilir, kuantize edilir ve verimli bir şekilde depolama için paketlenir (nibbles).
* Bu yaklaşım, depolama gereksinimlerini önemli ölçüde azaltırken, orijinal ve kuantize edilmiş ağırlıklar arasında nispeten basit ve deterministik bir dönüşüm sağlar.




---
## NF4, GGML, GPTQ
<kbd>
   <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*yz7rSvjKtukVXdVHwxGfAQ.png">
</kbd>

* Bu sonuçlara dayanarak, GGML modellerinin karmaşıklık (perplexity) açısından hafif bir avantaja sahip olduğunu söyleyebiliriz. Fark çok belirgin değildir, bu yüzden token/saniye cinsinden üretim hızına odaklanmak daha iyi olacaktır.
* En iyi teknik, GPU'nuzun özelliklerine bağlıdır: Eğer tüm kuantize edilmiş modelin sığabileceği kadar yeterli VRAM'e sahipseniz, GPTQ ve ExLlama ile olan yöntem en hızlı olacaktır. Eğer bu mümkün değilse, bazı katmanları yükleyip GGML modellerini llama.cpp ile kullanarak LLM'nizi çalıştırabilirsiniz.




--- 
## Kuantizasyon vs. Pruning vs. Bilgi Distilasyonu
## Pruning
> Sinir ağımızdaki bazı bağlantıları kaldırmak, bu da **seyrek ağ** (sparse network) ile sonuçlanır.
<kbd>
   <img src="https://miro.medium.com/v2/resize:fit:559/1*0mGJU7eNWgrqU5sgk-7RoQ.png">
</kbd>

* İşlem:
  * Pruning faktörü X'i seçin.
  * Her katmanda, ağırlıkların en düşük X%'ini sıfıra ayarlayın.
  * Opsiyonel: Doğruluğu geri kazanmak için modeli yeniden eğitin.


### Genel MatMul vs. Seyrek MatMul
<kbd>
   <img src="https://github.com/ElDokmak/LLMs/assets/85394315/c7a3526e-995b-43e7-8a99-6b1e43883826">
</kbd>

* Genel MatMul: Her satır ve sütun birbirleriyle çarpılır, sıfırlar olsa bile.
* Seyrek MatMul: Sıfır olan satır ve sütunlar atlanır, yalnızca değeri olanlar çarpılır.

## Distilasyon
> **Öğrenci modeli**'ni, **Öğretmen modeli**'nin çıktı etiketleri üzerinde eğitiriz.
<kbd>
   <img src="https://github.com/ElDokmak/LLMs/assets/85394315/160351cd-1a97-450d-b8f8-ab9a9dac65f0">
</kbd>

---
## Özet
<img src="https://github.com/ElDokmak/LLMs/assets/85394315/15cf3903-455c-4811-8f7e-14b6eaf30237">

---
