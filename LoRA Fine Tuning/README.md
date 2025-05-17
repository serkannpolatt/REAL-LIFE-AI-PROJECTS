# LoRA Fine-Tuning for TinyLlama Models

This repository contains code for fine-tuning and evaluating TinyLlama models using Low-Rank Adaptation (LoRA).

## Overview

The project includes the following components:

-   **Fine-Tuning Scripts:**
    -   `Fine-Tuning_TinyLlama_Frobinate.ipynb`: Fine-tunes TinyLlama on a custom dataset ("frobinate").
    -   `Fine-Tuning_TinyLlama_Math.ipynb`: Fine-tunes TinyLlama on the GSM8k math dataset.
-   **Evaluation Scripts:**
    -   `Evaluation_TinyLlama_Frobinate.ipynb`: Evaluates the fine-tuned model on the "frobinate" dataset.
    -   `Evaluation_TinyLlama_Math.ipynb`: Evaluates the fine-tuned model on the GSM8k math dataset.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install torch datasets transformers peft
    ```

2.  **Download Datasets:**

    -   For "frobinate", ensure you have `frobinate.jsonl` and `frobinate_test.jsonl` in the project directory.
    -   For GSM8k, the dataset will be automatically downloaded by the `datasets` library.

## Usage

### Fine-Tuning

1.  Run the desired fine-tuning script:

    ```bash
    jupyter notebook Fine-Tuning_TinyLlama_Frobinate.ipynb
    # or
    jupyter notebook Fine-Tuning_TinyLlama_Math.ipynb
    ```

2.  Follow the instructions in the notebook to fine-tune the model.

### Evaluation

1.  Run the desired evaluation script:

    ```bash
    jupyter notebook Evaluation_TinyLlama_Frobinate.ipynb
    # or
    jupyter notebook Evaluation_TinyLlama_Math.ipynb
    ```

2.  Follow the instructions in the notebook to evaluate the model.

## Details

-   **LoRA Configuration:** The LoRA configuration is set in the fine-tuning scripts with rank `r=8` and alpha `alpha=16`.
    More information about LoRA can be found in the [LoRA paper](https://arxiv.org/abs/2106.09685).
-   **Datasets:**
    -   "frobinate" is a custom dataset (ensure the `.jsonl` files are available).
    -   GSM8k is loaded from the `openai/gsm8k` dataset.
-   **Models:** TinyLlama models are loaded from the Hugging Face Model Hub.

## Türkçe Açıklama

# TinyLlama Modelleri için LoRA ile İnce Ayar

Bu depo, TinyLlama modellerini Düşük Dereceli Adaptasyon (LoRA) kullanarak ince ayarlamak ve değerlendirmek için kod içerir.

## Genel Bakış

Proje aşağıdaki bileşenleri içerir:

-   **İnce Ayar Betikleri:**
    -   `Fine-Tuning_TinyLlama_Frobinate.ipynb`: TinyLlama modelini özel bir veri kümesi ("frobinate") üzerinde ince ayarlar.
    -   `Fine-Tuning_TinyLlama_Math.ipynb`: TinyLlama modelini GSM8k matematik veri kümesi üzerinde ince ayarlar.
-   **Değerlendirme Betikleri:**
    -   `Evaluation_TinyLlama_Frobinate.ipynb`: İnce ayarlı modeli "frobinate" veri kümesi üzerinde değerlendirir.
    -   `Evaluation_TinyLlama_Math.ipynb`: İnce ayarlı modeli GSM8k matematik veri kümesi üzerinde değerlendirir.

## Kurulum

1.  **Bağımlılıkları Yükleyin:**

    ```bash
    pip install torch datasets transformers peft
    ```

2.  **Veri Kümelerini İndirin:**

    -   "frobinate" için, `frobinate.jsonl` ve `frobinate_test.jsonl` dosyalarının proje dizininde bulunduğundan emin olun.
    -   GSM8k için, veri kümesi `datasets` kütüphanesi tarafından otomatik olarak indirilecektir.

## Kullanım

### İnce Ayar

1.  İstenilen ince ayar betiğini çalıştırın:

    ```bash
    jupyter notebook Fine-Tuning_TinyLlama_Frobinate.ipynb
    # veya
    jupyter notebook Fine-Tuning_TinyLlama_Math.ipynb
    ```

2.  Modeli ince ayarlamak için not defterindeki talimatları izleyin.

### Değerlendirme

1.  İstenilen değerlendirme betiğini çalıştırın:

    ```bash
    jupyter notebook Evaluation_TinyLlama_Frobinate.ipynb
    # veya
    jupyter notebook Evaluation_TinyLlama_Math.ipynb
    ```

2.  Modeli değerlendirmek için not defterindeki talimatları izleyin.

## Detaylar

-   **LoRA Yapılandırması:** LoRA yapılandırması, `r=8` rankı ve `alpha=16` değeri ile ince ayar betiklerinde ayarlanmıştır.
    LoRA hakkında daha fazla bilgiyi [LoRA makalesinde](https://arxiv.org/abs/2106.09685) bulabilirsiniz.
-   **Veri Kümeleri:**
    -   "frobinate" özel bir veri kümesidir (`.jsonl` dosyalarının mevcut olduğundan emin olun).
    -   GSM8k, `openai/gsm8k` veri kümesinden yüklenir.
-   **Modeller:** TinyLlama modelleri, Hugging Face Model Hub'dan yüklenir.