# English
# Content
* **[LoRA](#lora-low-rank-adaption-for-llms)**
* **[QLoRA](#qlora-quantized-llms-and-low-rank-adaption)**
* **[QA-LoRA](#qa-lora-quantization-aware-low-rank-adaptation)**
* **[LongLoRA](#LongLoRA-Long-Sequence-Low-Rank-Adaptation)**
* **[LoftQ](#loftq-lora-fine-tuning-aware-quantization)**
* **[NEFTune](#neftune-noisy-embeddings-improve-instruction-finetuning)**



---
## LoRA (Low Rank Adaption for LLMs)
LoRA is a training method that accelerates the training of large language models while consuming less memory. It adds pairs of trainable rank-decomposition weight matrices (Called Update matrices) to existing weights, and only trains those newly added added weights.
<kbd>
 <img width="600" src="https://images.ctfassets.net/xjan103pcp94/6fct47v2q8PU36X9A1TUzN/62bf8834293c1ec4a7e591f42ed1ffd1/pretrainined-weights-diagram-lora-blog.png">
</kbd>

* **Method:** The technique constrains the rank of the update matrix ΔW using its rank decomposition. It represents ΔWₙₖ as the product of 2 low-rank matrices Bₙᵣ and Aᵣₖ where r << min(n, k). This implies that the forward pass of the layer, originally Wx, is modified to Wx + BAx.

* A random Gaussian initialization is used for A and B is initially to 0, so BA=0 at the start of training. The update BA is additionally scaled with a factor α/r.

### **Advantages:**
 * Previos pretrained weights are kept frozen so the model is not as prone to catastrophic forgetting.
 * Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
<kbd>
 <img width="600" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*F7uWJePoMc6Qc1O2WxmQqQ.png">
</kbd>

> e.g. for the previos image: let's say that d=100 , k=100 and r=5 then the original matrix size is 100 * 100 = 10000 which means 10000 parameters.
> But after using Rank-decomposition matrices you have 100 * 5 = 500 and 5 * 100 = 500 which means 500 + 500 = 1000 parameters and that is a huge improvement which result in less computations.

* rank-decomposition weight matrices are generally added to the attention layers of the original model.
* The greater memory-efficiency allows you to run fine-tuning on consumer GPUs like the Tesla T4, RTX 4080 or even the RTX 3080 Ti! GPUs like the T4 are free and readily accessible in Kaggle or Google Colab notebooks.


### LoRA Implementation
**1. Load the model and tokenizer**
```
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    torch_dtype = torch.float16,
    device_map = 'auto'
)

tokenizer = AutoTokenizer.from_pretrained("model_name")
```

**2. LoRA configuration**
```
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8, # rank of update matrices
    lora_alpha = 16, # scaling factor
    target_modules=['query_key_value'], # The modules to apply the LoRA update matrices.
    lora_dropout = 0.05, # for regularization
    bias = 'none', # whether to train bias params or not
    task_type = 'CAUSAL_LM' # task of the model
)

model = get_peft_model(model, config)     
```
* For more about parameters selection check this [blog](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)

**3. Load dataset and create prompt template**
```
from datasets import load_dataset

dataset = load_dataset('sateset_name')

# This example is for Q&A task
def create_prompt(context, question, answer) :
  if len(answer['text']) < 1 :
    answer = 'Can not find answer'
  else :
    answer = answer['text'][0]

  prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}"
  return prompt_template

mapped_qa_dataset = qa_dataset.map(lambda samples: tokenizer(create_prompt(samples['context'], samples['question'], samples['answers'])))
```

**4. Training arguments and training**
```
import transformers
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir = 'outputs',
    learning_rate = 1e-5,
    num_train_epochs = 1,
    weight_decay = 0.01,
    logging_steps=1,
    max_steps = 100,
    per_device_train_batch_size=4
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = mapped_qa_dataset['train'],
    eval_dataset = mapped_qa_dataset['validation'],
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cashe = False

trainer.train()
```

**5. inference and push to hub (optional)**
```
HUGGING_FACE_USER_NAME = 'your hugging face username'
from huggingface_hub import notebook_login
notebook_login() # enter token
model_name = "mdoel_name"

model.push_to_hub(f"{HUGGING_FACE_USER_NAME}/{model_name}", use_auth_token=True)
```
```
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = f"{HUGGING_FACE_USER_NAME}/{model_name}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
qa_model = PeftModel.from_pretrained(model, peft_model_id)
```
> [!NOTE]
> You may encounter latency issues during inference due to separately loading the base model and the LoRA model.
> To eliminate latency, use the merge_and_unload() function to merge the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model.
>
> This works because during training, the smaller weight matrices (A and B in the diagram above) are separate. But once training is complete, the weights can actually be merged into a new weight matrix that is identical.
>
>> Use merge_adapter() to merge the LoRa layers into the base model while retaining the PeftModel. This will help in later unmerging, deleting, loading different adapters and so on.
>> 
>> Use unmerge_adapter() to unmerge the LoRa layers from the base model while retaining the PeftModel. This will help in later merging, deleting, loading different adapters and so on.
>>
>> Use unload() to get back the base model without the merging of the active lora modules. This will help when you want to get back the pretrained base model in some applications when you want to reset the model to its original state. For example, in Stable Diffusion WebUi, when the user wants to infer with base model post trying out LoRAs.
>>
>> Use delete_adapter() to delete an existing adapter.
>>
>> Use add_weighted_adapter() to combine multiple LoRAs into a new adapter based on the user provided weighing scheme.
>
> For more check [huggingface](https://huggingface.co/docs/peft/conceptual_guides/lora)



---
## QLoRA (Quantized LLMs and Low Rank Adaption)
<kbd>
 <img src="https://miro.medium.com/v2/resize:fit:1400/0*oV_KwvWnFYzuWzlz.png">
</kbd>

**QLoRA**, is a technique that helps in training and fine-tuning large language models (LLMs) on regular computers with limited memory. It addresses the challenge of memory requirements when working with these models.
* The key idea behind QLoRA is to make LLMs more efficient by reducing their memory usage while maintaining reliable performance. It achieves this through several steps: by introducing 4-bit quantization, a new data type called 4-bit NormalFloat (NF4), double quantization, and paged optimizers.
  - **4-bit quantization:**
    - 4-bit quantization of weights and apply PEFT, inject LoRA adapters in each layer in 32-bit precision, and start to fine-tune the complete Language model on a specific task, **for the quantized configuration to reduce the quantization error of the system.**
    - Perform additionally a mixed precision training to balance the trade-off between accuracy and speed/memory usage.
    - QLoRA has one storage data type (NF4) and a computation data type (FP16).
    - We dequantize the storage data type to the computation data type to perform the forward and backward pass, **but we only compute weight gradients for the LoRA parameters which use 16-bit BrainFloat.**
  - **Double Quantization:**
    - This is a technique that combines 4-bit quantization with 8-bit quantization to further reduce the memory footprint.
    - By applying a second quantization to the quantization constants, the memory footprint of these constants can be significantly reduced. 
  - **Paged Optimizers:**
    - This allows QLoRa to use more memory than is available on a single GPU by paging in and out data as needed.

### QLoRA Implementation
**1. Load the model and apply 4bit quantization**
```
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TrainingArguments

quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4", # as explained we use 4bit for the pretrained weights while using BF-16 for computations.
        bnb_4bit_compute_dtype = torch.float16,
)

model_name = "Enter your model name"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_safetensors = True,
    quantization_config = quantization_config,
    trust_remote_code = True,
    device_map = 'auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**2. LoRA configuration**
```
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha = 64,
    lora_dropout = 0.1,
    target_modules = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj",],
    bias = "none",
    task_type = "CAUSAL_LM",
)
```

**3. Load dataset and create prompt**
```
from datasets import load_dataset, Dataset

dataset = load_dataset()
```

**4. Training aruguments and trainer**
```
import transformers
from transformers import TrainingArguments
from trl import SFTTrainer


training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()
```

**5. Load model, inference and apply merge_and_unload() as discussed above**
```
from peft import AutoPeftModelForCausalLM

trained_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    low_cpu_mem_usage=True,
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")
```

### QLoRA summary
* The original pre-trained weights are quantized into 4-bit and kept frozen. Then a small number of trainable parameters in the form of low-rank adpters are introduced during fine-tuning. These adapters are trained to adapt the pre-trained model to specific task in FP16.
* When it comes to computations the 4-bit quantized weights are dequantized back to FP16.
* After fine-tuning the model consists of the original weights in 4-bit and the additional low-rank adapters in their higher precision format.
* The adpaters are in higher format for a few reasons:
  - Higher precision allows the model to capture more subtle patterns in the data. This is important for the low-rank adapters, as they are responsible for adapting the pre-trained model to the specific task it is being fine-tuned for.
  - Higher precision formats ensures that updates are accurately captured.
  - Computations with FP16 can be faster than with lower precision.
* QLoRA backpropagates gradients through a frozen, 4-bit quantized pretraining language model into low rank adpaters (LoRA).
* QLoRA added the adaption weights back to pre-trained weights and turned them into FP16 again, and thus the deployed model is still slow. We solve this problem with the proposed **QA-LoRA** approach.



---
## QA-LoRA (Quantization-Aware Low-Rank Adaptation)    
<kbd>
 <img src="https://pbs.twimg.com/media/F7lJvyUXIAAfzzB?format=jpg&name=large">
</kbd>

* The motivation behind QA-LoRA lies in the imbalanced degrees of freedom of quantization and adaptation. Specifically, each column of the pre-trained weight
matrix is accompanied by only one pair of scaling and zero parameters (explained [here](https://github.com/ElDokmak/LLMs/edit/main/Fine-Tuning/README.md#min-max-quantization)) but many more LoRA parameters. This imbalance not only results in large quantization errors but also makes it difficult to integrate the auxiliary weights into the main model.
  * **Solution:** is to use use **group-wise operators** which increase the degree of freedom of quantization meanwhile decreasing that of adaptation.

* QA-LoRA equips the original LoRA with two-fold abilities: **(i)** during fine-tuning, the LLM’s weights are quantized
(e.g., into INT4) to reduce time and memory usage; **(ii)** after fine-tuning, the
LLM and auxiliary weights are naturally integrated into a quantized model without loss of accuracy without the need for PTQ which often incurs loss of accuracy..

* To get things together **our goal is to have a final quantized model out of the fine-tuning process without the need of PTQ** like in (LoRA and QLoRA) which results in unsatisfying accuracy especially when
the quantization bit width is low.  

### The proposed approach
In LoRA the output y = (W + s · AB)<sup>⊤</sup>x, where W is replaced by W′ = W + s · AB for fast inference.
* A way to reduce computational costs lies in low-bit quantization. In particular, we apply a simple method named **min-max quantization**.

#### Min-Max quantization
> Mathematically, given the bit width N and a pre-trained weight matrix W,
we compute the minimum and maximum values across all elements of W, denoted as min(W) and
max(W), respectively. Then, W is quantized into W˜ by computing 

<kbd>
 <img width="400" src="https://github.com/ElDokmak/LLMs/assets/85394315/bff9d40b-fa4f-426a-85a5-07284dab0b73">
</kbd>

* where α = (max(W) − min(W))/(2N − 1) and β = min(W) are called the **scaling and zero
factors**, respectively; ⌊·⌉ denotes the integer rounding operation. All elements in Wˆ are in the set of
{0, 1, . . . , 2N −1} and thus stored as B-bit integers.

* The computation, y = W<sup>⊤</sup>x, is approximated as **y = W˜<sup>⊤</sup>x = α·⌊(W−β)/α⌉<sup>⊤</sup>x+βx.** The quantization brings two-fold benefits:
  - W is reduced (e.g., from FP16 to INT4) and the computation of W<sup>⊤</sup>x becomes faster.
  - The cost is that W˜ is an approximation of W, which may harm the accuracy.

> To reduce the quantization loss between W and W˜ , an effective way is to perform an individual quantization for each column of W.
> 
> Compared to the original (holistic) quantization, the computational cost is unchanged while the storage cost of the scaling and zero factors increases from 2 to 2Dout floating point numbers. 

### Main objective: 
**1. During fine-tuning, the pretrained weights W are converted to a low-bit format to allow LLMs to be fine-tuned using minimal GPUs.**   
**2. Post fine-tuning, the adjusted and combined weights W' remain in a quantized format for efficient LLM deployment.**

* QLoRA achieved the first goal by quantizing W from FP16 to NF4 during fine-tuning. This joint optimization of quantization and adaptation is feasible as the accuracy difference between W and W~ is offset by the low-rank weights s * AB. However, after fine-tuning, the side weights s * AB are reintegrated to W~, reverting the final weights W' to FP16. Post-training quantization on W' can lead to notable accuracy loss, especially with a low bit width.

* For the second goal as mentioned before the solution was to use: group-wise quantization with low-rank adaptation:
* The primary objective is to merge the quantized W~ and s * AB without resorting to high-precision numbers like FP16. In the original setting, this is unattainable due to the column-wise quantization of W and the unconstrained nature of matrices A and B.
  - The first idea of the authors requires all row vectors of A to be identical. However, this approach results in a significant accuracy drop since it limits the rank of AB to 1, affecting the fine-tuning ability.
  - To address this, the constraints for both quantization and adaptation are relaxed. W is partitioned into L groups, and individual scaling and zero factors are used for quantization within each group. This also requires row vectors of A within the same group to be identical. 
* QA-LoRA offers time and memory benefits. It requires extra storage for scaling and zero factors but reduces the parameters of A.

> [!NOTE]
> **Quantization Group Size:**
> Larger L implies larger degrees of freedom thus samller quantization loss, and a larger number of adaption parameters. But also requires a larger number of storage and computation.
>
> **Impact of fine-tuning datasets:**
> The smaller the dataset the weaker accuracy on MMLU.
>
> **Impact of the size of fine-tuning datasets:**
> Low-bit quantization asks for more data.

### QA-LoRA summary
* The original pre-trained weights are quantized into INT4.
* W is partitioned into L groups.
* Row vectors of A within the same group have to be identical. 
* Individual scaling and zero factors are used for quantization within each group.
* Summation within each group of the input vector, x.
* Parameter-free operation reduces the dimension of x from Din to L, hence we can set A to be a L × Dint
* After fine-tuning we have a quantized model without the need of PTQ.



---
## LongLoRA-Long Sequence Low-Rank Adaptation
An efficient fine-tuning approach that extends the context sizes of pre-trained large language models (LLMs), with limited computation cost

* LoRA modifies the linear projection layers in self-attention blocks by utilizing low-rank matrices, which are generally efficient and reduce the number of trainable parameters.
However, for long context this is neither sufficiently effective nor efficient.
  * In terms of effectiveness, plain low-rank adaptation results in a high perplexity in long context extension.
  * In terms of efficiency, regardless of whether LoRA is employed or not, computational cost increases dramatically as the context size expands, primarily due to the standard self-attention mechanism.

* The attention layer is the main bottlenech is scaling to longer sequences, as its runtime and memory increase quadratically is the sequence length. So LongLoRA proposed **S2-Attention (Shift Short Attention)** an effective approach that shifts the start and end points of the groups by half the group size, each shifted group overlaps with half of the previous group and half of the next group. This overlap ensures information exchange between adjacent groups. This shift ensures information exchange between adjacent groups. (Look at the following image)
<kbd>
 <img width="800" src="https://github.com/ElDokmak/LLMs/assets/85394315/8156fc08-ca82-4752-98fd-7bb963fd2e7b">
</kbd>

* Shift short attention involves three steps. First, it splits features along the head dimension into two chunks. Second, tokens in one of the chunks are shifted by half of the group size. Third, we split tokens into groups and reshape them into batch dimensions.
* Another approach that LongLoRA proposed was making embedding and normalization layers trainable.
<kbd>
 <img width="800" src="https://github.com/ElDokmak/LLMs/assets/85394315/89d980aa-a18d-445f-bfef-855c4a8e4237">
</kbd>

> [!NOTE]
> LoRA  + making embedding and normalization layers trainable = LoRA+
> 
> S2-Attention + LoRA+ = LongLoRA
> 
> FA2 (Flash attention 2) is also applied: It allows Transformers to handle longer sequences without using as much memory or taking as much time.
>> FA2 is a computation optimization, specifically designed to enhance GPU processing performance (Optimize process management inside GPU).

### Performance of LongLoRA
<kbd>
 <img width="800" src="https://github.com/ElDokmak/LLMs/assets/85394315/4fa5535d-8a2b-458d-acd2-ecb92a5c9a6c">
</kbd>

In term of Training hours LongLoRA has achieved significan improvement than LoRA and Full FT.

### Summary of key components of LongLoRA
1. S2-Attn (Shift Short Attention).
2. Maintaining original attention architecture.
3. Learnable Embeddings and Normalization layers.
4. Flash Attention.
5. LongLoRA uses RoPE (Rotary Position Embedding).



---
## LoftQ (LoRA Fine-Tuning aware Quantization)
> Recent researches has focused on compressing Large lagnuage models to make them more more efficient and practical to use, One popular approach is quantization (QAT, PTQ) which quantize the model and while retaining accuracy,as explained in Quantization part

LoftQ method improves on these by considering the subsequent "fine-tuning" process after quantization. It tries to quantize the model in a way that results in a good starting point for fine-tuning.

### How it works 
1. Given a pretrained model, LoftQ first quantizes the original weights W to quantized weights Q (in lower precision such as NF4).
2. There is some discrepancy between (difference) the original weight W and the quantized weights and the low rank approximated weights **Q+AB<sup>T</sup>**.
   * **Our goas is to minimize the discrepancy (difference) between the original weight W and the combined quantized and low-rank approximated weight Q+AB^T.**
     - The authors approached it as an **optimization** problem, formulated as a **Frobenius norm minimization** **min<sub>Q,A,B</sub>||W-Q-AB<sup>T</sup>||<sub>F</sub>**.
     - This loss function encapsulate a balance between storage efficiency through Q and compuational effiency through A and B.
     - The Echart-Young-Mirsky theorem states: the best low-rank approximation of a matrix in the Frobenius norm is given by the truncated singular value decompostion (SVD).
       - In other words the best way to approximate a matrix with a lower-rank is to keep the largest singular values and discard the smaller ones.
       - Singular values of a matrix are the non negative square roots of its eigen values.
 3. **Alternating Oprimization** is employed to update Q,A, and B, given the current estimates of A and B you find Q that minimzes the norm. Then given this new Q, SVD is used to find low-rank approximation A and B that minimize the residuals **W-Q**.
 4. The process then alternates - quantize the weights, do low-rank SVD on the residual, update the quantized + low-rank approximation. Over a few iterations, the quantized initialization converges closer and closer to the original weights.
 5. Finally, the optimized Q, A, and B are used to for LoRA fine-tuning. During this phase, Q is kept frozen, and only A and B are subject to change via optimization algorithms like AdamW. 

### Results
* LoftQ is shown to outperform existing SOTA methods like QLoRA on NLU and NLG datasets, especially at very low precision like 2-bits.
* Works well with different quantization techniques like uniform quantization or NormalFloat.



---
## NEFTune (Noisy Embeddings Improve Instruction Finetuning) 
NEFTune is a new technique used to boost the performance of chat models.

* **NEFTune** introduces **stochastic noise**, this noise is added to the input embeddings before they are fed to the model's transformers layers. This noise can be dran from various distributions, such as uniform, gaussian.
  - Introducing noise can **smooth** the optimization, the **noise can help the model escape poor local minima.**
* This technique is **similar to dropout** but in **different spaces**:
  - **Dropout:** in neurons space
  - **NEFTune:** in embeddings space
* **NEFTune** helps **generalize** better to unseen data.
> Notice the following image to see difference between Overfitting vs. Dropout vs. NEFTune
<kbd>
 <img src="https://github.com/ElDokmak/LLMs/assets/85394315/734e3f47-4eb8-40b9-8119-e43703642c4d">
</kbd>

### How to code NEFTune
Use SFTTrainer from hugging face
```
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    neftune_noise_alpha=5,
)
trainer.train()
```

# Türkçe

# İçerik
* **[LoRA](#LoRA-LLMler-için-Düşük-Dereceli-Uyarlama)**
* **[QLoRA](#QLoRA-Niceltilmiş-LLMler-ve-Düşük-Rank-Uyarlaması)**
* **[QA-LoRA](#QA-LoRA-Nicemelendirme-Duyarlı-Düşük-Sıralı-Adaptasyon)**
* **[LongLoRA](#LongLoRA-Uzun-Sekanslı-Düşük-Rank-Adaptasyonu)**
* **[LoftQ](#LoftQ-LoRA-İnce-Ayarına-Duyarlı-Nicemelendirme)**
* **[NEFTune](#NEFTune-Gürültülü-Gömülü-Temsiller-Talimat-İnce-Ayarını-İyileştirir)**

## LoRA (LLM'ler için Düşük Dereceli Uyarlama)
LoRA, büyük dil modellerinin eğitimini hızlandıran ve daha az bellek tüketen bir eğitim yöntemidir. Mevcut ağırlıklara çiftler halinde eğitilebilir dereceli ayrışım ağırlık matrisleri (Güncelleme matrisleri olarak adlandırılır) ekler ve yalnızca bu yeni eklenen ağırlıkları eğitir.
<kbd>
 <img width="600" src="https://images.ctfassets.net/xjan103pcp94/6fct47v2q8PU36X9A1TUzN/62bf8834293c1ec4a7e591f42ed1ffd1/pretrainined-weights-diagram-lora-blog.png">
</kbd>

* **Yöntem:** Bu teknik, güncelleme matrisi ΔW'nin derecesini, dereceli ayrışım kullanarak sınırlar. ΔWₙₖ, 2 düşük dereceli matris olan Bₙᵣ ve Aᵣₖ'nin çarpımı olarak ifade edilir; burada r << min(n, k). Bu, katmanın ileri geçişini (forward pass) başlangıçta Wx iken, Wx + BAx olarak değiştirir.

* A ve B için rastgele bir Gauss başlangıcı kullanılır ve B başlangıçta 0'a eşitlenir, böylece eğitim başlangıcında BA=0 olur. Güncelleme BA ayrıca bir α/r faktörü ile ölçeklendirilir.

### **Avantajlar:**
 * Önceden eğitilmiş ağırlıklar donuk (sabit) tutulur, bu da modelin felaketvi unutmaya daha az eğilimli olmasını sağlar.
 * Derece-ayrışım matrisleri, orijinal modelden önemli ölçüde daha az parametreye sahiptir, bu da eğitilmiş LoRA ağırlıklarının kolayca taşınabilir olduğu anlamına gelir.
<kbd>
 <img width="600" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*F7uWJePoMc6Qc1O2WxmQqQ.png">
</kbd>

> Örneğin, önceki görsel için: diyelim ki d=100, k=100 ve r=5. Bu durumda orijinal matris boyutu 100 * 100 = 10000 olur, yani 10000 parametre. 
> Ancak Rank-ayrışım matrisleri kullandıktan sonra 100 * 5 = 500 ve 5 * 100 = 500 elde edersiniz, yani 500 + 500 = 1000 parametre. Bu, hesaplama açısından büyük bir iyileştirme sağlar.

* Derece-ayrışım ağırlık matrisleri genellikle orijinal modelin dikkat (attention) katmanlarına eklenir.
* Daha yüksek bellek verimliliği, Tesla T4, RTX 4080 veya hatta RTX 3080 Ti gibi tüketici GPU'larında ince ayar çalıştırmanızı sağlar! T4 gibi GPU'lar ücretsizdir ve Kaggle veya Google Colab not defterlerinde kolayca erişilebilir durumdadır.

### LoRA Uygulaması
**1. Model ve Tokenizer'ı Yükle**
```
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    torch_dtype = torch.float16,
    device_map = 'auto'
)

tokenizer = AutoTokenizer.from_pretrained("model_name")
```

**2. LoRA Yapılandırması**
```
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8, # Güncelleme matrislerinin sıralaması
    lora_alpha = 16, # scaling factor
    target_modules=['query_key_value'], # LoRA güncelleme matrislerini uygulamak için modüller.
    lora_dropout = 0.05, # # Düzenleme için
    bias = 'none', # Bias parametrelerinin eğitimini yapıp yapmamaya karar verme
    task_type = 'CAUSAL_LM' # Modelin görevi
)

model = get_peft_model(model, config)     
```

* Parametre seçimi hakkında daha fazla bilgi için bu [blog](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2) kontrol edin.

**3. Veri kümesini yükleyin ve prompt şablonu oluşturun**
```
from datasets import load_dataset

dataset = load_dataset('sateset_name')

# This example is for Q&A task
def create_prompt(context, question, answer) :
  if len(answer['text']) < 1 :
    answer = 'Can not find answer'
  else :
    answer = answer['text'][0]

  prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}"
  return prompt_template

mapped_qa_dataset = qa_dataset.map(lambda samples: tokenizer(create_prompt(samples['context'], samples['question'], samples['answers'])))
```

**4. Eğitim argümanları ve eğitim**
```
import transformers
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir = 'outputs',
    learning_rate = 1e-5,
    num_train_epochs = 1,
    weight_decay = 0.01,
    logging_steps=1,
    max_steps = 100,
    per_device_train_batch_size=4
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = mapped_qa_dataset['train'],
    eval_dataset = mapped_qa_dataset['validation'],
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cashe = False

trainer.train()
```

**5. Çıkarım ve hub'a yükleme (isteğe bağlı)**
```
HUGGING_FACE_USER_NAME = 'hugging face kullanıcı adınız'
from huggingface_hub import notebook_login
notebook_login() # token girin
model_name = "model_adı"

model.push_to_hub(f"{HUGGING_FACE_USER_NAME}/{model_name}", use_auth_token=True)
```
```
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = f"{HUGGING_FACE_USER_NAME}/{model_name}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# LoRA modelini yükle
qa_model = PeftModel.from_pretrained(model, peft_model_id)
```

> [!NOT]
> Çıkarım sırasında, temel model ve LoRA modelinin ayrı ayrı yüklenmesi nedeniyle gecikme sorunlarıyla karşılaşabilirsiniz.
> Gecikmeyi ortadan kaldırmak için, adapter ağırlıklarını temel modelle birleştirmek için merge_and_unload() fonksiyonunu kullanarak, yeni birleştirilmiş modeli bağımsız bir model olarak etkili bir şekilde kullanabilirsiniz.
> 
> Bu, eğitim sırasında daha küçük ağırlık matrislerinin (yukarıdaki diyagramda A ve B) ayrı olmasından kaynaklanır. Ancak eğitim tamamlandığında, ağırlıklar aslında yeni bir ağırlık matrisine birleştirilebilir ve bu matris aynıdır.
> 
>> LoRa katmanlarını temel modelle birleştirmek için merge_adapter() kullanın, böylece PeftModel korunur. Bu, daha sonra birleştirme, silme, farklı adapter'lar yükleme vb. işlemlerinde yardımcı olur.
>> 
>> LoRa katmanlarını temel modelden ayırmak için unmerge_adapter() kullanın, böylece PeftModel korunur. Bu, daha sonra birleştirme, silme, farklı adapter'lar yükleme vb. işlemlerinde yardımcı olur.
>>
>> unload() kullanarak, aktif LoRa modüllerinin birleştirilmesi olmadan temel modeli geri alabilirsiniz. Bu, bazı uygulamalarda modelin orijinal durumuna sıfırlanması gerektiğinde, örneğin, LoRA'ları deneyip temel modelle çıkarım yapmak isteyen kullanıcılar için faydalıdır.
>>
>> delete_adapter() kullanarak mevcut bir adapter'ı silebilirsiniz.
>>
>> add_weighted_adapter() kullanarak, kullanıcı tarafından sağlanan ağırlıklandırma şemasına göre birden fazla LoRA'yı yeni bir adapter olarak birleştirebilirsiniz.
>
> Daha fazla bilgi için [huggingface](https://huggingface.co/docs/peft/conceptual_guides/lora) kontrol edin.

---
## QLoRA (Niceltilmiş LLM'ler ve Düşük Rank Uyarlaması)
<kbd>
 <img src="https://miro.medium.com/v2/resize:fit:1400/0*oV_KwvWnFYzuWzlz.png">
</kbd>

**QLoRA**, sınırlı belleğe sahip normal bilgisayarlarda büyük dil modellerini (LLM'ler) eğitmek ve ince ayar yapmak için kullanılan bir tekniktir. Bu teknik, bu modellerle çalışırken bellek gereksinimleri sorununu ele alır.
* QLoRA'nın arkasındaki temel fikir, LLM'leri daha verimli hale getirerek bellek kullanımını azaltırken güvenilir performansı korumaktır. Bunu, birkaç adım aracılığıyla başarır: 4-bit kuantizasyon, 4-bit NormalFloat (NF4) adı verilen yeni bir veri türü, çift kuantizasyon ve sayfalı optimizasyonlar.

  - **4-bit kuantizasyon:**
    - Ağırlıkların 4-bit kuantizasyonunu gerçekleştirir ve PEFT uygular, her katmanda LoRA adaptörlerini 32-bit hassasiyetle enjekte eder ve dil modelini belirli bir görev için ince ayar yapmaya başlar, **kuantize yapılandırmanın sistemin kuantizasyon hatasını azaltmasına yardımcı olur.**
    - Ayrıca doğruluk ve hız/bellek kullanımı arasındaki dengeyi sağlamak için karışık hassasiyetli eğitim yapar.
    - QLoRA'nın bir depolama veri türü (NF4) ve bir hesaplama veri türü (FP16) vardır.
    - Depolama veri türünü hesaplama veri türüne dekuantize ederek ileri ve geri geçişi gerçekleştiririz, **ancak yalnızca 16-bit BrainFloat kullanan LoRA parametreleri için ağırlık gradyanlarını hesaplarız.**
  
  - **Çift Kuantizasyon:**
    - Bu, 4-bit kuantizasyonu 8-bit kuantizasyonu ile birleştiren bir tekniktir ve bellek ayak izini daha da azaltır.
    - Kuantizasyon sabitlerine ikinci bir kuantizasyon uygulayarak, bu sabitlerin bellek ayak izi önemli ölçüde azaltılabilir.
  
  - **Sayfalı Optimizasyonlar:**
    - Bu, QLoRA'nın tek bir GPU'da mevcut olandan daha fazla bellek kullanmasını sağlar, çünkü veriyi gerektiği şekilde sayfalar halinde içerip dışarı alır.

### QLoRA Uygulaması
**1. Modeli yükleyin ve 4-bit kuantizasyon uygulayın**
```
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TrainingArguments

quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
)

model_name = "Enter your model name"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_safetensors = True,
    quantization_config = quantization_config,
    trust_remote_code = True,
    device_map = 'auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**2. LoRA yapılandırması**
```
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha = 64,
    lora_dropout = 0.1,
    target_modules = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj",],
    bias = "none",
    task_type = "CAUSAL_LM",
)
```

**3. Load dataset and create prompt**

```
from datasets import load_dataset, Dataset

dataset = load_dataset()
```

**4. Eğitim argümanları ve eğitmen**

```
import transformers
from transformers import TrainingArguments
from trl import SFTTrainer


training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()
```

**5. Modeli yükle, çıkarım yap ve yukarıda tartışıldığı gibi merge_and_unload() uygulayın**
```
from peft import AutoPeftModelForCausalLM

trained_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    low_cpu_mem_usage=True,
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")
```
### QLoRA Özeti
* Orijinal önceden eğitilmiş ağırlıklar 4-bit'e kuantize edilip dondurulur. Ardından, düşük sıralı adaptörler şeklinde az sayıda eğitim parametresi, ince ayar sırasında tanıtılır. Bu adaptörler, önceden eğitilmiş modeli belirli bir göreve uyarlamak için FP16 formatında eğitilir.
* Hesaplamalar söz konusu olduğunda, 4-bit kuantize edilmiş ağırlıklar FP16'ya geri kuantize edilir.
* İnce ayar sonrasında model, 4-bit'lik orijinal ağırlıklar ve ek düşük sıralı adaptörlerden oluşur ve bunlar yüksek hassasiyetli formatta bulunur.
* Adaptörler yüksek hassasiyetli formattadır, çünkü birkaç nedenden dolayı:
  - Yüksek hassasiyet, modelin verideki daha ince desenleri yakalamasına olanak tanır. Bu, düşük sıralı adaptörler için önemlidir, çünkü bunlar, önceden eğitilmiş modeli, belirli bir görev için ince ayar yapmakla sorumludur.
  - Yüksek hassasiyet formatı, güncellemelerin doğru bir şekilde yakalanmasını sağlar.
  - FP16 ile yapılan hesaplamalar, daha düşük hassasiyetle yapılanlara göre daha hızlı olabilir.
* QLoRA, gradyanları, donmuş 4-bit kuantize edilmiş ön eğitim dil modelinden düşük sıralı adaptörlere (LoRA) geri yayar.
* QLoRA, adaptasyon ağırlıklarını tekrar önceden eğitilmiş ağırlıklara ekleyip bunları tekrar FP16'ya dönüştürür ve bu nedenle dağıtılan model hala yavaş olur. Bu sorunu, önerilen **QA-LoRA** yaklaşımıyla çözüyoruz.

---
## QA-LoRA (Nicemelendirme-Duyarlı Düşük Sıralı Adaptasyon)
<kbd>
 <img src="https://pbs.twimg.com/media/F7lJvyUXIAAfzzB?format=jpg&name=large">
</kbd>

* QA-LoRA'nın motivasyonu, kuantizasyon ve adaptasyonun dengesiz serbestlik derecelerinden kaynaklanmaktadır. Özellikle, önceden eğitilmiş ağırlık matrisinin her bir sütunu yalnızca bir çift ölçekleme ve sıfır parametresi ile birlikte gelir (bu [burada](https://github.com/ElDokmak/LLMs/edit/main/Fine-Tuning/README.md#min-max-quantization) açıklanmıştır), ancak çok daha fazla LoRA parametresi bulunur. Bu dengesizlik, yalnızca büyük kuantizasyon hatalarına yol açmakla kalmaz, aynı zamanda yardımcı ağırlıkların ana modele entegrasyonunu da zorlaştırır.
  * **Çözüm:** **grup bazlı operatörler** kullanmaktır. Bu, kuantizasyonun serbestlik derecesini artırırken adaptasyonun serbestlik derecesini azaltır.

* QA-LoRA, orijinal LoRA'yı iki yönlü yetenekle donatır: **(i)** ince ayar sırasında, LLM'nin ağırlıkları (örneğin, INT4'e) kuantize edilir, böylece zaman ve bellek kullanımı azaltılır; **(ii)** ince ayar sonrası, LLM ve yardımcı ağırlıklar doğruluk kaybı olmadan, PTQ'ya ihtiyaç duymadan, kuantize edilmiş bir modelde doğal olarak entegrasyon sağlanır.

* İşleri bir araya getirmek için **bizim hedefimiz, LoRA ve QLoRA'da olduğu gibi, düşük kuantizasyon bit genişliğinde doğruluk kaybı olmadan** ince ayar sürecinden çıkan son bir kuantize edilmiş model elde etmektir. Bu, özellikle kuantizasyon bit genişliği düşük olduğunda tatmin edici doğruluk elde edilmesini sağlar.

### Önerilen Yaklaşım
LoRA'da çıktı y = (W + s · AB)<sup>⊤</sup>x şeklinde ifade edilir, burada W, hızlı çıkarım için W′ = W + s · AB ile değiştirilir.
* Hesaplama maliyetlerini azaltmanın bir yolu, düşük bit kuantizasyonudur. Özellikle, **min-max kuantizasyonu** adında basit bir yöntem uygularız.

#### Min-Max Kuantizasyonu
> Matematiksel olarak, verilen bit genişliği N ve önceden eğitilmiş ağırlık matrisi W için, W'nin tüm elemanları arasındaki minimum ve maksimum değerler hesaplanır ve sırasıyla min(W) ve max(W) olarak gösterilir. Ardından, W, şu şekilde hesaplanarak W˜'ye kuantize edilir:

<kbd>
 <img width="400" src="https://github.com/ElDokmak/LLMs/assets/85394315/bff9d40b-fa4f-426a-85a5-07284dab0b73">
</kbd>

* Burada α = (max(W) − min(W))/(2N − 1) ve β = min(W) **ölçekleme ve sıfır faktörleri** olarak adlandırılır; ⌊·⌉, tam sayıya yuvarlama işlemini belirtir. Wˆ içindeki tüm elemanlar {0, 1, . . . , 2N −1} kümesindedir ve bu nedenle B-bit tam sayılar olarak saklanır.

* Hesaplama, y = W<sup>⊤</sup>x, şu şekilde yaklaşık olarak hesaplanır: **y = W˜<sup>⊤</sup>x = α·⌊(W−β)/α⌉<sup>⊤</sup>x+βx.** Kuantizasyonun iki katmanlı faydaları vardır:
  - W azaltılır (örneğin, FP16'dan INT4'e) ve W<sup>⊤</sup>x hesaplaması daha hızlı hale gelir.
  - Maliyet ise, W˜'nin W'nin bir yakınsaması olmasıdır, bu da doğruluğu olumsuz etkileyebilir.

> W ve W˜ arasındaki kuantizasyon kaybını azaltmak için etkili bir yöntem, W'nin her sütunu için bireysel kuantizasyon yapmaktır.
> 
> Orijinal (bütünsel) kuantizasyona kıyasla, hesaplama maliyeti değişmezken, ölçekleme ve sıfır faktörlerinin depolama maliyeti 2'den 2Dout kayan nokta sayısına çıkar.

### Ana hedef: 
**1. İnce ayar sırasında, önceden eğitilmiş ağırlıklar W, LLM'lerin minimal GPU'larla ince ayar yapabilmesi için düşük bit formatına dönüştürülür.**  
**2. İnce ayar sonrası, ayarlanmış ve birleştirilmiş ağırlıklar W', verimli LLM dağıtımı için kuantize edilmiş formatta kalır.**

* QLoRA, ilk hedefi, W'yi FP16'dan NF4'e kuantize ederek başardı. Kuantizasyon ve adaptasyonun bu ortak optimizasyonu, W ve W~ arasındaki doğruluk farkının, düşük ranklı ağırlıklar s * AB ile telafi edilmesi sayesinde mümkündür. Ancak, ince ayar sonrası, yan ağırlıklar s * AB, W~'ye yeniden entegre edilerek nihai ağırlıklar W'yi FP16'ya döndürür. W' üzerinde yapılan eğitim sonrası kuantizasyon, özellikle düşük bit genişliğinde, belirgin doğruluk kaybına yol açabilir.

* İkinci hedef için, daha önce bahsedildiği gibi çözüm şuydu: düşük ranklı adaptasyonla grup bazlı kuantizasyon kullanmak:
* Temel amaç, kuantize edilmiş W~ ve s * AB'yi yüksek hassasiyetli sayılar (örneğin FP16) kullanmadan birleştirmektir. Orijinal ayarda, bu, W'nin sütun bazında kuantize edilmesi ve A ile B matrislerinin kısıtlanmamış yapısı nedeniyle mümkün değildir.
  - Yazarların ilk fikri, A'nın tüm satır vektörlerinin aynı olmasını gerektirir. Ancak bu yaklaşım, AB'nin rankını 1 ile sınırladığı için büyük bir doğruluk kaybına yol açar ve ince ayar yeteneğini olumsuz etkiler.
  - Bunun çözülmesi için, hem kuantizasyon hem de adaptasyon için kısıtlamalar gevşetilmiştir. W, L gruba ayrılır ve her grup içindeki kuantizasyon için bireysel ölçekleme ve sıfır faktörleri kullanılır. Bu, aynı grup içindeki A'nın satır vektörlerinin de aynı olmasını gerektirir.
* QA-LoRA, zaman ve bellek faydaları sunar. Ölçekleme ve sıfır faktörleri için ekstra depolama gerektirir ancak A'nın parametrelerini azaltır.

> [!NOT]
> **Kuantizasyon Grup Boyutu:**
> Daha büyük L, daha büyük serbestlik dereceleri anlamına gelir, dolayısıyla daha küçük kuantizasyon kaybı ve daha fazla adaptasyon parametresi gerektirir. Ancak aynı zamanda daha fazla depolama ve hesaplama gerektirir.
>
> **İnce Ayar Verisetlerinin Etkisi:**
> Verisetinin boyutu ne kadar küçükse, MMLU'daki doğruluk o kadar zayıftır.
>
> **İnce Ayar Verisetlerinin Boyutunun Etkisi:**
> Düşük-bit kuantizasyon daha fazla veri gerektirir.

### QA-LoRA Özeti
* Orijinal önceden eğitilmiş ağırlıklar INT4 formatında kuantize edilir.
* W, L gruba ayrılır.
* Aynı grup içindeki A'nın satır vektörlerinin aynı olması gerekir.
* Her grup içinde kuantizasyon için bireysel ölçekleme ve sıfır faktörleri kullanılır.
* Giriş vektörü x'in her grup içinde toplamı yapılır.
* Parametresiz işlem, x'in boyutunu Din'den L'ye indirger, bu nedenle A'yı L × Dint olarak ayarlayabiliriz.
* İnce ayar sonrasında PTQ'ya ihtiyaç duymadan kuantize edilmiş bir model elde edilir.



---
## LongLoRA (Uzun Sekanslı Düşük-Rank Adaptasyonu)
Önceden eğitilmiş büyük dil modellerinin (LLM'ler) bağlam boyutlarını genişleten verimli bir ince ayar yaklaşımı, sınırlı hesaplama maliyeti ile

* LoRA, düşük dereceli matrisler kullanarak kendiliğinden dikkat katmanlarındaki doğrusal projeksiyon katmanlarını değiştirir. Bu, genellikle verimlidir ve eğitilebilir parametre sayısını azaltır.
Ancak, uzun bağlamlar için bu yeterince etkili ve verimli değildir.
  * Etkililik açısından, düz düşük dereceli adaptasyon uzun bağlam genişlemesinde yüksek bir karmaşıklığa (perplexity) yol açar.
  * Verimlilik açısından, LoRA kullanılıp kullanılmadığına bakılmaksızın, bağlam boyutu genişledikçe hesaplama maliyeti dramatik şekilde artar, bu da esasen standart kendiliğinden dikkat mekanizmasından kaynaklanır.

* Dikkat katmanı, daha uzun dizilere ölçeklendirilmesinde ana darboğazdır, çünkü çalışma süresi ve bellek, dizi uzunluğu ile karekök oranında artar. Bu nedenle LongLoRA, **S2-Dikkat (Shift Short Attention)** adlı etkili bir yaklaşımı önerir; bu yaklaşım, grupların başlangıç ve bitiş noktalarını grup boyutunun yarısı kadar kaydırarak her kaydırılmış grubun, önceki grubun ve sonraki grubun yarısı ile örtüşmesini sağlar. Bu örtüşme, komşu gruplar arasında bilgi alışverişini garanti eder. Bu kaydırma, komşu gruplar arasında bilgi alışverişini sağlar. (Aşağıdaki görsele bakınız)
<kbd>
 <img width="800" src="https://github.com/ElDokmak/LLMs/assets/85394315/8156fc08-ca82-4752-98fd-7bb963fd2e7b">
</kbd>

* Shift short attention üç adımdan oluşur. İlk olarak, özellikler başlık boyutu boyunca iki parçaya ayrılır. İkinci olarak, bu parçalardan birindeki token'lar, grup boyutunun yarısı kadar kaydırılır. Üçüncü olarak, token'lar gruplara ayrılır ve bunlar batch boyutlarına yeniden şekillendirilir.
* LongLoRA'nın önerdiği bir diğer yaklaşım ise, gömme (embedding) ve normalizasyon katmanlarının eğitilebilir hale getirilmesidir.
<kbd>
 <img width="800" src="https://github.com/ElDokmak/LLMs/assets/85394315/89d980aa-a18d-445f-bfef-855c4a8e4237">
</kbd>

> [!NOT]
> LoRA + gömme (embedding) ve normalizasyon katmanlarını eğitilebilir hale getirme = LoRA+
>
> S2-Attention + LoRA+ = LongLoRA
>
> FA2 (Flash attention 2) de uygulanır: Bu, Transformers'ın daha uzun dizileri daha az bellek kullanarak ve daha az zaman harcayarak işleyebilmesini sağlar.
>> FA2, özellikle GPU işleme performansını artırmak için tasarlanmış bir hesaplama optimizasyonudur (GPU içindeki işlem yönetimini optimize eder).

### LongLoRA Performansı
<kbd>
 <img width="800" src="https://github.com/ElDokmak/LLMs/assets/85394315/4fa5535d-8a2b-458d-acd2-ecb92a5c9a6c">
</kbd>

LongLoRA, eğitim saatleri açısından LoRA ve Tam Fine-Tuning'e göre önemli bir gelişim kaydetmiştir.

### LongLoRA'nın Temel Bileşenlerinin Özeti
1. S2-Attn (Shift Short Attention).
2. Orijinal dikkat mimarisinin korunması.
3. Öğrenilebilir Gömme (Embedding) ve Normalizasyon katmanları.
4. Flash Attention.
5. LongLoRA, RoPE (Rotary Position Embedding) kullanır.




---
## LoftQ (LoRA İnce Ayarına Duyarlı Nicemelendirme)
> Son araştırmalar, büyük dil modellerini sıkıştırarak daha verimli ve pratik hale getirmeye odaklanmıştır. Popüler bir yaklaşım, modeli nicemlemek (QAT, PTQ) ve doğruluğu koruyarak modelin boyutunu küçültmektir, bu da Nicemeleme (Quantization) bölümünde açıklanmıştır.

LoftQ yöntemi, nicemeleme işleminden sonra gelen "fine-tuning" sürecini dikkate alarak bu yaklaşımı geliştirir. Modeli, fine-tuning için iyi bir başlangıç noktası sağlayacak şekilde nicemeler.

### Nasıl Çalışır
1. Önceden eğitilmiş bir model verildiğinde, LoftQ ilk olarak orijinal ağırlıkları W, düşük hassasiyetle (örneğin NF4) kuantize edilmiş ağırlıklara Q'ya dönüştürür.
2. Orijinal ağırlık W ile kuantize edilmiş ağırlıklar arasındaki fark ve düşük sıralı yaklaşık ağırlıklar **Q+AB<sup>T</sup>** arasında bazı tutarsızlıklar vardır.
   * **Amacımız, orijinal ağırlık W ile birleşik kuantize edilmiş ve düşük sıralı yaklaşık ağırlık Q+AB<sup>T</sup> arasındaki farkı en aza indirmektir.**
     - Yazarlar bunu bir **optimizasyon** problemi olarak ele almışlar ve bunu **Frobenius normu minimizasyonu** olarak formüle etmişlerdir: **min<sub>Q,A,B</sub>||W-Q-AB<sup>T</sup>||<sub>F</sub>**.
     - Bu kayıp fonksiyonu, Q ile depolama verimliliği ve A ve B ile hesaplama verimliliği arasında bir dengeyi kapsar.
     - Echart-Young-Mirsky teoremi şöyle der: Bir matrisin en iyi düşük sıralı yaklaşıklaması, Frobenius normunda kesilmiş tekil değer ayrıştırması (SVD) ile verilir.
       - Başka bir deyişle, bir matrisi daha düşük sıralı bir şekilde yaklaşıklamanın en iyi yolu, en büyük tekil değerleri tutmak ve daha küçük olanları atmak olacaktır.
       - Bir matrisin tekil değerleri, özdeğerlerinin negatif olmayan karekökleridir.
3. **Alternatif Optimizasyon** yöntemi, Q, A ve B'yi güncellemek için kullanılır; A ve B'nin mevcut tahminlerine dayalı olarak Q, normu minimize edecek şekilde bulunur. Ardından bu yeni Q ile, kalanları **W-Q** minimize eden düşük sıralı yaklaşıklamayı bulmak için SVD kullanılır.
4. Süreç daha sonra alternatiftir - ağırlıkları kuantize et, kalanı üzerinde düşük sıralı SVD yap, kuantize edilmiş + düşük sıralı yaklaşıklamayı güncelle. Birkaç iterasyon boyunca, kuantize başlangıcı orijinal ağırlıklara daha yakın hale gelir.
5. Son olarak, optimize edilmiş Q, A ve B LoRA ince ayarını yapmak için kullanılır. Bu aşamada Q sabit tutulur ve yalnızca A ve B, AdamW gibi optimizasyon algoritmaları ile değiştirilebilir.

### Sonuçlar
* LoftQ'nun, özellikle 2-bit gibi çok düşük hassasiyetlerde, QLoRA gibi mevcut SOTA yöntemlerini NLU ve NLG veri setlerinde geride bıraktığı gösterilmiştir.
* Farklı kuantizasyon teknikleriyle, örneğin uniform kuantizasyon veya NormalFloat ile iyi çalışır.


---
## NEFTune (Gürültülü Gömülü Temsiller, Talimat İnce Ayarını İyileştirir)
NEFTune, sohbet modellerinin performansını artırmak için kullanılan yeni bir tekniktir.

* **NEFTune**, **stokastik gürültü** ekler; bu gürültü, modelin transformer katmanlarına verilmeden önce giriş gömme vektörlerine eklenir. Bu gürültü, uniform, gaussian gibi farklı dağılımlardan türetilebilir.
  - Gürültü eklemek, **optimizasyonu yumuşatmaya** yardımcı olabilir, **gürültü, modelin kötü yerel minimumlardan kaçmasına yardımcı olur.**
* Bu teknik, **dropout** ile **benzer** fakat **farklı alanlarda** uygulanır:
  - **Dropout:** nöronlar alanında
  - **NEFTune:** gömme (embeddings) alanında
* **NEFTune**, **görülmemiş verilere** daha iyi genelleme yapmaya yardımcı olur.
> Aşağıdaki görsele bakarak Overfitting, Dropout ve NEFTune arasındaki farkı görebilirsiniz.
<kbd>
 <img src="https://github.com/ElDokmak/LLMs/assets/85394315/734e3f47-4eb8-40b9-8119-e43703642c4d">
</kbd>

### NEFTune nasıl kodlanır
Hugging Face'ten SFTTrainer kullanın
```
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    neftune_noise_alpha=5,
)
trainer.train()
```