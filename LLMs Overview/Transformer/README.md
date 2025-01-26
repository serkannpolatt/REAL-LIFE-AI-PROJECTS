# English
# Transformers
<img src="https://miro.medium.com/v2/resize:fit:1400/1*10K7SmGoJ5zAtjkGfNfjkg.png">

---
## Content
* **[Input Embeddings](#Embeddings)**
* **[Postitional Encoding](#Postional-Encoding)**
* **[Self Attention](#Self-Attention)**
* **[Multi-Head Attention](#Multi-Head-Attention)**
* **[Layer Normalization/ Residual Connections/ Feed Forward Network](#layer-normalization-residual-connections-feed-forward-network)**
* **[Encoder](#Encoder)**
* **[Decoder](#Decoder)**



---
## Embeddings
Word Embedding can be thought of as a learned vector representation of each word. A vector which captures contextual information of these words.
* Neural networks learn through numbers so each word maps to a vector with continuous values to represent that word.
<kbd>
  <img width=300 src="https://miro.medium.com/v2/resize:fit:582/format:webp/0*6MnniQMOBPu4kFq3.png">
</kbd>



---
## Postional Encoding
Embedding represents token in a d-dimensional space where tokens with similar meaning are closer to each other. However, these embeddings don't encode the relative position of the tokens in a sentence.
* Same as the name Postional Encodding encodes the postion of the words in the sequence.
* Formula for calculating the positional encoding is :
<img width="500" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*z3Rfl0wGsBsaZcprpqR8Uw.png">

* Postional Encoding works because absolute position is less important than relative position.
  > e.g. "My fried chicken was very good" 
  > We do not know that the word "good" is at index 6 and the word “looks” is at index 5. It’s sufficient to remember that the word “good” tends to follows the word “looks”.



---
## Self Attention (Scaled Dot-Product Attention)
<kbd>
  <img width="500" src="https://miro.medium.com/v2/resize:fit:640/0*NEPbOP47PlMTXoIb">
</kbd>

After feeding the query, key, and value vectors through a linear layer, we calculate the dot product of the query and key vectors. 
The values in the resulting matrix determine how much attention should be payed to the other words in the sequence given the current word.
* In other words, each word (row) will have an attention score for every other word (column) in the sequence.
> e.g. "On the river bank"
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*8BFdH5nY0KoLq1I8">
</kbd>

* For short, you can think of attention as which part of the sentence should I focus on.
* The dot-product is scaled by a square root of the depth. This is done because for large values of depth, the dot product grows large in magnitude pushing the softmax function where it has small gradients which make it difficult to learn.
  
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*WmNb5ugFkawpvSqYqvTaNg.png">
</kbd>

* Then apply softmax to obtain values between 0 and 1.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*KgHYTtBIJcn8sNUZ">
</kbd>

* Then multiply by the value vector.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:828/0*G-mOxnggdLyNlc8y">
</kbd>



---
## Multi-Head Attention
Instead of one single head attention, Q, K, and V are split into multiple heads. Which allows the model to jointly attend to information from different representation subspaces at different positions.

> e.g. "On the river bank" for the first head the word "The" will attend the word "bank" while for the second head the word "The" will attend to the word "river"
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*Nc9uK_gEwm18OrNmwP_lBg.png">
</kbd>

* **Note:** After splitting the total computation ramains the same as a single head.

* The attention output is concatenated and put through a Dense layer.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*j_1WMIQxKPh-L7OY">
</kbd>



---
## Layer Normalization/ Residual Connections/ Feed Forward Network
Residual Connections = Input Embedding + Positional Encodding are added to Multi-Head Attention.

* Normalization means having Mean = 0 and variance = 1.
* Residual Connections help avoiding the vanishing gradient problem in deep nets.
* Each hidden layer has a residual connection around it followed by a layer normalization.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*21NPCniNISCCVfxn">
</kbd>

* Then the ouput finishes by passing through a point wise feed forward network.
<kbd>
<img src="https://miro.medium.com/v2/resize:fit:622/format:webp/1*ItvJ0KeOKCFSXDUNce2zAA.png">
</kbd>



---
## Encoder
The Encoder's job is to map all input sequences into an abstract continous representation that holds information.
* You can stack the encoder N times to further encode the information, where each layer has the opportunity to learn different attention representations therefore potentially boosting the predictive power of the transformer network.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*K67VOXrh_xgyCiHS">
</kbd>



---
## Decoder
The Decoder's job is to generate text sequences.

<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*lx0m8R-k0dwq7sy3">
</kbd>

* It has two multi-headed attention layers, a pointwise feed-forward layer, and residual connections, and layer normalization after each sub-layer.
* These sub-layers behave similarly to the layers in the encoder but each multi-headed attention layer has a different job.
* The decoder is capped off with a linear layer that acts as a classifier, and a softmax to get the word probabilities.
* The deocoder is autoregressive, it begins with a start token, and takes previos output as inputs, as well as the encoder outputs that contain attention information from the input.
* Decoder's Input Embeddings & Positional Encoding is almost the same as the Encoder.


### Decoder's Multi-Head Attention
The second one operates just like the Encoder while the first one operates slight different than the Encoder since the decoder is autoregressive  and generates the seq word by word, you need to prevent it from conditioning to future tokens using **Masking**.
* Mask is a matrix that’s the same size as the attention scores filled with values of 0’s and negative infinities. When you add the mask to the scaled attention scores, you get a matrix of the scores, with the top right triangle filled with negativity infinities.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/0*QYFua-iIKp5jZLNT.png">
</kbd>

* The reason for the mask is because once you take the softmax of the masked scores, the negative infinities get zeroed out, leaving zero attention scores for future tokens.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/0*3ykVCJ9okbgB0uUR.png">
</kbd>


### Linear Classifier and Final Softmax
* The output of the final pointwise feedforward layer goes through a final linear layer, that acts as a classifier.
* The output of the classifier then gets fed into a softmax layer, which will produce probability scores between 0 and 1. We take the index of the highest probability score, and that equals our predicted word.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:786/format:webp/0*1OyVUO-s-uBh8EV2.png">
</kbd>

* The decoder then takes the output, add’s it to the list of decoder inputs, and continues decoding again until a token is predicted. For our case, the highest probability prediction is the final class which is assigned to the end token.
* The decoder can also be stacked N layers high, each layer taking in inputs from the encoder and the layers before it. By stacking the layers, the model can learn to extract and focus on different combinations of attention from its attention heads, potentially boosting its predictive power.


# Türkçe
# Transformerlar
<img src="https://miro.medium.com/v2/resize:fit:1400/1*10K7SmGoJ5zAtjkGfNfjkg.png">

---
## İçerik
* **[Girdi Gömme (Input Embeddings)](#Embeddings-Gömmeller)**
* **[Pozisyonel Kodlama (Postional Encoding)](#Postional-Encoding)**
* **[Kendi Kendine Dikkat (Self Attention)](#Self-Attention)**
* **[Çoklu Başlık Dikkat (Multi-Head Attention)](#Multi-Head-Attention)**
* **[Katman Normalizasyonu / Artık Bağlantılar / Besleme İleri Ağı (Layer Normalization/Residual Connections/Feed Forward Network)](#layer-normalization-residual-connections-feed-forward-network)**
* **[Encoder](#Encoder)**
* **[Decoder](#Decoder)**



---
## Embeddings (Gömmeller)
Kelime Gömme, her kelimenin öğrenilmiş bir vektör temsili olarak düşünülebilir. Bu vektör, bu kelimelerin bağlamsal bilgisini yakalar.
* Sinir ağları sayılarla öğrenir, bu nedenle her kelime, o kelimeyi temsil etmek için sürekli değerlere sahip bir vektöre haritalanır.
<kbd>
  <img width=300 src="https://miro.medium.com/v2/resize:fit:582/format:webp/0*6MnniQMOBPu4kFq3.png">
</kbd>

* Pozisyonel Kodlama (Postional Encoding), mutlak pozisyonun göreceli pozisyondan daha az önemli olduğu için çalışır.
  > örneğin, "My fried chicken was very good"
  > "Good" kelimesinin 6. indekste ve "looks" kelimesinin 5. indekste olduğunu bilmemize gerek yoktur. Yeterli olan, "good" kelimesinin genellikle "looks" kelimesini takip ettiğini hatırlamaktır.

---
## Self Attention (Scaled Dot-Product Attention)
<kbd>
  <img width="500" src="https://miro.medium.com/v2/resize:fit:640/0*NEPbOP47PlMTXoIb">
</kbd>

Sorgu (query), anahtar (key) ve değer (value) vektörlerini bir doğrusal katmandan geçirerek, sorgu ve anahtar vektörlerinin nokta çarpımını hesaplarız. 
Elde edilen matrisin içindeki değerler, mevcut kelimeye bağlı olarak dizideki diğer kelimelere ne kadar dikkat edilmesi gerektiğini belirler.
* Diğer bir deyişle, her kelime (satır) dizideki her diğer kelimeye (sütun) karşı bir dikkat puanına sahip olacaktır.
> örneğin, "On the river bank"
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*8BFdH5nY0KoLq1I8">
</kbd>

* Kısacası, dikkati cümlenin hangi kısmına odaklanmam gerektiği olarak düşünebilirsiniz.
* Nokta-çarpan, derinliğin karekökü ile ölçeklendirilir. Bu, derinlik değerleri büyük olduğunda, nokta çarpımının büyüklüğünün artması ve softmax fonksiyonunun küçük gradyanlar üretmesi nedeniyle öğrenmeyi zorlaştırmamak için yapılır.

<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*WmNb5ugFkawpvSqYqvTaNg.png">
</kbd>

* Ardından, 0 ile 1 arasında değerler elde etmek için softmax uygulanır.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*KgHYTtBIJcn8sNUZ">
</kbd>

* Ardından, değer vektörüyle çarpılır.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:828/0*G-mOxnggdLyNlc8y">
</kbd>




---
## Çoklu Başlık Dikkat (Multi-Head Attention)
Tek bir başlık dikkat yerine, Q, K ve V birden fazla başlığa bölünür. Bu, modelin farklı konumlardaki farklı temsil altuzaylarından gelen bilgilere aynı anda dikkat etmesini sağlar.

> örneğin, "On the river bank" cümlesinde, birinci başlık için "The" kelimesi "bank" kelimesine dikkat ederken, ikinci başlık için "The" kelimesi "river" kelimesine dikkat eder.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*Nc9uK_gEwm18OrNmwP_lBg.png">
</kbd>

* **Not:** Başlıkları böldükten sonra, toplam hesaplama aynı kalır, tıpkı tek bir başlık gibi.

* Dikkat çıktısı birleştirilir ve bir Dense katmandan geçirilir.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*j_1WMIQxKPh-L7OY">
</kbd>



---
## Katman Normalizasyonu / Artık Bağlantılar / Besleme İleri Ağı (Feed Forward Network)
Artık Bağlantılar = Girdi Gömmesi (Input Embedding) + Pozisyonel Kodlama (Positional Encoding), Çoklu Başlık Dikkat (Multi-Head Attention) ile eklenir.

* Normalizasyon, Ortalama = 0 ve varyans = 1 olma durumudur.
* Artık Bağlantılar, derin ağlarda kaybolan gradyan problemiyle mücadele etmeye yardımcı olur.
* Her gizli katmanın etrafında bir artık bağlantı bulunur ve ardından bir katman normalizasyonu yapılır.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*21NPCniNISCCVfxn">
</kbd>

* Ardından, çıktı, nokta bazında bir besleme ileri ağı (feed forward network) üzerinden geçerek tamamlanır.
<kbd>
<img src="https://miro.medium.com/v2/resize:fit:622/format:webp/1*ItvJ0KeOKCFSXDUNce2zAA.png">
</kbd>



---
## Encoder
Encoder'ın görevi, tüm giriş dizilerini bilgi taşıyan soyut sürekli bir temsile dönüştürmektir.
* Encoder'ı N kez üst üste yığarak bilgiyi daha fazla kodlayabilirsiniz, her katman farklı dikkat temsilleri öğrenme fırsatına sahip olduğundan, bu da transformer ağının tahmin gücünü artırabilir.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*K67VOXrh_xgyCiHS">
</kbd>



---
## Decoder
Decoder'ın görevi, metin dizileri üretmektir.

<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*lx0m8R-k0dwq7sy3">
</kbd>

* İki çoklu başlık dikkat katmanı, bir nokta bazında besleme ileri katmanı, ve her alt katmandan sonra artık bağlantılar ve katman normalizasyonu vardır.
* Bu alt katmanlar, encoder'daki katmanlarla benzer şekilde çalışır, ancak her çoklu başlık dikkat katmanının farklı bir görevi vardır.
* Decoder, bir sınıflandırıcı gibi çalışan doğrusal bir katman ve kelime olasılıklarını elde etmek için softmax ile tamamlanır.
* Decoder, otoregresif bir yapıdır, bir başlangıç tokeni ile başlar ve önceki çıktıyı girdi olarak alır, ayrıca girişten gelen dikkat bilgisini içeren encoder çıktıları da kullanılır.
* Decoder'ın Girdi Gömmeleri ve Pozisyonel Kodlaması, Encoder ile hemen hemen aynıdır.

### Decoder'ın Çoklu Başlık Dikkati (Multi-Head Attention)
İkinci katman, Encoder gibi çalışırken, birinci katman Encoder'dan biraz farklı çalışır çünkü decoder otoregresif bir yapıdır ve diziyi kelime kelime üretir, bu nedenle gelecekteki token'lara koşullanmaması için **Masking** kullanmanız gerekir.
* Maske, dikkat puanlarıyla aynı boyutta, 0'lar ve negatif sonsuzluklarla dolu bir matristir. Maskeyi ölçeklendirilmiş dikkat puanlarına eklediğinizde, puanların bir matrisi elde edersiniz, sağ üst üçgen negatif sonsuzluklarla doldurulur.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/0*QYFua-iIKp5jZLNT.png">
</kbd>

* Maskenin nedeni, maskelenmiş puanların softmax'ını aldığınızda, negatif sonsuzlukların sıfırlanmasıdır, böylece gelecekteki token'lar için sıfır dikkat puanları kalır.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/0*3ykVCJ9okbgB0uUR.png">
</kbd>


### Doğrusal Sınıflandırıcı ve Son Softmax
* Son nokta bazında besleme ileri katmanının çıktısı, bir sınıflandırıcı olarak işlev gören son doğrusal bir katmandan geçer.
* Sınıflandırıcının çıktısı daha sonra bir softmax katmanına iletilir, bu katman 0 ile 1 arasında olasılık puanları üretir. En yüksek olasılık puanının indeksini alırız ve bu, tahmin ettiğimiz kelimeye eşittir.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:786/format:webp/0*1OyVUO-s-uBh8EV2.png">
</kbd>

* Decoder daha sonra çıktıyı alır, bunu decoder girişleri listesine ekler ve bir token tahmin edilene kadar tekrar çözümleme işlemine devam eder. Bizim durumumuzda, en yüksek olasılık tahmini, son token'a atanan nihai sınıftır.
* Decoder ayrıca N katman yüksekliğinde yığılabilir, her katman encoder'dan ve önceki katmanlardan giriş alır. Katmanları yığarak, model dikkat başlıklarından farklı dikkat kombinasyonlarını çıkarmayı ve bunlara odaklanmayı öğrenebilir, bu da tahmin gücünü artırabilir.
