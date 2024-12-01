# English
# AI-Powered RSS Feed Management System: Database Schema and Initialization

This document outlines the database schema and initialization process for an AI-powered RSS feed management system. The system collects and processes RSS feed data, allowing users to follow feeds, categorize items using AI-based recommendations, and interact with articles. This schema is designed to support scalable data storage for feed management and user interaction tracking, with AI features for content categorization and personalization.

## Purpose

This system aims to deliver personalized and organized content to users based on their interests in a constantly changing content landscape. AI models automatically categorize data collected from RSS feeds, recommending the most relevant content to users. Additionally, the system learns user preferences over time, analyzing these preferences to improve suggestions.

### Key Features:
- **RSS Feed Tracking**: Users can follow RSS feeds from their favorite websites.
- **AI-Powered Categorization**: AI models analyze articles and automatically categorize them into appropriate sections.
- **Personalized Recommendations**: Personalized content recommendations are made based on user interests.
- **Article Interaction Analysis**: The system tracks how users interact with articles (likes, shares, etc.) and continuously improves its recommendation algorithm.

## Who is It For?

This system is useful for various types of users:

1. **Individuals**: Those who want to follow content related to their interests, such as technology, news, finance, or science.
2. **Content Managers**: Content managers or publishers who follow and distribute content from multiple sources to their audiences.
3. **Researchers**: Academics or professional researchers who continuously gather information on specific topics via RSS feeds.
4. **Digital Marketers**: Marketers who use the system to stay updated with the most relevant and timely content to drive their campaigns.

## Real-World Applications

This type of system can be applied in the following areas:

1. **Personalized News Applications**: Users can access customized news feeds based on their interests. For example, tech enthusiasts can only receive technology-related news.
   
2. **Automated Content Categorization**: Large news agencies or blog platforms can quickly categorize their content with the help of AI. This reduces manual labor and enables more efficient content management.

3. **Information Flow Management**: Large companies or research organizations can monitor rapidly changing information sources in one place, ensuring they stay competitive with up-to-date insights.

4. **Marketing and Content Distribution**: Marketing professionals can automatically deliver engaging content to specific audiences, allowing them to run personalized campaigns based on their needs.

## Database Tables and Structure

### `rss_feeds`
- **Purpose:** Stores metadata about various RSS feeds that the system monitors.
- **Columns:**
  - `id` (Primary Key): Unique identifier for each feed.
  - `name`: The name of the RSS feed.
  - `url`: The source URL for fetching feed updates.
  - `description`: A short description of the feed.
  - `site_link`: Link to the original site hosting the feed.
  - `language`: The language of the feed content.

### `rss_items`
- **Purpose:** Contains the individual items/articles retrieved from the RSS feeds.
- **Columns:**
  - `id` (Primary Key): Unique identifier for each item.
  - `feed_id` (Foreign Key): Links the item to its corresponding feed.
  - `title`: The title of the RSS item.
  - `link`: URL to the full article or item.
  - `description`: A brief summary of the item.
  - `content`: Full content of the article (if available).
  - `author`: Author information.
  - `published_date`: Date and time the article was published.
  - `ai_generated_tags`: Automatically generated tags using an AI model for better categorization.

### `categories`
- **Purpose:** Defines various categories used to classify RSS items, either manually or using AI-based classification.
- **Columns:**
  - `id` (Primary Key): Unique identifier for each category.
  - `name`: The name of the category (e.g., AI, Machine Learning, Ethics).

### `rss_item_categories`
- **Purpose:** Links `rss_items` to `categories` in a many-to-many relationship.
- **Columns:**
  - `item_id` (Foreign Key): Links to the `rss_items` table.
  - `category_id` (Foreign Key): Links to the `categories` table.
  - `confidence_score`: Confidence score for AI-generated categorization (if applicable).

### `users`
- **Purpose:** Stores user account information, enabling personalized recommendations and tracking interactions.
- **Columns:**
  - `id` (Primary Key): Unique user identifier.
  - `username`: Username for login.
  - `email`: Email address.
  - `password_hash`: Hashed password for authentication.
  - `created_at`: Timestamp for when the user registered.
  - `last_login`: Timestamp for the user's last login.

### `user_category_preferences`
- **Purpose:** Tracks user preferences for specific categories to tailor recommendations.
- **Columns:**
  - `user_id` (Foreign Key): Links to the `users` table.
  - `category_id` (Foreign Key): Links to the `categories` table.
  - `preference_level`: Indicates the user's interest level in the category (e.g., High, Medium, Low).

### `user_feed_preferences`
- **Purpose:** Tracks which RSS feeds each user follows.
- **Columns:**
  - `user_id` (Foreign Key): Links to the `users` table.
  - `feed_id` (Foreign Key): Links to the `rss_feeds` table.
  - `subscription_date`: Timestamp when the user subscribed to the feed.

### `article_interactions`
- **Purpose:** Logs user interactions with articles, which can be analyzed to improve recommendations using machine learning algorithms.
- **Columns:**
  - `user_id` (Foreign Key): Links to the `users` table.
  - `item_id` (Foreign Key): Links to the `rss_items` table.
  - `interaction_type`: Type of interaction (e.g., View, Like, Share).
  - `interaction_time`: Timestamp of when the interaction occurred.

### `feed_views`
- **Purpose:** Logs when users view specific RSS feeds.
- **Columns:**
  - `user_id` (Foreign Key): Links to the `users` table.
  - `feed_id` (Foreign Key): Links to the `rss_feeds` table.
  - `view_time`: Timestamp of the view event.

### `user_sessions`
- **Purpose:** Tracks active user sessions to manage login states and ensure security.
- **Columns:**
  - `user_id` (Foreign Key): Links to the `users` table.
  - `session_token`: Unique token for the session.
  - `expiration_time`: Expiration time for the session token.


# Türkçe
# AI Destekli RSS Akış Yönetim Sistemi: Veritabanı Şeması ve Başlangıç

Bu belge, AI destekli bir RSS akış yönetim sistemi için veritabanı şemasını ve başlatma sürecini detaylandırmaktadır. Sistem, kullanıcıların RSS akışlarını takip etmelerine, AI tabanlı önerilerle öğeleri kategorize etmelerine ve makalelerle etkileşime girmelerine olanak tanır. Bu şema, ölçeklenebilir veri depolama ve kullanıcı etkileşim takibi için tasarlanmıştır ve içerik kategorileştirme ile kişiselleştirme için AI özellikleri içerir.

## Amacı

Bu sistem, sürekli değişen içerik dünyasında kullanıcılara ilgi alanlarına göre kişiselleştirilmiş ve organize edilmiş içerik sunmayı amaçlamaktadır. AI modelleri, RSS akışlarından toplanan verileri otomatik olarak kategorize eder ve kullanıcılara en uygun içerikleri önerir. Ayrıca sistem, zamanla kullanıcı tercihlerini öğrenerek bu tercihleri analiz eder ve önerilerini geliştirir.

### Temel Özellikler:
- **RSS Akış Takibi**: Kullanıcılar, favori sitelerinden RSS akışlarını takip edebilir.
- **AI Destekli Kategorileştirme**: AI modelleri makaleleri analiz eder ve uygun kategorilere otomatik olarak yerleştirir.
- **Kişiselleştirilmiş Öneriler**: Kullanıcıların ilgi alanlarına göre kişiselleştirilmiş içerik önerileri sunulur.
- **Makale Etkileşim Analizi**: Sistem, kullanıcıların makalelerle nasıl etkileşimde bulunduğunu (beğenme, paylaşma vb.) takip eder ve öneri algoritmasını sürekli iyileştirir.

## Kimler İçin Uygundur?

Bu sistem çeşitli kullanıcı grupları için faydalıdır:

1. **Bireyler**: Teknoloji, haber, finans veya bilim gibi ilgi alanlarına yönelik içerikleri takip etmek isteyenler.
2. **İçerik Yöneticileri**: Birden fazla kaynaktan içerik toplayarak bunu hedef kitlesine dağıtan içerik yöneticileri veya yayıncılar.
3. **Araştırmacılar**: Sürekli olarak belirli konular hakkında bilgi toplayan akademisyenler veya profesyonel araştırmacılar.
4. **Dijital Pazarlamacılar**: Sistemden en güncel ve alakalı içerikleri kullanarak pazarlama kampanyaları oluşturan pazarlamacılar.

## Gerçek Hayatta Uygulamalar

Bu tip bir sistem aşağıdaki alanlarda kullanılabilir:

1. **Kişiselleştirilmiş Haber Uygulamaları**: Kullanıcılar, ilgi alanlarına göre kişiselleştirilmiş haber akışlarına erişebilirler. Örneğin, teknoloji meraklıları sadece teknoloji ile ilgili haberler alabilir.
   
2. **Otomatik İçerik Kategorileştirme**: Büyük haber ajansları veya blog platformları, içeriklerini AI yardımıyla hızlıca kategorize edebilir. Bu, manuel iş gücünü azaltır ve içerik yönetimini daha verimli hale getirir.

3. **Bilgi Akışı Yönetimi**: Büyük şirketler veya araştırma kuruluşları, hızla değişen bilgi kaynaklarını tek bir yerde izleyerek güncel kalabilir ve rekabet avantajı elde edebilirler.

4. **Pazarlama ve İçerik Dağıtımı**: Pazarlama profesyonelleri, belirli hedef kitlelere ilgi çekici içerikler otomatik olarak sunabilir ve bu sayede ihtiyaçlarına göre kişiselleştirilmiş kampanyalar yürütebilirler.

## Veritabanı Tabloları ve Yapısı

### `rss_feeds`
- **Amaç:** Sistem tarafından izlenen çeşitli RSS akışları hakkında meta verileri depolar.
- **Sütunlar:**
  - `id` (Birincil Anahtar): Her akış için benzersiz tanımlayıcı.
  - `name`: RSS akışının adı.
  - `url`: Akış güncellemelerinin kaynağı olan URL.
  - `description`: Akış hakkında kısa açıklama.
  - `site_link`: Akışı barındıran orijinal siteye bağlantı.
  - `language`: Akış içeriğinin dili.

### `rss_items`
- **Amaç:** RSS akışlarından alınan bireysel öğeleri/makaleleri içerir.
- **Sütunlar:**
  - `id` (Birincil Anahtar): Her öğe için benzersiz tanımlayıcı.
  - `feed_id` (Yabancı Anahtar): Öğeyi ilgili akışla ilişkilendirir.
  - `title`: RSS öğesinin başlığı.
  - `link`: Tam makaleye veya öğeye bağlantı.
  - `description`: Öğenin kısa özeti.
  - `content`: Makalenin tam içeriği (varsa).
  - `author`: Yazar bilgisi.
  - `published_date`: Makalenin yayınlandığı tarih ve saat.
  - `ai_generated_tags`: AI modeli kullanılarak oluşturulmuş otomatik etiketler.

### `categories`
- **Amaç:** RSS öğelerinin manuel veya AI tabanlı sınıflandırma ile sınıflandırıldığı çeşitli kategorileri tanımlar.
- **Sütunlar:**
  - `id` (Birincil Anahtar): Her kategori için benzersiz tanımlayıcı.
  - `name`: Kategorinin adı (örneğin, AI, Makine Öğrenimi, Etik).

### `rss_item_categories`
- **Amaç:** `rss_items` ile `categories` tablolarını birçok-çok ilişkisinde bağlar.
- **Sütunlar:**
  - `item_id` (Yabancı Anahtar): `rss_items` tablosuna bağlar.
  - `category_id` (Yabancı Anahtar): `categories` tablosuna bağlar.
  - `confidence_score`: AI tarafından üretilen kategorileştirme için güven puanı (varsa).

### `users`
- **Amaç:** Kullanıcı hesap bilgilerini depolar, kişiselleştirilmiş öneriler ve etkileşim takibini sağlar.
- **Sütunlar:**
  - `id` (Birincil Anahtar): Benzersiz kullanıcı tanımlayıcı.
  - `username`: Giriş için kullanıcı adı.
  - `email`: E-posta adresi.
  - `password_hash`: Kimlik doğrulama için şifrelenmiş parola.
  - `created_at`: Kullanıcının kayıt olduğu zaman damgası.
  - `last_login`: Kullanıcının son giriş yaptığı zaman damgası.

### `user_category_preferences`
- **Amaç:** Kullanıcıların belirli kategorilere olan tercihlerini izler ve önerileri buna göre kişiselleştirir.
- **Sütunlar:**
  - `user_id` (Yabancı Anahtar): `users` tablosuna bağlar.
  - `category_id` (Yabancı Anahtar): `categories` tablosuna bağlar.
  - `preference_level`: Kullanıcının kategorideki ilgi düzeyini gösterir (örneğin, Yüksek, Orta, Düşük).

### `user_feed_preferences`
- **Amaç:** Her kullanıcının hangi RSS akışlarını takip ettiğini izler.
- **Sütunlar:**
  - `user_id` (Yabancı Anahtar): `users` tablosuna bağlar.
  - `feed_id` (Yabancı Anahtar): `rss_feeds` tablosuna bağlar.
  - `subscription_date`: Kullanıcının akışa abone olduğu zaman damgası.

### `article_interactions`
- **Amaç:** Makalelerle kullanıcı etkileşimlerini kaydeder ve bu veriler, makine öğrenimi algoritmaları kullanılarak önerilerin iyileştirilmesi için analiz edilir.
- **Sütunlar:**
  - `user_id` (Yabancı Anahtar): `users` tablosuna bağlar.
  - `item_id` (Yabancı Anahtar): `rss_items` tablosuna bağlar.
  - `interaction_type`: Etkileşim türü (örneğin, Görüntüleme, Beğenme, Paylaşma).
  - `interaction_time`: Etkileşim gerçekleştiği zaman damgası.

### `feed_views`
- **Amaç:** Kullanıcıların belirli RSS akışlarını ne zaman görüntülediğini kaydeder.
- **Sütunlar:**
  - `user_id` (Yabancı Anahtar): `users` tablosuna bağlar.
  - `feed_id` (Yabancı Anahtar): `rss_feeds` tablosuna bağlar.
  - `view_time`: Görüntüleme olayının zaman damgası.

### `user_sessions`
- **Amaç:** Aktif kullanıcı oturumlarını izler, giriş durumunu yönetir ve güvenliği sağlar.
- **Sütunlar:**
  - `user_id` (Yabancı Anahtar): `users` tablosuna bağlar.
  - `session_token`: Oturum için benzersiz token.
  - `expiration_time`: Oturum token'ının sona erme zamanı.
