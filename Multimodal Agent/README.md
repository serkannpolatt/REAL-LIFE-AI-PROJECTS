# Multimodal RAG (PDF With Images)

This project demonstrates a Multimodal Retrieval-Augmented Generation (RAG) pipeline for PDFs containing both text and images. It uses CLIP for unified embeddings and GPT-4 Vision for answering queries with both text and image context.

## Features
- Extracts text and images from PDF files
- Embeds both text and images using CLIP
- Stores embeddings in a FAISS vector store for efficient retrieval
- Retrieves relevant text and images for a given query
- Constructs multimodal messages for GPT-4 Vision
- Supports queries that require both textual and visual context

## Requirements
- Python 3.8+
- PyMuPDF (fitz)
- langchain
- transformers
- Pillow
- torch
- numpy
- scikit-learn
- python-dotenv
- FAISS

## Usage
1. Place your PDF file (e.g., `multimodal_sample.pdf`) in the project directory.
2. Set your OpenAI API key in a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```
3. Run the notebook `main.ipynb`.
4. Example queries:
   - What does the chart on page 1 show about revenue trends?
   - Summarize the main findings from the document
   - What visual elements are present in the document?

## How It Works
- **PDF Processing:** Extracts text and images from each page. Text is split into chunks; images are converted to base64 for GPT-4V.
- **Embedding:** Both text and images are embedded using CLIP, normalized, and stored.
- **Retrieval:** For a user query, the system embeds the query and retrieves the most relevant text and images from the FAISS vector store.
- **Multimodal Message:** Constructs a message containing both text excerpts and images, formatted for GPT-4 Vision.
- **Response:** GPT-4 Vision answers the query using the provided multimodal context.

## Main Functions
- `embed_image(image_data)`: Embeds images using CLIP.
- `embed_text(text)`: Embeds text using CLIP.
- `retrieve_multimodal(query, k=5)`: Retrieves top-k relevant documents (text and images).
- `create_multimodal_message(query, retrieved_docs)`: Prepares a multimodal message for GPT-4V.
- `multimodal_pdf_rag_pipeline(query)`: Main pipeline for answering queries.


## License
This project is for educational and research purposes.
