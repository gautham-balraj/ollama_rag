# RAG with Ollama and Pinecone

Semantic Search is an application that allows users to perform semantic searches on textual data using the Ollama language model and Pinecone for serverless vector database. This application extracts text from a given URL, splits it into chunks, and indexes the chunks in Pinecone for fast and efficient semantic searching. It then allows users to ask questions related to the content, with Ollama providing contextualized answers based on the indexed text.

## Features

- Extracts text from a URL
- Splits the text into chunks for indexing
- Indexes the text chunks in Pinecone for semantic searching
- Provides a user interface for asking questions related to the content
- Ollama provides contextualized answers based on the indexed text

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/semantic-search.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   - Set your Pinecone API key:

     ```bash
     export PINECONE_API_KEY=your_pinecone_api_key
     ```

   - Set your Cohere API key:

     ```bash
     export COHERE_API_KEY= your_cohere_api_key
     ```

## Usage

1. Run the main script:

   ```bash
   python main.py --url <URL> --model <ollama_model>
   ```

   Replace `<URL>` with the URL you want to extract text from and `<ollama_model>` with the Ollama model you want to use.

2. Once the script finishes indexing the text and setting up the server, you can start asking questions. Enter your question when prompted. To exit, type `bye`.

## Example

```bash
python main.py --url http://example.com --model mistral
```

## Credits

- [Ollama](https://github.com/ollama-dev/ollama): Open-source language model for natural language understanding.
- [Pinecone](https://www.pinecone.io/): Managed vector database service.
- [Cohere](https://cohere.ai/): API for state-of-the-art natural language processing models.



