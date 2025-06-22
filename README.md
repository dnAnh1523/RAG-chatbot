<h1 align="center">RAG Chatbot</h1>

<p align="center">
  <img alt="Github top language" src="https://img.shields.io/github/languages/top/dnAnh1523/RAG-chatbot?color=56BEB8">
  <img alt="Github language count" src="https://img.shields.io/github/languages/count/dnAnh1523/RAG-chatbot?color=56BEB8">
  <img alt="Repository size" src="https://img.shields.io/github/repo-size/dnAnh1523/RAG-chatbot?color=56BEB8">
</p>

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#sparkles-features">Features</a> &#xa0; | &#xa0;
  <a href="#rocket-technologies">Technologies</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-starting">Starting</a> &#xa0; | &#xa0;
  <a href="https://github.com/dnAnh1523" target="_blank">Author</a>
</p>

<br>

## :dart: About ##

A chatbot system based on Retrieval-Augmented Generation (RAG), designed to answer user questions using context retrieved from structured and unstructured documents. This project supports both domain-specific knowledge (e.g., student handbooks) and general queries.

## :sparkles: Features ##

:heavy_check_mark: Document ingestion pipeline with chunking and embedding;\
:heavy_check_mark: Dual-mode QA (retrieval-based + LLM response);\
:heavy_check_mark: Streamlit-based UI for interactive use.

## :rocket: Technologies ##

The following tools were used in this project:

- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)
- [Python](https://www.python.org/)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)

## :white_check_mark: Requirements ##

Before starting :checkered_flag:, you need to have the following installed:

- [Git](https://git-scm.com)
- [Python 3.10+](https://www.python.org/)
- [pip](https://pip.pypa.io/en/stable/)
- (Optional) [Virtualenv](https://virtualenv.pypa.io/)

## :checkered_flag: Starting ##

```bash
# Clone this project
$ git clone https://github.com/dnAnh1523/rag-chatbot

# Access the project folder
$ cd rag-chatbot

# (Optional) Create and activate virtual environment
$ python -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
$ pip install -r requirements.txt

# Run the app (example for Streamlit)
$ streamlit run app.py
```

Made with :heart: by <a href="https://github.com/dnAnh1523" target="_blank">Nhat Anh</a>

&#xa0;

<a href="#top">Back to top</a>
