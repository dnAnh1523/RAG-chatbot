"""
ingest.py — Run this ONCE to parse the PDF and build the vector store.
Usage: python ingest.py

It will:
1. Parse SO_TAY_SINH_VIEN.pdf with layout-aware extraction (handles complex tables)
2. Chunk the text semantically
3. Embed with bge-m3 (runs on your RTX 3050, ~500MB VRAM)
4. Save the vector store locally to ./vectorstore/
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma          # dedicated package — not community wrapper

from langchain_community.vectorstores.utils import filter_complex_metadata

load_dotenv()

PDF_PATH = Path("data/SO_TAY_SINH_VIEN.pdf")
VECTORSTORE_PATH = "./vectorstore"
EMBED_MODEL = "BAAI/bge-m3"  # Best multilingual model, strong Vietnamese support


def load_pdf(pdf_path: Path):
    """
    Uses UnstructuredPDFLoader with 'hi_res' strategy.
    This reconstructs tables properly instead of garbling them —
    critical for your 3-4 page spanning tables.
    """
    print(f"📄 Loading PDF: {pdf_path}")
    print("   (Using hi_res strategy — this takes 1-2 min on first run, be patient)")

    loader = UnstructuredPDFLoader(
        str(pdf_path),
        mode="elements",           # returns each element (title, text, table) separately
        strategy="hi_res",         # layout-aware, best for complex tables
        infer_table_structure=True # reconstructs table as structured text
    )
    elements = loader.load()
    print(f"   ✓ Loaded {len(elements)} elements from PDF")
    return elements


def chunk_documents(documents):
    """
    RecursiveCharacterTextSplitter with Vietnamese-friendly settings.
    chunk_size=800 is a good balance for this type of regulatory document.
    chunk_overlap=150 prevents information loss at chunk boundaries.
    """
    print("✂️  Chunking documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", ".", " ", ""],  # handles Vietnamese sentence endings
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"   ✓ Created {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    """
    Embeds chunks using bge-m3 on your GPU, saves to ChromaDB locally.
    bge-m3 is ~570MB — fits easily on your 8GB VRAM alongside the Groq calls.
    """
    print(f"🧠 Loading embedding model: {EMBED_MODEL}")
    print("   (First time will download ~570MB — subsequent runs use cache)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"},  # use your RTX 3050
        encode_kwargs={"normalize_embeddings": True},  # required for bge models
    )

    print("📦 Building vector store...")

    # ←←← THIS IS THE CRITICAL LINE ←←←
    filtered_chunks = filter_complex_metadata(chunks)

    vectorstore = Chroma.from_documents(
        documents=filtered_chunks,           # ← use the filtered version
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH,
        collection_name="sotay_sinh_vien",
    )

    print(f"   ✓ Vector store saved to {VECTORSTORE_PATH}/")
    print(f"   ✓ Total vectors: {vectorstore._collection.count()}")
    return vectorstore


def main():
    if not PDF_PATH.exists():
        print(f"❌ PDF not found at {PDF_PATH}")
        print(f"   Please place your PDF in: {PDF_PATH.parent.resolve()}/")
        return

    print("=" * 50)
    print("  Sổ Tay Sinh Viên — RAG Ingestion Pipeline")
    print("=" * 50)

    elements = load_pdf(PDF_PATH)
    chunks = chunk_documents(elements)
    build_vectorstore(chunks)

    print("\n✅ Done! Vector store is ready.")
    print("   Now run: chainlit run app.py")


if __name__ == "__main__":
    main()
