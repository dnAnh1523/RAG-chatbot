import os
import re
import pdfplumber
import pymupdf4llm
from typing import Dict, List, Optional

class PDFToMarkdownConverter:
    """Convert PDF files to Markdown format with proper structure and formatting."""
    
    def __init__(self):
        """Initialize the converter"""
        pass
    
    def _clean_text(self, text: str) -> str:
        """Clean and format extracted text for Markdown, preserving headings, subheadings, and bullets"""
        # Remove page numbers and excessive dots (e.g., '..... 12' or '12') at line ends
        text = re.sub(r'\.{3,}\s*\d+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(^|\n)\s*\d+\s*(\n|$)', '\n', text)  # Remove lines that are just numbers
        # Normalize line breaks
        text = text.replace('\r', '')
        # Convert bullet points (various unicode bullets and dashes)
        text = re.sub(r'^[\u2022\u2023\u25E6\u2043\u2219\-\*]\s+', '- ', text, flags=re.MULTILINE)
        # Convert numbered lists (1. or 1) or 1 -
        text = re.sub(r'^(\d+)\s*[\.|\)|\-]\s+', r'\1. ', text, flags=re.MULTILINE)
        # Detect headings (all caps, possibly with diacritics, or lines surrounded by ===/---)
        def heading_replacer(match):
            line = match.group(1).strip()
            if len(line.split()) <= 12:
                return f'\n# {line}\n'
            return line
        text = re.sub(r'^(\s*[A-ZÀ-Ỹ][A-ZÀ-Ỹ0-9\s\-:,]+)$', heading_replacer, text, flags=re.MULTILINE)
        # Detect subheadings (lines with some uppercase, not all, or lines with ---/=== below)
        text = re.sub(r'^(=+|\-+)$', '', text, flags=re.MULTILINE)
        # Remove excessive whitespace and blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        # Clean up line breaks and strip
        text = text.strip()
        # Remove leading/trailing separators and blank lines
        text = re.sub(r'^(\-|\*|#|\s)+', '', text)
        text = re.sub(r'(\-|\*|#|\s)+$', '', text)
        return text
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Convert table to Markdown format with clear separation"""
        if not table or not table[0]:
            return ""
        # Clean each cell and ensure no newlines inside cells
        clean_table = [[(str(cell) if cell else '').replace('\n', ' ').strip() for cell in row] for row in table]
        # Create header row
        md_table = ["| " + " | ".join(clean_table[0]) + " |"]
        # Add separator row
        md_table.append("|" + " --- |" * len(clean_table[0]))
        # Add data rows
        for row in clean_table[1:]:
            md_table.append("| " + " | ".join(row) + " |")
        # Add extra line before and after table for clarity
        return "\n\n" + "\n".join(md_table) + "\n\n"
    
    def convert_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """Convert PDF to Markdown format with enhanced structure for RAG"""
        try:
            markdown_content = pymupdf4llm.to_markdown(pdf_path)
        except Exception as e:
            print(f"pymupdf4llm conversion failed, falling back to pdfplumber: {str(e)}")
            markdown_content = self._convert_with_pdfplumber(pdf_path)
        # Clean up excessive page separators
        markdown_content = re.sub(r'(\n---\n\s*)+', '\n---\n', markdown_content)
        # Save to file if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        return markdown_content
    
    def _convert_with_pdfplumber(self, pdf_path: str) -> str:
        """Convert PDF using pdfplumber with improved structure for RAG"""
        markdown_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables first to avoid mixing with text
                tables = page.extract_tables() or []
                text = page.extract_text() or ""
                # Remove table text from main text if possible (not always reliable)
                if tables and text:
                    for table in tables:
                        for row in table:
                            for cell in row:
                                if cell and cell.strip() in text:
                                    text = text.replace(cell.strip(), '')
                # Clean and append text
                if text.strip():
                    markdown_content.append(self._clean_text(text))
                # Format and append tables
                for table in tables:
                    if table:
                        markdown_content.append(self._format_table(table))
                # Page separator for chunking
                markdown_content.append("\n---\n")
        return "\n".join(markdown_content)

def main():
    """Main function to run the converter"""
    # Initialize converter
    converter = PDFToMarkdownConverter()
    
    # Set paths
    pdf_folder = "Doctuments RAG ictu"
    output_folder = "data/Markdown_2"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert all PDFs in the folder
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            output_filename = os.path.splitext(filename)[0].replace(' ', '_') + '.md'
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"Converting {filename} to Markdown...")
            converter.convert_pdf(pdf_path, output_path)
            print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()