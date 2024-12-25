# pdf_reader.py
import fitz  # PyMuPDF

def load_pdf(file_path):
    """Extract text from a PDF file."""
    document = fitz.open(file_path)
    pdf_text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pdf_text += page.get_text()
    return pdf_text

def main():
    file_path = "example.pdf"  # Specify your PDF file here
    text = load_pdf(file_path)
    print(f"Extracted Text from {file_path}:\n{text[:500]}...")  # Displaying the first 500 characters

if __name__ == "__main__":
    main()
