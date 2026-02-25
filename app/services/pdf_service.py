import io
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    async def extract_text_from_bytes(self, file_content: bytes) -> str:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def create_chunks(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)