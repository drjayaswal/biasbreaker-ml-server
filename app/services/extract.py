import io
import httpx
from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_url(content: bytes, mime_type: str) -> str:
    text = ""
    try:
        if not content: return ""
        stream = io.BytesIO(content)

        if "pdf" in mime_type:
            reader = PdfReader(stream)
            text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])

        elif "wordprocessingml" in mime_type or mime_type.endswith("docx"):
            doc = Document(stream)
            text = " ".join([para.text for para in doc.paragraphs if para.text])

        else:
            text = content.decode("utf-8", errors="ignore")

    except Exception as e:
        print(f"Extraction Error: {str(e)}")
    return text.strip()

async def extext(url: str):
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise Exception(f"Download Error: {resp.status_code}")
        
        m_type = resp.headers.get("Content-Type", "").lower()
        if not m_type or "octet-stream" in m_type:
            ext = url.split('?')[0].lower()
            if ext.endswith(".pdf"): m_type = "application/pdf"
            elif ext.endswith(".docx"): m_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else: m_type = "text/plain"

        return extract_text_from_url(content=resp.content, mime_type=m_type)