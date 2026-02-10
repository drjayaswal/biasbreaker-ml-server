# Internal ------------------------------------------
import os
import nltk
import httpx
import urllib.parse
import app.services.pre_process as extract

nltk_data_path = os.getenv("NLTK_DATA", "/home/user/app/nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

from app.config import settings
from app.services.process import process
from app.libs.aws_client import s3_client
from fastapi import FastAPI, HTTPException
from fastapi import Depends,BackgroundTasks
from app.libs.model import VideoIngestRequest
from fastapi.security.api_key import APIKeyHeader
from app.libs.model import VideoIngestRequest, PDFIngestRequest, GenerateRequest, VectorRequest
import app.services.transcription as transcription
# ---------------------------------------------------

# External ------------------------------------------
import os
import logging
import warnings
import transformers.utils.logging
# -------------------------------
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, HTTPException, BackgroundTasks
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains.combine_documents import create_stuff_documents_chain
# -------------------------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

load_dotenv()
env = settings()

os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

transformers.utils.logging.disable_progress_bar()

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'},
    query_encode_kwargs={"prompt": "Represent this sentence for searching relevant passages: "}
)
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="text-generation",
    huggingfacehub_api_token=env.HF_ACCESS_TOKEN,
    temperature=0.7,
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=llm_endpoint)
vector_db = Chroma(
    collection_name="annotate_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db" 
)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context from a video information by its transcription or document information from its text to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 3}), question_answer_chain)

app = FastAPI()
env = settings()
key = env.ML_SERVER_API_KEY
key_header = APIKeyHeader(name=env.API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(key_header)):
    if api_key == key:
        return key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

app = FastAPI()



# ----------------------- Sync Service ---------------------------
async def update_to_neon(source_id: str, chunks):
    SYNC_URL = f"{env.BACKEND_URL}/update-source-chunks"
    texts = [doc.page_content for doc in chunks]
    vectors = embeddings.embed_documents(texts)
    payload = {
        "source_id": source_id,
        "chunks": [
            {"content": text, "embedding": vector} 
            for text, vector in zip(texts, vectors)
        ]
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(SYNC_URL, json=payload)
        if response.status_code != 200:
            raise Exception(f"Backend Sync Failed: {response.text}")
        return True

# ----------------------- Process Services ---------------------------
async def process_pdf(text: str, source_id: str, filename: str):
    STATUS_URL = f"{env.BACKEND_URL}/update-source-status"
    try:
        doc = Document(page_content=text, metadata={"source_id": source_id, "name": filename})
        chunks = text_splitter.split_documents([doc])
        await update_to_neon(source_id, chunks)
        vector_db.add_documents(chunks)
        async with httpx.AsyncClient() as client:
            await client.patch(STATUS_URL, json={"source_id": source_id, "status": "completed"})
    except Exception as e:
        async with httpx.AsyncClient() as client:
            await client.patch(STATUS_URL, json={"source_id": source_id, "status": "failed"})
async def process_video(url: str, source_id: str):
    STATUS_URL = f"{env.BACKEND_URL}/update-source-status"
    try:
        text = transcription.get_from_url(url)
        doc = Document(page_content=text, metadata={"source_id": source_id, "name": url})
        chunks = text_splitter.split_documents([doc])
        
        await update_to_neon(source_id, chunks)
        vector_db.add_documents(chunks)
        async with httpx.AsyncClient() as client:
            await client.patch(STATUS_URL, json={"source_id": source_id, "status": "completed"})
    except Exception as e:
        logging.error(f"ML Video Processing Failed: {e}")
        async with httpx.AsyncClient() as client:
            await client.patch(STATUS_URL, json={"source_id": source_id, "status": "failed"})










@app.get("/")
def read_root():
    return {"status": "Basal ML Server is running..."}

@app.get("/health")
async def health_check():
    return {"service":"ML Server", "status": "healthy", "active":True}

@app.post("/get-vector")
async def get_vector(req: VectorRequest):
    try:
        vector = embeddings.embed_query(req.text)
        return {"vector": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

@app.post("/generate-answer")
async def generate_answer(request: GenerateRequest):
    try:
        if not request.context or not request.context.strip():
            return {
                "answer": "I couldn't find any relevant information in the documents to answer this.",
                "status": "no_context"
            }
        system_instructions = (
            "You are a helpful AI assistant. Use the provided context to answer the question. "
            "If the answer isn't in the context, say you don't know and politely ask for context.\n\n"
            f"CONTEXT:\n{request.context}"
        )
        messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=request.question)
        ]
        response = llm.invoke(messages)
        return {"answer": response.content, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Generation failed: {str(e)}")

@app.post("/analyze-video",dependencies=[Depends(get_api_key)])
async def analyze_video(request: VideoIngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_video, request.url, request.source_id)
    return {"status": "accepted"}

@app.post("/analyze-document")
async def analyze_document(request: PDFIngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_pdf, request.text, request.source_id, request.filename)
    return {"status": "accepted"}

@app.post("/analyze-s3",dependencies=[Depends(get_api_key)])
async def analyze_s3(data: dict):
    file_url = data.get('file_url')
    description = data.get('description')
    filename = data.get('filename', 's3_file')
    if not file_url or not description:
        raise HTTPException(status_code=400, detail="Missing data")
    try:
        resume_text = await extract.text_from_url(file_url)
        results = await process(resume_text, description, filename)
        try:
            parsed_url = urllib.parse.urlparse(file_url)
            s3_key = urllib.parse.unquote(parsed_url.path.lstrip('/'))
            s3_key = s3_key.split('?')[0]
            bucket_name = env.AWS_BUCKET_NAME
            if s3_key.startswith(f"{bucket_name}/"):
                s3_key = s3_key.replace(f"{bucket_name}/", "", 1)
            s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        except Exception as delete_err:
            raise Exception({"message":delete_err})
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-drive",dependencies=[Depends(get_api_key)])
async def analyze_drive(data: dict):
    file_id = data.get("file_id")
    token = data.get("google_token")
    description = data.get("description", "")
    filename = data.get("filename", "drive_file")
    mime_type = data.get("mime_type", "")
    if not file_id or not token:
        raise HTTPException(status_code=400, detail="Missing credentials")
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            if "google-apps" in mime_type:
                url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=application/pdf"
                target_mime = "application/pdf"
            else:
                url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
                target_mime = mime_type
            resp = await client.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=45.0)
            if resp.status_code != 200:
                raise Exception(f"Google Drive Error {resp.status_code}: {resp.text[:100]}")
            if len(resp.content) < 200:
                 raise Exception("File content too small; likely a failed download.")
            resume_text = extract.text(resp.content, target_mime)
        results = await process(resume_text, description, filename)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))