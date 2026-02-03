import os
import nltk
import httpx
import urllib.parse
import app.services.pre_process as extract

nltk_data_path = os.getenv("NLTK_DATA", "/home/user/app/nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

from fastapi import Depends
from app.config import settings
from app.services.process import process
from app.libs.aws_client import s3_client
from fastapi import FastAPI, HTTPException
from fastapi.security.api_key import APIKeyHeader

app = FastAPI()
get_settings = settings()

api_key = get_settings.ML_SERVER_API_KEY
api_key_header = APIKeyHeader(name=get_settings.API_KEY_NAME, auto_error=False)

async def get_api_key(key: str = Depends(api_key_header)):
    if key == api_key:
        return key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "BiasBreaker ML Server is running..."}

@app.get("/health")
async def health_check():
    return {"service":"ML Server", "status": "healthy", "active":True}

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
            bucket_name = get_settings.AWS_BUCKET_NAME
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