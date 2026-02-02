import httpx
import urllib.parse
from app.config import settings
from app.libs.aws_client import s3_client
from app.services.process import process
from fastapi import FastAPI, HTTPException
import app.services.pre_process as extract

app = FastAPI()
get_settings = settings()

@app.get("/")
def read_root():
    return {"status": "BiasBreaker ML Server is running"}

@app.post("/analyze-s3")
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


@app.post("/analyze-drive")
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
        return await process(resume_text, description, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))