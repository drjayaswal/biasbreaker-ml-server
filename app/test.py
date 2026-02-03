import os
import nltk
import spacy
from fastapi import FastAPI, UploadFile, File
import app.services.pre_process as extract
from app.services.process import process2

app = FastAPI()

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
data_path = os.path.join(os.getcwd(), "nltk_data")
if data_path not in nltk.data.path:
    nltk.data.path.append(data_path)
for res in ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']:
    nltk.download(res, download_dir=data_path, quiet=True)

@app.post("/get-score")
async def get_score(resume: UploadFile = File(...), jd: UploadFile = File(...)):
    resume_raw = extract.text(await resume.read(), resume.filename)
    jd_raw = extract.text(await jd.read(), jd.filename)
    resume_info = extract.get_info(resume_raw)
    resume_processed = resume_raw.replace(".", " ")
    return await process2(resume_processed,jd_raw,resume.filename)
    jd_skills, jd_noise = extract.filter_noise(jd_raw)
    res_skills, res_noise = extract.filter_noise(resume_processed)
    set_jd = set(jd_skills)
    set_res = set(res_skills)
    matched = set_jd.intersection(set_res)
    missing = set_jd - set_res
    unrelated = set_res - set_jd
    lexical_score = (len(matched) / len(set_jd)) * 100 if set_jd else 0
    doc_res = nlp(" ".join(res_skills))
    doc_jd = nlp(" ".join(jd_skills))
    semantic_score = doc_res.similarity(doc_jd) * 100 if jd_skills and res_skills else 0
    final_score = (lexical_score * 0.6) + (semantic_score * 0.4)


    return {
            "status": "success",
            "filename": resume.filename,
            "match_score": round(final_score, 2),
            "analysis_details": {
                "matched_keywords": sorted(list(matched))[:15],
                "missing_keywords": sorted(list(missing))[:15],
                "total_matches": sorted(list(matched)),
                "total_lags": sorted(list(missing)),
                "summary": f"Analyzed {resume.filename}"
            },
            "candidate_info": resume_info,
        }