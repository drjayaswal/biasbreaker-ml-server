import os
import nltk
import spacy
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity
from app.services.pre_process import filter_noise,get_info
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

nltk_data_path = os.getenv("NLTK_DATA", "/home/user/app/nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

try:
    nlp = spacy.load("en_core_web_md")
except:
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")
data_path = os.path.join(os.getcwd(), "nltk_data")
if data_path not in nltk.data.path:
    nltk.data.path.append(data_path)
for res in ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']:
    nltk.download(res, download_dir=data_path, quiet=True)


async def process(resume_text: str, jd_text: str, filename: str):
    try:
        # 1. Validation & Pre-extraction
        if not resume_text or len(resume_text.strip()) < 20:
            raise ValueError("Insufficient resume text for analysis.")
        
        resume_info = get_info(resume_text)
        
        # 2. Advanced NLP Extraction
        resume_processed = resume_text.replace(".", " ")
        jd_skills, jd_noise = filter_noise(jd_text)
        resume_skills, resume_noise = filter_noise(resume_processed)

        # Lexical (Set-based) Analysis
        set_jd = set(jd_skills)
        set_res = set(resume_skills)
        matched = set_jd.intersection(set_res)
        missing = set_jd - set_res
        unrelated = set_res - set_jd
        
        lexical_score = (len(matched) / len(set_jd)) * 100 if set_jd else 0

        # Semantic (Vector-based) Similarity
        doc_res = nlp(" ".join(resume_skills))
        doc_jd = nlp(" ".join(jd_skills))
        semantic_score = doc_res.similarity(doc_jd) * 100 if jd_skills and resume_skills else 0

        # 3. TF-IDF Statistical Analysis (Function 2 Logic)
        texts = [resume_text.lower(), jd_text.lower()]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

        # 4. Weighted Final Score
        # Adjust weights based on priority: Lexical (40%), Semantic (30%), TF-IDF (30%)
        final_score = (lexical_score * 0.4) + (semantic_score * 0.3) + (tfidf_sim * 0.3)

        chart_data = [
                    {"subject": "Lexical Match", "A": round(lexical_score, 2)},
                    {"subject": "Semantic Alignment", "A": round(semantic_score, 2)},
                    {"subject": "Keyword Density", "A": round(tfidf_sim, 2)},
                    {"subject": "Skill Coverage", "A": round((len(matched) / (len(matched) + len(missing))) * 100 if (matched or missing) else 0, 2)},
                    {"subject": "Overall Fit", "A": round(final_score, 2)}
                ]

        # 5. Unified Response Construction
        return {
            "status": "success",
            "filename": filename,
            "match_score": round(final_score, 2),
            "analysis_details": {

                "matched_skills": sorted(list(matched))[:15],
                "total_matched_skills": len(sorted(list(matched))),

                "missing_skills": sorted(list(missing))[:15],
                "total_missed_skills": len(sorted(list(missing))),
                
                "unrelated_skills": sorted(list(unrelated))[:15],
                "total_unrelated_skills": len(sorted(list(unrelated))),
                
                "jd_noise": sorted(list(jd_noise))[:15],
                "total_jd_noise": len(sorted(list(jd_noise))),
                
                "resume_noise": sorted(list(resume_noise))[:15],
                "total_resume_noise": len(sorted(list(resume_noise))),

                "summary": f"Analyzed {filename}",
                "radar_data": chart_data,
            },
            "candidate_info": resume_info,
        }

    except Exception as e:
        return {
            "status": "failed",
            "filename": filename,
            "match_score": 0,
            "error": str(e)
        }