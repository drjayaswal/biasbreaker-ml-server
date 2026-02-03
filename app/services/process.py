import os
import nltk
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from app.services.pre_process import filter_noise,get_info
import nltk
import os

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)
nltk.download('punkt')
nltk.download('punkt_tab')


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

async def process2(resume_text: str, description: str, filename: str):
    try:
        if not resume_text or len(resume_text.strip()) < 20:
            raise ValueError("Insufficient text extracted.")

        # 1. POS Tagging & Noise Filtering (Precision Logic)
        resume_info = get_info(resume_text)
        resume_clean = resume_text.replace(".", " ")
        
        jd_skills, jd_noise = filter_noise(description)
        res_skills, res_noise = filter_noise(resume_clean)

        # Lexical Intersection (Exact Matches)
        set_jd = set(jd_skills)
        set_res = set(res_skills)
        matched_skills = sorted(list(set_jd.intersection(set_res)))
        missing_skills = sorted(list(set_jd - set_res))
        extra_skills = sorted(list(set_res - set_jd))
        
        lexical_score = (len(matched_skills) / len(set_jd)) * 100 if set_jd else 0

        # 2. SpaCy Semantic Similarity (Meaning Matching)
        doc_res = nlp(" ".join(res_skills))
        doc_jd = nlp(" ".join(jd_skills))
        semantic_score = doc_res.similarity(doc_jd) * 100 if jd_skills and res_skills else 0

        # 3. TF-IDF Cosine Similarity (Statistical Context)
        # This catches high-frequency words that might be missed by the specific POS filter
        texts = [resume_text.lower(), description.lower()]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]) * 100

        # 4. Final Weighted Scoring
        # 50% Hard Skill Match, 25% AI Meaning, 25% Statistical Context
        final_score = (lexical_score * 0.50) + (semantic_score * 0.25) + (tfidf_similarity * 0.25)

        return {
            "status": "success",
            "filename": filename,
            "match_score": f"{round(final_score, 2)}%",
            "ml_insights": {
                "skill_match_precision": f"{round(lexical_score, 2)}%",
                "semantic_ai_similarity": f"{round(semantic_score, 2)}%",
                "statistical_tfidf_similarity": f"{round(tfidf_similarity, 2)}%"
            },
            "analysis_details": {
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "extra_candidate_skills": extra_skills,
                "total_matches": len(matched_skills),
                "total_missing": len(missing_skills),
                "noise_stats": {
                    "resume_noise_count": len(res_noise),
                    "jd_noise_count": len(jd_noise)
                }
            },
            "candidate_info": resume_info
        }

    except Exception as e:
        return {
            "status": "failed",
            "filename": filename,
            "match_score": "0%",
            "error": str(e)
        }