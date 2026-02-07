import re
from youtube_transcript_api import YouTubeTranscriptApi

def get_from_url(url: str):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not video_id_match:
        return None
    
    video_id = video_id_match.group(1)

    try:
        yt_api = YouTubeTranscriptApi() 
        transcript_list = yt_api.list(video_id)
        transcript = transcript_list.find_transcript(['en'])
        data = transcript.fetch()
        
        full_text = " ".join([item.text for item in data]) 
        return full_text
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return None
