from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    AWS_ACCESS_KEY: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    AWS_BUCKET_NAME: str
    ML_SERVER_API_KEY: str
    API_KEY_NAME: str
    BACKEND_URL: str
    HF_ACCESS_TOKEN: str
    model_config = SettingsConfigDict(env_file=".env")

@lru_cache
def settings():
    return Settings()
