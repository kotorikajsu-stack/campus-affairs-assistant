from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "campus-affairs-assistant"
    app_env: str = "local"
    log_level: str = "INFO"

    llm_provider: str = "mock"
    llm_base_url: str = "http://127.0.0.1:8001/v1"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_api_key: str = ""

    embedding_model: str = "BAAI/bge-m3"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "campus_affairs_docs"

    default_tenant_id: str = "demo-university"
    max_context_chunks: int = 5
    sse_heartbeat_seconds: int = 15

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

