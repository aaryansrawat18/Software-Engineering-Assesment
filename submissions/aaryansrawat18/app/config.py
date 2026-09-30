from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    jwt_secret: str
    redis_url: str = ""
    # Shared folder for the API and the worker. Each file is named with its document id.
    upload_dir: str = "uploads"
    gemini_api_key: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_model: str = "gemini-2.5-flash"
    similarity_threshold: float = 0.35

    @field_validator("upload_dir", mode="before")
    @classmethod
    def default_upload_dir(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "uploads"
        return value

    @field_validator("database_url", "jwt_secret")
    @classmethod
    def required_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("embedding_model", mode="before")
    @classmethod
    def default_embedding_model(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "sentence-transformers/all-MiniLM-L6-v2"
        return value

    @field_validator("gemini_model", mode="before")
    @classmethod
    def default_gemini_model(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "gemini-2.5-flash"
        return value

    @field_validator("similarity_threshold", mode="before")
    @classmethod
    def default_similarity_threshold(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return 0.35
        return value


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
