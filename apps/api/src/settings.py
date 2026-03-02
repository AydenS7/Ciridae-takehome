from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    gateway_api_key: str
    llm_gateway_base_url: str = "https://llm-gateway-5q22j.ondigitalocean.app"

settings = Settings()
