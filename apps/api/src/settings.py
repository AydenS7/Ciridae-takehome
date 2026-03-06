"""Typed application settings loaded from environment variables and defaults."""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    gateway_api_key: str
    llm_gateway_base_url: str = "https://llm-gateway-5q22j.ondigitalocean.app"
    extract_primary_model: str = "gemini/gemini-2.5-pro"
    extract_secondary_model: str = "anthropic/claude-opus-4-5"
    extract_enable_secondary: bool = False
    extract_enable_vision: bool = True
    extract_vision_model: str = "gemini/gemini-2.5-pro"
    extract_chunk_max_chars: int = 18000
    extract_max_workers: int = 3
    extract_min_confidence_primary: float = 0.40
    extract_min_confidence_secondary: float = 0.40
    extract_secondary_only_min_confidence: float = 0.60
    matching_first_pass_model: str = "openai/gpt-4.1-mini"
    matching_second_pass_models: str = "anthropic/claude-3-5-sonnet-latest"
    roommap_model: str = "openai/gpt-4.1"
    matching_second_pass_trigger_confidence: float = 0.90
    matching_second_pass_amount_tolerance_pct: float = 0.02
    matching_first_pass_unsure_confidence: float = 0.50
    matching_first_pass_unsure_null_confidence: float = 0.62
    matching_first_pass_scope_unsure_confidence: float = 0.68
    matching_force_review_on_null: bool = True
    matching_second_pass_max_workers: int = 4
    matching_accept_confidence: float = 0.35
    matching_green_amount_tolerance_pct: float = 0.02
    matching_fallback_similarity_threshold: float = 0.28
    matching_reconcile_similarity_threshold: float = 0.34
    matching_reconcile_scope_similarity_threshold: float = 0.44
    matching_scope_similarity_threshold: float = 0.34
    matching_scope_overlap_threshold: float = 0.44
    matching_assignment_unmatched_threshold: float = 0.28
    matching_blue_guard_unused_threshold: float = 0.30
    matching_blue_guard_consumed_threshold: float = 0.38
    matching_green_text_exact_tolerance_pct: float = 0.06
    # Price-proximity rescue: after all other matching, rescue still-blue A items by price closeness
    rescue_green_price_tol: float = 0.05    # ±5% → can force green (if desc_sim also passes)
    rescue_orange_price_tol: float = 0.15   # ±15% → can force orange
    rescue_min_desc_sim_green: float = 0.40  # min token-Jaccard for green rescue
    rescue_min_desc_sim_orange: float = 0.20 # min token-Jaccard for orange rescue

    @property
    def matching_second_pass_model_list(self) -> list[str]:
        models = [m.strip() for m in self.matching_second_pass_models.split(",")]
        return [m for m in models if m]

settings = Settings()
