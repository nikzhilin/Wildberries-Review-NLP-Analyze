from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://wb:wb@localhost:5432/wb_reviews"
    marketplace: str = "wildberries"
    max_rows: int = 400_000
    hf_dataset: str = "nyuuzyou/wb-feedbacks"
    batch_size: int = 5_000
    sentiment_model: str = "seara/rubert-tiny2-russian-sentiment"
    sentiment_batch_size: int = 64
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = 128
    min_cluster_reviews: int = 50

    model_config = {"env_file": ".env"}


settings = Settings()
