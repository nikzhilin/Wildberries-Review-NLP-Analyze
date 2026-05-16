from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    marketplace = Column(String(64), nullable=False)
    external_id = Column(String(64), nullable=False)  # nmId

    reviews = relationship("Review", back_populates="product")

    __table_args__ = (
        UniqueConstraint(
            "marketplace", "external_id", name="uq_products_marketplace_external_id"
        ),
    )


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    external_id = Column(String(128), nullable=False)  # sha256[:32] of "{nmId}:{text}"
    text = Column(Text, nullable=False)
    text_clean = Column(Text)
    rating = Column(Integer)  # productValuation 1–5
    color = Column(String(256))
    answer = Column(Text)
    has_answer = Column(Boolean, default=False)
    review_length = Column(Integer)
    word_count = Column(Integer)
    answer_length = Column(Integer)

    product = relationship("Product", back_populates="reviews")

    __table_args__ = (UniqueConstraint("external_id", name="uq_reviews_external_id"),)


class Topic(Base):
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    polarity = Column(String(8), nullable=False)  # "pos" | "neg"
    keywords = Column(Text)  # comma-separated
    review_count = Column(Integer, default=0)

    product = relationship("Product")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    rule_name = Column(String(128), nullable=False)
    severity = Column(String(32), nullable=False)  # "low" | "medium" | "high"
    details = Column(JSON)
    created_at = Column(DateTime, nullable=False)

    product = relationship("Product")


class ReviewML(Base):
    __tablename__ = "review_ml"

    review_id = Column(Integer, ForeignKey("reviews.id"), primary_key=True)
    sentiment_label = Column(String(16))  # "positive" | "negative" | "neutral"
    sentiment_score = Column(Float)
    topic_id = Column(Integer, ForeignKey("topics.id"))
    fake_score = Column(Float)
    predicted_rating = Column(Float)
    embedding = Column(Vector(384))

    review = relationship("Review")
    topic = relationship("Topic")
