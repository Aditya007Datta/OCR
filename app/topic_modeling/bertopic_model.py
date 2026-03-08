"""
Topic modeling using BERTopic for regulatory document analysis.
Discovers themes and clusters in compliance requirements.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

TOPIC_MODEL_DIR = Path("indexes/topic_model")


class TopicModeler:
    """
    BERTopic-based topic modeler for compliance document analysis.
    Falls back to TF-IDF based clustering if BERTopic is unavailable.
    """

    def __init__(self, model_dir: Path = TOPIC_MODEL_DIR, n_topics: str = "auto"):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.n_topics = n_topics
        self._model = None

    def fit(self, texts: list[str], min_topic_size: int = 3) -> dict[str, Any]:
        """
        Fit topic model on a list of texts.

        Returns:
            Dict with topic_keywords, doc_topics, topic_sizes
        """
        if not texts:
            return {"topic_keywords": {}, "doc_topics": [], "topic_sizes": {}}

        logger.info(f"Running topic modeling on {len(texts)} documents")

        # Deduplicate and clean texts
        texts = [t.strip() for t in texts if t.strip() and len(t) > 20]
        if len(texts) < 5:
            logger.warning("Too few documents for topic modeling, using simple clustering")
            return self._simple_cluster(texts)

        # Try BERTopic first
        try:
            return self._bertopic_fit(texts, min_topic_size)
        except ImportError:
            logger.warning("BERTopic not available, using TF-IDF clustering")
            return self._tfidf_cluster(texts)
        except Exception as e:
            logger.error(f"BERTopic failed: {e}, falling back to TF-IDF")
            return self._tfidf_cluster(texts)

    def _bertopic_fit(self, texts: list[str], min_topic_size: int) -> dict[str, Any]:
        """Fit BERTopic model."""
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        logger.info("Fitting BERTopic model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=min_topic_size,
            calculate_probabilities=False,
            verbose=False,
        )

        topics, _ = topic_model.fit_transform(texts)
        self._model = topic_model

        # Extract topic info
        topic_info = topic_model.get_topic_info()
        topic_keywords = {}
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                continue  # Skip outlier topic
            words = topic_model.get_topic(topic_id)
            if words:
                topic_keywords[topic_id] = [w for w, _ in words[:10]]

        topic_sizes = {
            row["Topic"]: row["Count"]
            for _, row in topic_info.iterrows()
            if row["Topic"] != -1
        }

        logger.info(f"BERTopic found {len(topic_keywords)} topics")

        # Save model
        try:
            topic_model.save(str(self.model_dir / "bertopic_model"))
        except Exception as e:
            logger.warning(f"Could not save BERTopic model: {e}")

        return {
            "topic_keywords": topic_keywords,
            "doc_topics": topics,
            "topic_sizes": topic_sizes,
        }

    def _tfidf_cluster(self, texts: list[str]) -> dict[str, Any]:
        """Fallback: TF-IDF + KMeans clustering."""
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("Using TF-IDF + KMeans clustering")
        n_clusters = min(8, max(2, len(texts) // 5))

        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        X = vectorizer.fit_transform(texts)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Extract top keywords per cluster
        feature_names = vectorizer.get_feature_names_out()
        topic_keywords = {}
        topic_sizes = {}

        for cluster_id in range(n_clusters):
            center = kmeans.cluster_centers_[cluster_id]
            top_indices = center.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            topic_keywords[cluster_id] = keywords
            topic_sizes[cluster_id] = int(sum(labels == cluster_id))

        return {
            "topic_keywords": topic_keywords,
            "doc_topics": labels.tolist(),
            "topic_sizes": topic_sizes,
        }

    def _simple_cluster(self, texts: list[str]) -> dict[str, Any]:
        """Very simple clustering for tiny datasets."""
        compliance_themes = {
            0: ["access", "control", "authentication", "authorization", "identity"],
            1: ["data", "protection", "encryption", "privacy", "confidentiality"],
            2: ["incident", "response", "breach", "notification", "recovery"],
            3: ["risk", "assessment", "management", "vulnerability", "threat"],
            4: ["audit", "monitoring", "logging", "review", "compliance"],
        }

        topic_keywords = {k: v for k, v in compliance_themes.items()}
        doc_topics = []
        for text in texts:
            text_lower = text.lower()
            best_topic = 0
            best_score = 0
            for tid, keywords in compliance_themes.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > best_score:
                    best_score = score
                    best_topic = tid
            doc_topics.append(best_topic)

        topic_sizes = {k: doc_topics.count(k) for k in compliance_themes}

        return {
            "topic_keywords": topic_keywords,
            "doc_topics": doc_topics,
            "topic_sizes": topic_sizes,
        }
