from functools import lru_cache
from math import log1p
from urllib.parse import quote
import warnings

import numpy as np
import pandas as pd
import requests
import networkx as nx
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR


class SemanticScholarAuthorityScorer:
    """Uses Semantic Scholar metadata to estimate paper authority."""

    def __init__(self, timeout=10):
        self.timeout = timeout
        self.base_url = "https://api.semanticscholar.org/graph/v1"

    @lru_cache(maxsize=512)
    def score_title(self, title):
        if not title:
            return 1.0

        paper = self._fetch_paper_metadata(title)
        if not paper:
            return 1.0

        citation_count = float(paper.get("citationCount", 0) or 0)
        author_scores = []
        for author in paper.get("authors", [])[:3]:
            author_id = author.get("authorId")
            if author_id:
                author_scores.append(float(self._fetch_author_hindex(author_id)))

        avg_hindex = float(np.mean(author_scores)) if author_scores else 0.0
        return 1.0 + log1p(citation_count) + (avg_hindex / 20.0)

    def _fetch_paper_metadata(self, title):
        try:
            query = quote(title)
            url = (
                f"{self.base_url}/paper/search?query={query}"
                "&fields=title,citationCount,authors.authorId,authors.name"
                "&limit=1"
            )
            response = requests.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            payload = response.json()
            papers = payload.get("data", [])
            return papers[0] if papers else None
        except Exception:
            return None

    @lru_cache(maxsize=1024)
    def _fetch_author_hindex(self, author_id):
        try:
            url = f"{self.base_url}/author/{author_id}?fields=hIndex"
            response = requests.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return 0
            payload = response.json()
            return int(payload.get("hIndex", 0) or 0)
        except Exception:
            return 0


class TrendAnalyzer:
    """Strictly responsible for backward-looking analysis."""

    def __init__(self, authority_scorer=None):
        self.authority_scorer = authority_scorer or SemanticScholarAuthorityScorer()

    def _paper_topic_set(self, item, node_to_community):
        topic_nodes = set()
        for rel in item.get("triplet_objects", []):
            subject = getattr(rel.subject, "canonical_name", None)
            object_ = getattr(rel.object_, "canonical_name", None)
            if subject:
                topic_nodes.add(subject)
            if object_:
                topic_nodes.add(object_)

        topics = {node_to_community.get(node) for node in topic_nodes}
        return {topic for topic in topics if topic and topic != "Unknown"}

    def calculate_historical_share(
        self,
        graphs_by_year,
        node_to_community,
        community_labels=None,
        processed_data=None,
    ):
        """
        Calculates the historical market share of each detected community over time.
        Historical share is authority-weighted with Semantic Scholar citation/h-index signals.
        """
        community_labels = community_labels or {}
        processed_data = processed_data or []

        items_by_year = {}
        for item in processed_data:
            year = item.get("year")
            if year is None:
                continue
            items_by_year.setdefault(int(year), []).append(item)

        yearly_topic_counts = []
        topic_totals = {}
        topic_years = {}
        latest_bridge_scores = {}
        latest_shift_flags = {}

        for year, graph in sorted(graphs_by_year.items()):
            total_nodes = len(graph.nodes)
            if total_nodes == 0:
                continue

            weighted_counts = {}
            total_weight = 0.0
            yearly_items = items_by_year.get(int(year), [])

            if yearly_items:
                for item in yearly_items:
                    topics = self._paper_topic_set(item, node_to_community)
                    if not topics:
                        continue

                    title = item.get("title", "")
                    authority_weight = self.authority_scorer.score_title(title)
                    total_weight += authority_weight * len(topics)
                    for topic in topics:
                        weighted_counts[topic] = weighted_counts.get(topic, 0.0) + authority_weight
            else:
                for node in graph.nodes:
                    topic = node_to_community.get(node, "Unknown")
                    if topic == "Unknown":
                        continue
                    weighted_counts[topic] = weighted_counts.get(topic, 0.0) + 1.0
                total_weight = sum(weighted_counts.values())

            bridge_scores = self._topic_bridge_scores(graph, node_to_community)
            bridge_thresholds = self._bridge_shift_flags(graphs_by_year, node_to_community, year)

            for topic, count in weighted_counts.items():
                topic_name = community_labels.get(topic, topic.replace("_", " "))
                bridge_score = bridge_scores.get(topic, 0.0)
                bridge_flag = bridge_thresholds.get(topic, False)
                topic_totals[topic] = topic_totals.get(topic, 0.0) + float(count)
                topic_years[topic] = topic_years.get(topic, 0) + 1
                latest_bridge_scores[topic] = bridge_score
                latest_shift_flags[topic] = bridge_flag
                yearly_topic_counts.append({
                    "year": int(year),
                    "topic": topic,
                    "topic_name": topic_name,
                    "share": count / total_weight if total_weight else 0.0,
                    "authority_weight": float(count),
                    "bridge_score": float(bridge_score),
                    "potential_paradigm_shift": bool(bridge_flag),
                    "is_forecast": False,
                })

        historical_df = pd.DataFrame(yearly_topic_counts)
        if historical_df.empty:
            return historical_df

        ranking_df = (
            historical_df.groupby(["topic", "topic_name"], as_index=False)
            .agg(
                total_share=("share", "sum"),
                avg_share=("share", "mean"),
                latest_year=("year", "max"),
                authority_weight=("authority_weight", "sum"),
                bridge_score=("bridge_score", "max"),
                potential_paradigm_shift=("potential_paradigm_shift", "max"),
            )
        )
        ranking_df["topic_score"] = (
            ranking_df["total_share"] * 0.5
            + ranking_df["avg_share"] * 0.3
            + ranking_df["authority_weight"] * 0.1
            + ranking_df["bridge_score"] * 0.1
        )
        ranking_df = ranking_df.sort_values(["topic_score", "authority_weight"], ascending=False)

        top_topics = ranking_df["topic"].head(10).tolist()
        if top_topics:
            historical_df = historical_df[historical_df["topic"].isin(top_topics)].copy()

        historical_df["topic_score"] = historical_df["topic"].map(
            ranking_df.set_index("topic")["topic_score"].to_dict()
        )
        historical_df["topic_rank"] = historical_df["topic"].map(
            {topic: rank + 1 for rank, topic in enumerate(ranking_df["topic"].tolist())}
        )
        historical_df = historical_df.sort_values(["topic_score", "share", "year"], ascending=[False, False, True])
        return historical_df

    def _topic_bridge_scores(self, graph, node_to_community):
        if graph.number_of_nodes() == 0:
            return {}

        undirected = graph.to_undirected()
        betweenness = nx.betweenness_centrality(undirected) if undirected.number_of_edges() > 0 else {node: 0.0 for node in undirected.nodes}
        topic_scores = {}
        topic_counts = {}
        for node, score in betweenness.items():
            topic = node_to_community.get(node, "Unknown")
            if topic == "Unknown":
                continue
            topic_scores[topic] = topic_scores.get(topic, 0.0) + float(score)
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        return {
            topic: topic_scores[topic] / max(1, topic_counts.get(topic, 1))
            for topic in topic_scores
        }

    def _bridge_shift_flags(self, graphs_by_year, node_to_community, current_year):
        sorted_years = sorted(int(year) for year in graphs_by_year.keys())
        if int(current_year) not in sorted_years:
            return {}

        current_index = sorted_years.index(int(current_year))
        previous_years = sorted_years[max(0, current_index - 3):current_index]
        if not previous_years:
            return {}

        historical_scores = {}
        for year in previous_years:
            graph = graphs_by_year[year]
            scores = self._topic_bridge_scores(graph, node_to_community)
            for topic, score in scores.items():
                historical_scores.setdefault(topic, []).append(score)

        current_scores = self._topic_bridge_scores(graphs_by_year[int(current_year)], node_to_community)
        flags = {}
        for topic, current_score in current_scores.items():
            previous_values = historical_scores.get(topic, [])
            if not previous_values:
                continue
            baseline = float(np.mean(previous_values))
            flags[topic] = current_score > (baseline * 1.5) and current_score > 0.0
        return flags


class TrendForecaster:
    """Strictly responsible for forward-looking forecasting."""

    def __init__(self, forecast_method="arima"):
        self.forecast_method = forecast_method.lower().strip()

    def calculate_historical_share(self, graphs_by_year, node_to_community, community_labels=None, processed_data=None):
        analyzer = TrendAnalyzer()
        return analyzer.calculate_historical_share(
            graphs_by_year,
            node_to_community,
            community_labels=community_labels,
            processed_data=processed_data,
        )

    def _forecast_topic_arima(self, topic_data, future_years):
        share_series = topic_data.sort_values("year")["share"].astype(float)
        if len(share_series) < 2:
            return None

        order = (1, 1, 0) if len(share_series) >= 5 else (1, 0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(share_series, order=order)
            result = model.fit()
            forecast = result.forecast(steps=future_years)
        return np.asarray(forecast, dtype=float)

    def _forecast_topic_var(self, topic_data, future_years):
        if "bridge_score" not in topic_data.columns:
            return None

        multivariate = topic_data.sort_values("year")[["share", "bridge_score"]].astype(float).dropna()
        if len(multivariate) < 3:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(multivariate)
            lag_order = min(2, len(multivariate) - 1)
            result = model.fit(lag_order)
            forecast = result.forecast(multivariate.values[-lag_order:], steps=future_years)
        return np.asarray(forecast[:, 0], dtype=float)

    def forecast_trends(self, historical_df, future_years=3, method=None):
        """
        Applies ARIMA by default, with VAR available as an alternate forecast method.
        """
        if historical_df.empty:
            return historical_df

        forecast_method = (method or self.forecast_method or "arima").lower().strip()
        forecasts = []
        topics = historical_df["topic"].unique()
        topic_names = {}
        if "topic_name" in historical_df.columns:
            topic_names = historical_df.drop_duplicates("topic").set_index("topic")["topic_name"].to_dict()

        for topic in topics:
            topic_data = historical_df[historical_df["topic"] == topic].sort_values("year")
            if len(topic_data) < 2:
                continue

            predicted_values = None
            if forecast_method == "var":
                predicted_values = self._forecast_topic_var(topic_data, future_years)
            elif forecast_method in {"auto", "arima"}:
                try:
                    predicted_values = self._forecast_topic_arima(topic_data, future_years)
                except Exception:
                    predicted_values = self._forecast_topic_var(topic_data, future_years)
            else:
                try:
                    predicted_values = self._forecast_topic_arima(topic_data, future_years)
                except Exception:
                    predicted_values = self._forecast_topic_var(topic_data, future_years)

            if predicted_values is None:
                last_value = float(topic_data.sort_values("year")["share"].iloc[-1])
                predicted_values = np.repeat(last_value, future_years)

            last_year = int(topic_data["year"].max())
            topic_name = topic_names.get(topic, topic.replace("_", " "))
            latest_bridge = float(topic_data.sort_values("year")["bridge_score"].iloc[-1]) if "bridge_score" in topic_data.columns else 0.0
            latest_shift = bool(topic_data.sort_values("year")["potential_paradigm_shift"].iloc[-1]) if "potential_paradigm_shift" in topic_data.columns else False

            for offset, pred_share in enumerate(predicted_values, start=1):
                forecasts.append({
                    "year": int(last_year + offset),
                    "topic": topic,
                    "topic_name": topic_name,
                    "share": max(0.0, float(pred_share)),
                    "bridge_score": latest_bridge,
                    "potential_paradigm_shift": latest_shift,
                    "forecast_method": forecast_method.upper(),
                    "is_forecast": True,
                })

        if not forecasts:
            return historical_df

        forecast_df = pd.DataFrame(forecasts)
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
        return combined_df
