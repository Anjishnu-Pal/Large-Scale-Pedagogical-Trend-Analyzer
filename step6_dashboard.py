import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import re


def _topic_column(df):
    return 'topic_name' if 'topic_name' in df.columns else 'topic'


def _load_aliases():
    try:
        with open('aliases.json', 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return {}


ALIASES = _load_aliases()
TECHNICAL_HINTS = {
    'model', 'network', 'neural', 'transformer', 'attention',
    'embedding', 'retrieval', 'classifier', 'diffusion', 'graph', 'language',
    'vision', 'adaptation', 'optimization', 'architecture', 'inference',
    'representation', 'reinforcement', 'generation', 'knowledge', 'foundation',
    'semantic', 'adapter', 'llm', 'gnn', 'cnn', 'vit', 'rlhf', 'rag',
    'encoder', 'decoder', 'prompt', 'tuning', 'pretraining', 'fine', 'self',
    'supervised', 'multimodal', 'retrieval', 'policy', 'agent', 'bayesian'
}
GENERIC_WORDS = {
    'we', 'that', 'them', 'they', 'this', 'these', 'those', 'which', 'what',
    'where', 'when', 'why', 'how', 'benefits', 'properties', 'results', 'methods',
    'agreement', 'advances', 'energy', 'place', 'calculation', 'approach', 'paper',
    'signals', 'sources', 'rules', 'tools', 'us', 'n', 'li', 'it', 'people', 'things', 'problem'
}


def _normalize_label(text):
    normalized = str(text).replace('_', ' ').strip().lower()
    return re.sub(r'\s+', ' ', normalized)


def _is_research_topic_label(text):
    normalized = _normalize_label(text)
    if not normalized or normalized == 'unknown topic':
        return False
    if normalized in ALIASES.values() or normalized in ALIASES.keys():
        return True
    if any(word in normalized.split() for word in GENERIC_WORDS):
        return False
    if len(normalized.split()) == 1 and str(text).isupper() and 2 <= len(str(text)) <= 6:
        return True
    if len(normalized.split()) >= 2 and any(token in TECHNICAL_HINTS for token in normalized.split()):
        return True
    if 'language model' in normalized or 'neural network' in normalized or 'graph neural' in normalized:
        return True
    return False


def _research_only_df(df, topic_column):
    if df.empty:
        return df
    filtered = df[df[topic_column].astype(str).map(_is_research_topic_label)].copy()
    return filtered if not filtered.empty else df


def _forecast_mask(df):
    if 'is_forecast' not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    normalized = df['is_forecast'].astype(str).str.strip().str.lower()
    return normalized.isin({'true', '1', 'yes', 'y'})


def _latest_historical_year(df):
    historical = df[~_forecast_mask(df)] if 'is_forecast' in df.columns else df
    if historical.empty:
        return None
    return int(historical['year'].max())


def _get_latest_topic_rankings(df, topic_column):
    latest_year = _latest_historical_year(df)
    if latest_year is None:
        return pd.DataFrame(), pd.DataFrame()

    historical = df[~_forecast_mask(df)] if 'is_forecast' in df.columns else df.copy()
    historical = historical[historical[topic_column].astype(str).map(_is_research_topic_label)].copy()

    ranking_df = (
        historical.groupby(topic_column, as_index=False)
        .agg(
            total_share=('share', 'sum'),
            avg_share=('share', 'mean'),
            latest_share=('share', 'last'),
            latest_year=('year', 'max')
        )
        .sort_values(['total_share', 'avg_share', 'latest_year'], ascending=False)
    )

    latest_year_df = historical[historical['year'] == latest_year].copy()
    latest_year_df = latest_year_df.sort_values('share', ascending=False)

    forecast_df = df[_forecast_mask(df)] if 'is_forecast' in df.columns else pd.DataFrame()
    growth_rows = pd.DataFrame(columns=[topic_column, 'baseline_share', 'forecast_share', 'forecast_growth'])
    if not forecast_df.empty:
        baseline = historical[historical['year'] == latest_year][[topic_column, 'share']].rename(columns={'share': 'baseline_share'})
        future_max = forecast_df.groupby(topic_column, as_index=False)['share'].max().rename(columns={'share': 'forecast_share'})
        growth_rows = future_max.merge(baseline, on=topic_column, how='left')
        growth_rows['forecast_growth'] = growth_rows['forecast_share'] - growth_rows['baseline_share'].fillna(0.0)
        growth_rows = growth_rows[growth_rows[topic_column].astype(str).map(_is_research_topic_label)]
        growth_rows = growth_rows.sort_values('forecast_growth', ascending=False)

        if growth_rows.empty:
            growth_rows = future_max.merge(baseline, on=topic_column, how='left')
            growth_rows['forecast_growth'] = growth_rows['forecast_share'] - growth_rows['baseline_share'].fillna(0.0)
            growth_rows = growth_rows[growth_rows[topic_column].astype(str) != 'Unknown Topic']
            growth_rows = growth_rows.sort_values('forecast_growth', ascending=False)

    if latest_year_df.empty:
        latest_year_df = ranking_df.rename(columns={'latest_share': 'share'})

    # Return the score-ranked set in the first output so the dashboard can show
    # the strongest research topics even if the latest year is sparse.
    ranked_topics = ranking_df.copy()
    ranked_topics = ranked_topics.rename(columns={'latest_share': 'share'})
    ranked_topics['topic_score'] = (
        ranked_topics['total_share'] * 0.6
        + ranked_topics['avg_share'] * 0.4
    )
    ranked_topics = ranked_topics.sort_values(['topic_score', 'total_share', 'avg_share'], ascending=False)

    return ranked_topics.head(10), growth_rows.head(10)


def _trend_summary(row, growth_value=None):
    share = row['share']
    if growth_value is not None:
        if growth_value > 0:
            return 'This topic is rising.'
        if growth_value < 0:
            return 'This topic is declining.'
        return 'This topic is stable.'

    if share >= 0.15:
        return 'This topic is strongly present in the latest year.'
    if share >= 0.05:
        return 'This topic is moderately active.'
    return 'This topic is currently niche.'

def run_dashboard():
    st.set_page_config(page_title="Trend Analyzer Dashboard", layout="wide")
    st.title("Large-Scale Pedagogical Trend Analyzer")
    st.markdown("### Step 6: Reactive UI Rendering")
    st.markdown("This dashboard visualizes historical aggregations and predictive forecasts for academic sub-fields automatically detected via NLP and Community Detection.")

    data_file = 'trends_output.csv'
    
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        topic_column = _topic_column(df)
        view_mode = st.sidebar.radio(
            'Dashboard Mode',
            ['TrendAnalyzer', 'TrendForecaster'],
            index=0,
            help='Switch between backward-looking historical analysis and forward-looking forecasting.'
        )

        research_df = _research_only_df(df, topic_column)
        top_topics, forecast_growth = _get_latest_topic_rankings(research_df, topic_column)
        active_df = research_df if not research_df.empty else df
        
        st.header("Temporal Topic Trends (Historical & Forecast)")
        st.caption(f'Active mode: {view_mode}')

        if view_mode == 'TrendAnalyzer':
            historical_only = active_df[~_forecast_mask(active_df)] if 'is_forecast' in active_df.columns else active_df
            if not top_topics.empty:
                st.subheader("Top research topics by overall score")
                lead_topic = top_topics.iloc[0]
                st.success(
                    f"Top topic: {lead_topic[topic_column]} - score {lead_topic['topic_score']:.2f}, share {lead_topic['share']:.2%}. "
                    f"{_trend_summary(lead_topic)}"
                )
                cols = st.columns(2)
                left_panel = top_topics.head(5)
                right_panel = top_topics.iloc[5:10]
                with cols[0]:
                    st.write("Top 5 by overall score")
                    for _, row in left_panel.iterrows():
                        st.metric(label=row[topic_column], value=f"{row['topic_score']:.2f}", delta=f"share {row['share']:.2%}")
                with cols[1]:
                    st.write("Next topics")
                    for _, row in right_panel.iterrows():
                        st.metric(label=row[topic_column], value=f"{row['topic_score']:.2f}", delta=f"share {row['share']:.2%}")

            st.subheader("Plain-English trend notes")
            if not top_topics.empty:
                for _, row in top_topics.iterrows():
                    st.write(f"- {row[topic_column]}: score {row['topic_score']:.2f}, share {row['share']:.2%}")
            else:
                st.info("No high-confidence research topics were found in the current data slice.")

            fig = px.line(
                historical_only,
                x='year',
                y='share',
                color=topic_column,
                markers=True,
                title='Historical Topic Market Share',
                labels={"share": "Market Share", "year": "Year", topic_column: "Topic"}
            )
            fig.update_layout(hovermode='x unified', xaxis_type='linear')
            st.plotly_chart(fig, use_container_width=True)

        else:
            if not forecast_growth.empty:
                st.subheader("Biggest forecasted growth topics")
                growth_display = forecast_growth[[topic_column, 'baseline_share', 'forecast_share', 'forecast_growth']].copy()
                growth_display['forecast_growth'] = growth_display['forecast_growth'].map(lambda x: f"{x:.2%}")
                st.dataframe(growth_display.rename(columns={
                    topic_column: 'Topic',
                    'baseline_share': 'Latest Share',
                    'forecast_share': 'Peak Forecast Share',
                    'forecast_growth': 'Forecast Growth'
                }), use_container_width=True)
                best_growth = forecast_growth.iloc[0]
                st.info(
                    f"{best_growth[topic_column]} is rising. Its forecasted share increases from "
                    f"{best_growth['baseline_share']:.2%} to {best_growth['forecast_share']:.2%}."
                )
            else:
                st.info("Forecast data is not available for the current research-topic slice.")

            fig = px.line(
                active_df,
                x='year',
                y='share',
                color=topic_column,
                line_dash='is_forecast',
                markers=True,
                title='Topic Market Share Over Time (Dashed lines indicate Forecasts)',
                labels={"share": "Market Share", "year": "Year", topic_column: "Topic"}
            )
            fig.update_layout(hovermode='x unified', xaxis_type='linear')
            top_topic_names = set(top_topics[topic_column].tolist()) if not top_topics.empty else set()
            for trace in fig.data:
                if trace.name in top_topic_names:
                    trace.line.width = 4
                    trace.opacity = 1.0
                else:
                    trace.line.width = 1.5
                    trace.opacity = 0.2
            st.plotly_chart(fig, use_container_width=True)

        active_display = active_df if view_mode == 'TrendForecaster' else research_df
        if active_display.empty:
            active_display = df

        if view_mode == 'TrendForecaster' and not top_topics.empty:
            st.subheader("Top research topics in the latest year")
            st.dataframe(top_topics[[topic_column, 'share']].rename(columns={topic_column: 'Topic', 'share': 'Latest Share'}), use_container_width=True)

        st.subheader("Raw Aggregated Data")
        st.dataframe(active_display.sort_values(['year', topic_column]))
    else:
        st.warning(f"Data file '{data_file}' not found. Please run main.py first to generate the trends.")

if __name__ == "__main__":
    run_dashboard()
