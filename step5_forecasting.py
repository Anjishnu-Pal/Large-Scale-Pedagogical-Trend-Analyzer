import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class TrendForecaster:
    def __init__(self):
        pass

    def calculate_historical_share(self, graphs_by_year, node_to_community):
        """
        Calculates the historical market share of each detected community over time.
        """
        yearly_topic_counts = []
        
        for year, graph in sorted(graphs_by_year.items()):
            total_nodes = len(graph.nodes)
            if total_nodes == 0:
                continue
                
            counts = {}
            for node in graph.nodes:
                topic = node_to_community.get(node, "Unknown")
                counts[topic] = counts.get(topic, 0) + 1
                
            for topic, count in counts.items():
                yearly_topic_counts.append({
                    'year': int(year),
                    'topic': topic,
                    'share': count / total_nodes,
                    'is_forecast': False
                })
                
        return pd.DataFrame(yearly_topic_counts)

    def forecast_trends(self, historical_df, future_years=3):
        """
        Applies a mathematical model (Linear Regression) to extend the 
        trajectories of topics into the future.
        """
        if historical_df.empty:
            return historical_df
            
        forecasts = []
        topics = historical_df['topic'].unique()
        
        for topic in topics:
            topic_data = historical_df[historical_df['topic'] == topic].sort_values('year')
            
            # Need at least two data points for a meaningful linear regression
            if len(topic_data) < 2:
                continue
                
            X = topic_data[['year']].values
            y = topic_data['share'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            last_year = int(topic_data['year'].max())
            future_X = np.array([[last_year + i] for i in range(1, future_years + 1)])
            future_y = model.predict(future_X)
            
            for year, pred_share in zip(future_X.flatten(), future_y):
                forecasts.append({
                    'year': int(year),
                    'topic': topic,
                    'share': max(0.0, float(pred_share)), # Cap bottom at 0% share
                    'is_forecast': True
                })
                
        if not forecasts:
            return historical_df
            
        forecast_df = pd.DataFrame(forecasts)
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
        return combined_df
