from collections import defaultdict, Counter
from datetime import datetime, timedelta
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
import random

class ProductAnalyzer:
    """
    A comprehensive class for analyzing product attributes and generating insights.
    """
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'with', 'by', 'from', 'up', 'about', 'into', 'over',
            'after', 'of', 'this', 'that', 'these', 'those'
        }
        
    def extract_features(self, product: Dict) -> Dict:
        """
        Extract and normalize product features for analysis.
        """
        features = {}
        
        # Basic attributes
        for attr in ['category', 'brand', 'color', 'style', 'material', 'size']:
            if attr in product:
                features[attr] = product[attr].lower()
        
        # Numerical features
        features['price'] = float(product.get('price', 0))
        features['inventory'] = int(product.get('inventory', 0))
        features['likes'] = int(product.get('likes', 0))
        features['views'] = int(product.get('views', 0))
        
        # Engagement metrics
        features['engagement_rate'] = (
            features['likes'] / features['views'] 
            if features['views'] > 0 else 0
        )
        
        # Time-based features
        if 'created_at' in product:
            features['days_since_creation'] = (
                datetime.now() - product['created_at']
            ).days
        
        return features

    def calculate_product_scores(
        self, 
        product: Dict,
        category_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate various scoring metrics for a product.
        """
        features = self.extract_features(product)
        scores = {}
        
        # Popularity score
        scores['popularity'] = self._calculate_popularity_score(features)
        
        # Freshness score
        scores['freshness'] = self._calculate_freshness_score(features)
        
        # Scarcity score
        scores['scarcity'] = self._calculate_scarcity_score(features)
        
        # Value score
        scores['value'] = self._calculate_value_score(features)
        
        # Category relevance
        if category_weights:
            scores['category_relevance'] = (
                category_weights.get(features.get('category', ''), 0)
            )
        
        # Weighted average of all scores
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def _calculate_popularity_score(self, features: Dict) -> float:
        """Calculate popularity score based on engagement metrics."""
        engagement_weight = 0.7
        views_weight = 0.3
        
        engagement_score = min(features['engagement_rate'] * 10, 1)
        views_score = min(features['views'] / 1000, 1)
        
        return (
            engagement_score * engagement_weight +
            views_score * views_weight
        )
    
    def _calculate_freshness_score(self, features: Dict) -> float:
        """Calculate freshness score based on product age."""
        days_old = features.get('days_since_creation', 0)
        return math.exp(-days_old / 30)  # 30-day half-life
    
    def _calculate_scarcity_score(self, features: Dict) -> float:
        """Calculate scarcity score based on inventory levels."""
        inventory = features['inventory']
        if inventory == 0:
            return 0
        elif inventory < 500:
            return 1
        elif inventory < 1000:
            return 0.7
        elif inventory < 5000:
            return 0.4
        else:
            return 0.2
    
    def _calculate_value_score(self, features: Dict) -> float:
        """Calculate value score based on price and engagement."""
        price = features['price']
        engagement = features['engagement_rate']
        
        # Normalize price (assume max price of 1000)
        price_score = 1 - min(price / 1000, 1)
        
        return (price_score + engagement) / 2

class RecommendationEngine:
    """
    Advanced recommendation engine using multiple algorithms.
    """
    def __init__(self):
        self.analyzer = ProductAnalyzer()
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
    
    def get_recommendations(
        self,
        products: List[Dict],
        user_id: str,
        liked_products: List[Dict],
        recent_views: List[Dict] = None,
        n_recommendations: int = 300
    ) -> List[Dict]:
        """
        Get personalized recommendations using ensemble approach.
        """
        if not products:
            return []
        
        # Convert products to DataFrame for easier manipulation
        df = pd.DataFrame(products)
        
        # Calculate different types of recommendations
        content_based = self._get_content_based_recommendations(
            df, liked_products, n_recommendations
        )
        collaborative = self._get_collaborative_recommendations(
            df, user_id, n_recommendations
        )
        popularity_based = self._get_popularity_based_recommendations(
            df, n_recommendations
        )
        
        # Ensemble the recommendations
        final_recommendations = self._ensemble_recommendations(
            [content_based, collaborative, popularity_based],
            weights=[0.4, 0.4, 0.2],
            n_recommendations=n_recommendations
        )
        
        # Apply diversity optimization
        final_recommendations = self._diversify_recommendations(
            final_recommendations,
            n_recommendations
        )
        
        return final_recommendations
    
    def _get_content_based_recommendations(
        self,
        df: pd.DataFrame,
        liked_products: List[Dict],
        n_recommendations: int
    ) -> List[Dict]:
        """
        Get content-based recommendations using TF-IDF and cosine similarity.
        """
        if not liked_products:
            return []
        
        # Prepare text features
        df['text_features'] = df.apply(
            lambda x: ' '.join([
                str(x.get('name', '')),
                str(x.get('description', '')),
                str(x.get('category', '')),
                str(x.get('brand', ''))
            ]),
            axis=1
        )
        
        # Calculate TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(df['text_features'])
        
        # Calculate similarity with liked products
        liked_indices = [
            df[df['_id'] == p['_id']].index[0]
            for p in liked_products
            if p['_id'] in df['_id'].values
        ]
        
        if not liked_indices:
            return []
        
        liked_vectors = tfidf_matrix[liked_indices]
        similarities = cosine_similarity(tfidf_matrix, liked_vectors)
        
        # Calculate average similarity scores
        avg_similarities = similarities.mean(axis=1)
        
        # Get top recommendations
        top_indices = np.argsort(avg_similarities)[::-1][:n_recommendations]
        
        return df.iloc[top_indices].to_dict('records')
    
    def _get_collaborative_recommendations(
        self,
        df: pd.DataFrame,
        user_id: str,
        n_recommendations: int
    ) -> List[Dict]:
        """
        Get collaborative filtering recommendations using user similarity.
        """
        # This would typically involve user-item interaction matrix
        # For simplicity, return popularity-based recommendations
        return self._get_popularity_based_recommendations(
            df,
            n_recommendations
        )
    
    def _get_popularity_based_recommendations(
        self,
        df: pd.DataFrame,
        n_recommendations: int
    ) -> List[Dict]:
        """
        Get recommendations based on popularity and engagement.
        """
        df['popularity_score'] = df.apply(
            lambda x: self.analyzer.calculate_product_scores(x)['popularity'],
            axis=1
        )
        
        return df.nlargest(
            n_recommendations,
            'popularity_score'
        ).to_dict('records')
    
    def _ensemble_recommendations(
        self,
        recommendation_lists: List[List[Dict]],
        weights: List[float],
        n_recommendations: int
    ) -> List[Dict]:
        """
        Combine multiple recommendation lists using weighted ensemble.
        """
        product_scores = defaultdict(float)
        
        for recs, weight in zip(recommendation_lists, weights):
            for rank, product in enumerate(recs):
                score = weight * (1 / (rank + 1))
                product_scores[product['_id']] += score
        
        # Sort products by ensemble score
        sorted_products = sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top N products
        top_product_ids = [p[0] for p in sorted_products[:n_recommendations]]
        
        # Convert back to product dictionaries
        df = pd.DataFrame(recommendation_lists[0])
        return df[df['_id'].isin(top_product_ids)].to_dict('records')
    
    def _diversify_recommendations(
        self,
        recommendations: List[Dict],
        n_recommendations: int
    ) -> List[Dict]:
        """
        Diversify recommendations using maximal marginal relevance.
        """
        if not recommendations:
            return []
        
        selected = []
        candidates = recommendations.copy()
        
        while len(selected) < n_recommendations and candidates:
            # Calculate diversity scores
            diversity_scores = []
            for candidate in candidates:
                score = self._calculate_diversity_score(
                    candidate,
                    selected
                )
                diversity_scores.append(score)
            
            # Select the most diverse candidate
            best_idx = np.argmax(diversity_scores)
            selected.append(candidates.pop(best_idx))
        
        return selected
    
    def _calculate_diversity_score(
        self,
        candidate: Dict,
        selected: List[Dict]
    ) -> float:
        """
        Calculate diversity score for a candidate product.
        """
        if not selected:
            return 1.0
        
        # Calculate feature similarity with selected products
        similarities = []
        candidate_features = self.analyzer.extract_features(candidate)
        
        for product in selected:
            product_features = self.analyzer.extract_features(product)
            similarity = self._calculate_feature_similarity(
                candidate_features,
                product_features
            )
            similarities.append(similarity)
        
        # Return inverse of maximum similarity
        return 1 - max(similarities)
    
    def _calculate_feature_similarity(
        self,
        features1: Dict,
        features2: Dict
    ) -> float:
        """
        Calculate similarity between two feature sets.
        """
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
        
        similarities = []
        for feature in common_features:
            if isinstance(features1[feature], (int, float)):
                # Normalize numerical features
                max_val = max(features1[feature], features2[feature])
                if max_val == 0:
                    similarities.append(1.0)
                else:
                    similarities.append(
                        1 - abs(features1[feature] - features2[feature]) / max_val
                    )
            else:
                # Categorical features
                similarities.append(
                    1.0 if features1[feature] == features2[feature] else 0.0
                )
        
        return sum(similarities) / len(similarities)

class TrendAnalyzer:
    """
    Advanced trend analysis and prediction.
    """
    def __init__(self):
        self.analyzer = ProductAnalyzer()
    
    def get_trending_products(
        self,
        products: List[Dict],
        timeframe_days: int = 7,
        n_products: int = 10,
        min_views: int = 100
    ) -> List[Dict]:
        """
        Get trending products using advanced trend detection.
        """
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        
        # Calculate trend scores
        trend_scores = []
        for product in products:
            if product.get('views', 0) >= min_views:
                score = self._calculate_trend_score(
                    product,
                    cutoff_date
                )
                trend_scores.append((product, score))
        
        # Sort by trend score
        trend_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in trend_scores[:n_products]]
    
    def _calculate_trend_score(
        self,
        product: Dict,
        cutoff_date: datetime
    ) -> float:
        """
        Calculate trend score using multiple factors.
        """
        features = self.analyzer.extract_features(product)
        
        # Base engagement score
        engagement_score = features['engagement_rate']
        
        # Recency boost
        days_old = features.get('days_since_creation', 0)
        recency_boost = math.exp(-days_old / 7)  # 7-day half-life
        
        # Velocity boost (rate of engagement change)
        velocity_boost = self._calculate_velocity_boost(product)
        
        # Combine scores
        trend_score = (
            engagement_score * 0.4 +
            recency_boost * 0.3 +
            velocity_boost * 0.3
        )
        
        return trend_score
    
    def _calculate_velocity_boost(self, product: Dict) -> float:
        """
        Calculate engagement velocity boost.
        """
        recent_likes = product.get('recent_likes', [])
        if not recent_likes:
            return 0.0
        
        # Calculate rate of change in likes
        total_likes = len(recent_likes)
        days_span = (max(recent_likes) - min(recent_likes)).days
        
        if days_span == 0:
            return 1.0
        
        daily_rate = total_likes / days_span
        return min(daily_rate / 10, 1.0)  # Normalize to [0,1]

'''def initialize_recommendation_system() -> Tuple[RecommendationEngine, TrendAnalyzer]:
    
    Initialize the recommendation system components.
    
    return RecommendationEngine(), TrendAnalyzer()'''

# Main recommendation function for the application
def get_recommendations(
    products: List[Dict],
    user_id: str,
    liked_products: List[Dict],
    recent_views: List[Dict] = None,
    n_recommendations: int = 500
) -> List[Dict]:
    """
    Get personalized product recommendations using all five classes.
    """
    # Initialize all components of the recommendation system
    engine, personalization_engine, trend_analyzer, seasonality_analyzer, inventory_optimizer = initialize_recommendation_system()

    # Step 1: Use PersonalizationEngine to build a user profile
    user_profile = personalization_engine.build_user_profile(
        user_id=user_id,
        liked_products=liked_products,
        viewed_products=recent_views if recent_views else [],
        purchase_history=[],  # Assuming no purchase history for simplicity
        browse_history=[]     # Assuming no browse history for simplicity
    )

    # Step 2: Use TrendAnalyzer to get trending products
    trending_products = trend_analyzer.get_trending_products(
        products=products,
        timeframe_days=7,  # Analyze trends over the last 7 days
        n_products=n_recommendations,
        min_views=100
    )

    # Step 3: Use SeasonalityAnalyzer to analyze seasonal trends
    # For simplicity, we'll analyze seasonality for the first product
    if products:
        seasonality_data = seasonality_analyzer.analyze_seasonality(
            product_history=[],  # Assuming no product history for simplicity
            timeframe_days=365
        )
    else:
        seasonality_data = None

    # Step 4: Use InventoryOptimizer to calculate optimal inventory levels
    # For simplicity, we'll calculate inventory for the first product
    if products:
        inventory_data = inventory_optimizer.calculate_optimal_inventory(
            product=products[0],
            sales_history=[],  # Assuming no sales history for simplicity
            seasonality_data=seasonality_data
        )
    else:
        inventory_data = None

    # Step 5: Use RecommendationEngine to get personalized recommendations
    personalized_recommendations = engine.get_recommendations(
        products=products,
        user_id=user_id,
        liked_products=liked_products,
        recent_views=recent_views,
        n_recommendations=n_recommendations
    )

    # Step 6: Combine all results (trending, personalized, etc.) into a single list
    # For simplicity, we'll prioritize personalized recommendations
    final_recommendations = personalized_recommendations

    # Optionally, you can add trending products to the final recommendations
    final_recommendations.extend(trending_products)

    # Remove duplicates (if any)
    unique_recommendations = []
    seen_ids = set()
    for product in final_recommendations:
        if product['_id'] not in seen_ids:
            unique_recommendations.append(product)
            seen_ids.add(product['_id'])

    # Return the final list of recommendations
    return unique_recommendations[:n_recommendations]

class PersonalizationEngine:
    """
    Advanced personalization engine for user behavior analysis and preference learning.
    """
    def __init__(self):
        self.analyzer = ProductAnalyzer()
        self.preference_decay_rate = 0.1  # Decay rate for old preferences
        
    def build_user_profile(
        self,
        user_id: str,
        liked_products: List[Dict],
        viewed_products: List[Dict],
        purchase_history: List[Dict],
        browse_history: List[Dict]
    ) -> Dict:
        """
        Build comprehensive user profile from multiple data sources.
        """
        profile = {
            'user_id': user_id,
            'category_preferences': self._calculate_category_preferences(
                liked_products,
                viewed_products,
                purchase_history
            ),
            'price_sensitivity': self._calculate_price_sensitivity(
                purchase_history,
                browse_history
            ),
            'brand_preferences': self._calculate_brand_preferences(
                liked_products,
                purchase_history
            ),
            'style_preferences': self._analyze_style_preferences(
                liked_products,
                purchase_history,
                viewed_products
            ),
            'engagement_patterns': self._analyze_engagement_patterns(
                liked_products,
                viewed_products,
                browse_history
            )
        }
        
        # Add behavioral segments
        profile['segments'] = self._determine_user_segments(profile)
        
        return profile
    
    def _calculate_category_preferences(
        self,
        liked_products: List[Dict],
        viewed_products: List[Dict],
        purchase_history: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate weighted category preferences from user interactions.
        """
        category_scores = defaultdict(float)
        
        # Weight different types of interactions
        weights = {
            'purchase': 1.0,
            'like': 0.7,
            'view': 0.3
        }
        
        # Process purchases
        for product in purchase_history:
            category = product.get('category', '')
            if category:
                category_scores[category] += weights['purchase']
        
        # Process likes
        for product in liked_products:
            category = product.get('category', '')
            if category:
                category_scores[category] += weights['like']
        
        # Process views
        for product in viewed_products:
            category = product.get('category', '')
            if category:
                category_scores[category] += weights['view']
        
        # Normalize scores
        total_score = sum(category_scores.values()) or 1
        return {
            category: score / total_score
            for category, score in category_scores.items()
        }
    
    def _calculate_price_sensitivity(
        self,
        purchase_history: List[Dict],
        browse_history: List[Dict]
    ) -> Dict:
        """
        Calculate user's price sensitivity metrics.
        """
        if not purchase_history and not browse_history:
            return {
                'sensitivity_score': 0.5,  # Default middle value
                'preferred_price_range': None,
                'max_observed_purchase': None
            }
        
        purchased_prices = [p.get('price', 0) for p in purchase_history]
        browsed_prices = [p.get('price', 0) for p in browse_history]
        
        # Calculate metrics
        metrics = {
            'max_purchase_price': max(purchased_prices) if purchased_prices else 0,
            'avg_purchase_price': (
                sum(purchased_prices) / len(purchased_prices)
                if purchased_prices else 0
            ),
            'avg_browse_price': (
                sum(browsed_prices) / len(browsed_prices)
                if browsed_prices else 0
            )
        }
        
        # Determine preferred price range
        all_prices = purchased_prices + browsed_prices
        if all_prices:
            metrics['preferred_price_range'] = {
                'min': np.percentile(all_prices, 25),
                'max': np.percentile(all_prices, 75)
            }
        
        # Calculate sensitivity score
        if purchased_prices:
            price_variance = np.std(purchased_prices)
            max_price = max(purchased_prices)
            metrics['sensitivity_score'] = (
                1 - min(price_variance / max_price, 1)
                if max_price > 0 else 0.5
            )
        else:
            metrics['sensitivity_score'] = 0.5
        
        return metrics
    
    def _calculate_brand_preferences(
        self,
        liked_products: List[Dict],
        purchase_history: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate brand preferences with confidence scores.
        """
        brand_interactions = defaultdict(lambda: {
            'purchase_count': 0,
            'like_count': 0,
            'total_spend': 0
        })
        
        # Process purchases
        for product in purchase_history:
            brand = product.get('brand', '')
            if brand:
                brand_interactions[brand]['purchase_count'] += 1
                brand_interactions[brand]['total_spend'] += product.get('price', 0)
        
        # Process likes
        for product in liked_products:
            brand = product.get('brand', '')
            if brand:
                brand_interactions[brand]['like_count'] += 1
        
        # Calculate preference scores
        brand_scores = {}
        total_interactions = sum(
            data['purchase_count'] + data['like_count']
            for data in brand_interactions.values()
        )
        
        if total_interactions > 0:
            for brand, data in brand_interactions.items():
                interaction_score = (
                    data['purchase_count'] * 2 + data['like_count']
                ) / total_interactions
                spend_score = data['total_spend'] / (
                    sum(d['total_spend'] for d in brand_interactions.values()) or 1
                )
                brand_scores[brand] = (interaction_score + spend_score) / 2
        
        return brand_scores
    
    def _analyze_style_preferences(
        self,
        liked_products: List[Dict],
        purchase_history: List[Dict],
        viewed_products: List[Dict]
    ) -> Dict:
        """
        Analyze user's style preferences across multiple dimensions.
        """
        style_data = {
            'colors': defaultdict(int),
            'materials': defaultdict(int),
            'patterns': defaultdict(int),
            'styles': defaultdict(int)
        }
        
        # Weight different interaction types
        weights = {
            'purchase': 3,
            'like': 2,
            'view': 1
        }
        
        # Process all product interactions
        for product_list, interaction_type in [
            (purchase_history, 'purchase'),
            (liked_products, 'like'),
            (viewed_products, 'view')
        ]:
            weight = weights[interaction_type]
            for product in product_list:
                # Update color preferences
                if 'color' in product:
                    style_data['colors'][product['color']] += weight
                
                # Update material preferences
                if 'material' in product:
                    style_data['materials'][product['material']] += weight
                
                # Update pattern preferences
                if 'pattern' in product:
                    style_data['patterns'][product['pattern']] += weight
                
                # Update style preferences
                if 'style' in product:
                    style_data['styles'][product['style']] += weight
        
        # Normalize preferences
        preferences = {}
        for category, counts in style_data.items():
            total = sum(counts.values()) or 1
            preferences[category] = {
                key: count / total
                for key, count in counts.items()
            }
        
        return preferences
    
    def _analyze_engagement_patterns(
        self,
        liked_products: List[Dict],
        viewed_products: List[Dict],
        browse_history: List[Dict]
    ) -> Dict:
        """
        Analyze user engagement patterns and behavior.
        """
        patterns = {
            'engagement_rate': self._calculate_engagement_rate(
                liked_products,
                viewed_products
            ),
            'browse_depth': self._calculate_browse_depth(browse_history),
            'category_exploration': self._calculate_category_exploration(
                viewed_products
            ),
            'time_patterns': self._analyze_time_patterns(browse_history)
        }
        
        return patterns
    
    def _calculate_engagement_rate(
        self,
        liked_products: List[Dict],
        viewed_products: List[Dict]
    ) -> float:
        """
        Calculate user's engagement rate.
        """
        if not viewed_products:
            return 0.0
        
        return len(liked_products) / len(viewed_products)
    
    def _calculate_browse_depth(
        self,
        browse_history: List[Dict]
    ) -> Dict:
        """
        Analyze browsing depth and patterns.
        """
        if not browse_history:
            return {
                'average_session_depth': 0,
                'max_session_depth': 0,
                'typical_session_length': 0
            }
        
        # Group browsing events into sessions
        sessions = self._group_into_sessions(browse_history)
        
        depths = []
        lengths = []
        for session in sessions:
            depths.append(len(session))
            if len(session) > 1:
                session_length = (
                    session[-1]['timestamp'] -
                    session[0]['timestamp']
                ).seconds / 60  # Convert to minutes
                lengths.append(session_length)
        
        return {
            'average_session_depth': np.mean(depths) if depths else 0,
            'max_session_depth': max(depths) if depths else 0,
            'typical_session_length': np.median(lengths) if lengths else 0
        }
    
    def _group_into_sessions(
        self,
        browse_history: List[Dict],
        session_timeout: int = 30
    ) -> List[List[Dict]]:
        """
        Group browsing events into sessions.
        """
        if not browse_history:
            return []
        
        sorted_history = sorted(
            browse_history,
            key=lambda x: x.get('timestamp', datetime.min)
        )
        
        sessions = []
        current_session = [sorted_history[0]]
        
        for event in sorted_history[1:]:
            last_timestamp = current_session[-1].get(
                'timestamp',
                datetime.min
            )
            current_timestamp = event.get('timestamp', datetime.min)
            
            if (
                current_timestamp - last_timestamp
            ).seconds / 60 > session_timeout:
                sessions.append(current_session)
                current_session = []
            
            current_session.append(event)
        
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def _calculate_category_exploration(
        self,
        viewed_products: List[Dict]
    ) -> Dict:
        """
        Analyze category exploration patterns.
        """
        if not viewed_products:
            return {
                'unique_categories': 0,
                'category_entropy': 0,
                'exploration_score': 0
            }
        
        # Count category occurrences
        category_counts = Counter(
            p.get('category', 'unknown')
            for p in viewed_products
        )
        
        # Calculate metrics
        total_views = len(viewed_products)
        probabilities = [
            count / total_views
            for count in category_counts.values()
        ]
        
        entropy = -sum(
            p * math.log(p)
            for p in probabilities
        )
        
        return {
            'unique_categories': len(category_counts),
            'category_entropy': entropy,
            'exploration_score': entropy / math.log(len(category_counts))
            if len(category_counts) > 1 else 0
        }
    
    def _analyze_time_patterns(
        self,
        browse_history: List[Dict]
    ) -> Dict:
        """
        Analyze temporal browsing patterns.
        """
        if not browse_history:
            return {
                'peak_hours': [],
                'active_days': [],
                'regularity_score': 0
            }
        
        # Extract timestamps
        timestamps = [
            event.get('timestamp', datetime.min)
            for event in browse_history
        ]
        
        # Analyze hourly patterns
        hours = [t.hour for t in timestamps]
        hour_counts = Counter(hours)
        peak_hours = sorted(
            hour_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Analyze daily patterns
        days = [t.weekday() for t in timestamps]
        day_counts = Counter(days)
        active_days = [
            day for day, count in day_counts.items()
            if count > len(timestamps) / 14  # Active if >50% of average
        ]
        
        # Calculate regularity score
        hour_entropy = self._calculate_entropy(hour_counts.values())
        day_entropy = self._calculate_entropy(day_counts.values())
        regularity_score = 1 - (
            (hour_entropy / math.log(24) + day_entropy / math.log(7)) / 2
        )
        
        return {
            'peak_hours': [hour for hour, _ in peak_hours],
            'active_days': active_days,
            'regularity_score': regularity_score
        }
    
    def _calculate_entropy(self, counts) -> float:
        """
        Calculate entropy of a distribution.
        """
        total = sum(counts)
        if total == 0:
            return 0
        
        probabilities = [count / total for count in counts]
        return -sum(
            p * math.log(p)
            for p in probabilities
            if p > 0
        )
    
    def _determine_user_segments(self, profile: Dict) -> List[str]:
        """
        Determine user segments based on behavior and preferences.
        """
        segments = []
        
        # Price sensitivity segments
        price_sensitivity = profile['price_sensitivity']['sensitivity_score']
        if price_sensitivity > 0.7:
            segments.append('price_sensitive')
        elif price_sensitivity < 0.3:
            segments.append('luxury_oriented')
        
        # Brand loyalty segments
        brand_preferences = profile['brand_preferences']
        if any(score > 0.4 for score in brand_preferences.values()):
            segments.append('brand_loyal')
        
        # Engagement segments
        engagement_rate = profile['engagement_patterns']['engagement_rate']
        if engagement_rate > 0.2:
            segments.append('high_engager')
        
        # Category explorer segments
        category_exploration = (
            profile['engagement_patterns']
            ['category_exploration']
            ['exploration_score']
        )
        if category_exploration > 0.7:
            segments.append('category_explorer')
        
        return segments

# Utility functions for the recommendation system
def calculate_diversity_metrics(recommendations: List[Dict]) -> Dict:
    """
    Calculate diversity metrics for a set of recommendations.
    """
    if not recommendations:
        return {
            'category_diversity': 0,
            'price_range_diversity': 0,
            'brand_diversity': 0,
            'overall_diversity': 0
        }
    
    # Extract attributes
    categories = [r.get('category', '') for r in recommendations]
    brands = [r.get('brand', '') for r in recommendations]
    prices = [r.get('price', 0) for r in recommendations]
    
    # Calculate diversity metrics
    metrics = {
        'category_diversity': _calculate_attribute_diversity(categories),
        'price_range_diversity': _calculate_numerical_diversity(prices),
        'brand_diversity': _calculate_attribute_diversity(brands),
    }
    
    # Calculate overall diversity
    metrics['overall_diversity'] = sum(metrics.values()) / len(metrics)
    
    return metrics

def _calculate_attribute_diversity(attributes: List[str]) -> float:
    """
    Calculate diversity score for categorical attributes using Shannon entropy.
    """
    if not attributes:
        return 0.0
    
    # Count occurrences
    counts = Counter(attributes)
    total = len(attributes)
    
    # Calculate entropy
    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    
    # Normalize by maximum possible entropy
    max_entropy = math.log(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0

def _calculate_numerical_diversity(values: List[float]) -> float:
    """
    Calculate diversity score for numerical values using range and distribution.
    """
    if not values:
        return 0.0
    
    # Calculate range metrics
    min_val = min(values)
    max_val = max(values)
    range_size = max_val - min_val if max_val > min_val else 1
    
    # Calculate distribution metrics
    quartiles = np.percentile(values, [25, 50, 75])
    iqr = quartiles[2] - quartiles[0]
    
    # Combine range and distribution metrics
    range_diversity = iqr / range_size if range_size > 0 else 0
    
    return range_diversity

class SeasonalityAnalyzer:
    """
    Analyze and predict seasonal trends in product performance.
    """
    def __init__(self):
        self.min_data_points = 30
        self.seasonality_threshold = 0.1
    
    def analyze_seasonality(
        self,
        product_history: List[Dict],
        timeframe_days: int = 365
    ) -> Dict:
        """
        Analyze seasonal patterns in product performance.
        """
        if len(product_history) < self.min_data_points:
            return {
                'has_seasonality': False,
                'seasonal_pattern': None,
                'peak_periods': [],
                'confidence': 0
            }
        
        # Convert history to time series
        dates = [entry['date'] for entry in product_history]
        values = [entry['value'] for entry in product_history]
        
        # Detect seasonal patterns
        seasonal_patterns = self._detect_seasonal_patterns(dates, values)
        
        # Identify peak periods
        peak_periods = self._identify_peak_periods(dates, values)
        
        # Calculate confidence
        confidence = self._calculate_seasonality_confidence(seasonal_patterns)
        
        return {
            'has_seasonality': confidence > self.seasonality_threshold,
            'seasonal_pattern': seasonal_patterns,
            'peak_periods': peak_periods,
            'confidence': confidence
        }
    
    def _detect_seasonal_patterns(
        self,
        dates: List[datetime],
        values: List[float]
    ) -> Dict:
        """
        Detect seasonal patterns in time series data.
        """
        # Convert to weekly aggregates
        weekly_data = self._aggregate_to_weekly(dates, values)
        
        # Calculate year-over-year correlation
        yoy_correlation = self._calculate_yoy_correlation(weekly_data)
        
        # Identify repeated patterns
        patterns = self._identify_repeated_patterns(weekly_data)
        
        return {
            'weekly_pattern': patterns.get('weekly'),
            'monthly_pattern': patterns.get('monthly'),
            'quarterly_pattern': patterns.get('quarterly'),
            'yoy_correlation': yoy_correlation
        }
    
    def _aggregate_to_weekly(
        self,
        dates: List[datetime],
        values: List[float]
    ) -> Dict[int, float]:
        """
        Aggregate daily data to weekly averages.
        """
        weekly_data = defaultdict(list)
        
        for date, value in zip(dates, values):
            week_number = date.isocalendar()[1]
            weekly_data[week_number].append(value)
        
        return {
            week: np.mean(values)
            for week, values in weekly_data.items()
        }
    
    def _calculate_yoy_correlation(
        self,
        weekly_data: Dict[int, float]
    ) -> float:
        """
        Calculate year-over-year correlation coefficient.
        """
        if len(weekly_data) < 52:
            return 0.0
        
        # Split data into years
        year1 = [weekly_data.get(i, 0) for i in range(1, 53)]
        year2 = [weekly_data.get(i + 52, 0) for i in range(1, 53)]
        
        # Calculate correlation
        correlation = np.corrcoef(year1, year2)[0, 1]
        return max(0, correlation)  # Only consider positive correlation
    
    def _identify_repeated_patterns(
        self,
        weekly_data: Dict[int, float]
    ) -> Dict:
        """
        Identify repeated patterns at different time scales.
        """
        patterns = {}
        
        # Weekly patterns (day of week effects)
        patterns['weekly'] = self._analyze_weekly_pattern(weekly_data)
        
        # Monthly patterns
        patterns['monthly'] = self._analyze_monthly_pattern(weekly_data)
        
        # Quarterly patterns
        patterns['quarterly'] = self._analyze_quarterly_pattern(weekly_data)
        
        return patterns
    
    def _analyze_weekly_pattern(
        self,
        weekly_data: Dict[int, float]
    ) -> Dict:
        """
        Analyze patterns within weeks.
        """
        return {
            'peak_days': self._find_peak_periods(weekly_data, 7),
            'variance': np.var(list(weekly_data.values()))
        }
    
    def _analyze_monthly_pattern(
        self,
        weekly_data: Dict[int, float]
    ) -> Dict:
        """
        Analyze patterns within months.
        """
        monthly_data = self._aggregate_to_monthly(weekly_data)
        return {
            'peak_months': self._find_peak_periods(monthly_data, 12),
            'variance': np.var(list(monthly_data.values()))
        }
    
    def _analyze_quarterly_pattern(
        self,
        weekly_data: Dict[int, float]
    ) -> Dict:
        """
        Analyze patterns within quarters.
        """
        quarterly_data = self._aggregate_to_quarterly(weekly_data)
        return {
            'peak_quarters': self._find_peak_periods(quarterly_data, 4),
            'variance': np.var(list(quarterly_data.values()))
        }
    
    def _find_peak_periods(
        self,
        data: Dict[int, float],
        period_length: int
    ) -> List[int]:
        """
        Find peak periods in a cyclic pattern.
        """
        if not data:
            return []
        
        # Calculate mean and standard deviation
        values = np.array(list(data.values()))
        mean = np.mean(values)
        std = np.std(values)
        
        # Find periods above mean + 0.5 std
        peak_threshold = mean + 0.5 * std
        peaks = [
            period for period, value in data.items()
            if value >= peak_threshold
        ]
        
        return sorted(peaks)
    
    def _calculate_seasonality_confidence(
        self,
        patterns: Dict
    ) -> float:
        """
        Calculate confidence in seasonality detection.
        """
        confidence_scores = []
        
        # YoY correlation confidence
        if 'yoy_correlation' in patterns:
            confidence_scores.append(patterns['yoy_correlation'])
        
        # Pattern strength confidence
        for pattern_type in ['weekly', 'monthly', 'quarterly']:
            if pattern_type in patterns:
                pattern_data = patterns[pattern_type]
                if pattern_data and 'variance' in pattern_data:
                    confidence_scores.append(
                        min(pattern_data['variance'] * 2, 1.0)
                    )
        
        return np.mean(confidence_scores) if confidence_scores else 0.0

class InventoryOptimizer:
    """
    Optimize inventory levels based on demand prediction and seasonality.
    """
    def __init__(self):
        self.seasonality_analyzer = SeasonalityAnalyzer()
    
    def calculate_optimal_inventory(
        self,
        product: Dict,
        sales_history: List[Dict],
        seasonality_data: Dict = None
    ) -> Dict:
        """
        Calculate optimal inventory levels considering multiple factors.
        """
        # Get or calculate seasonality
        if not seasonality_data:
            seasonality_data = self.seasonality_analyzer.analyze_seasonality(
                sales_history
            )
        
        # Calculate base demand
        base_demand = self._calculate_base_demand(sales_history)
        
        # Apply seasonal adjustments
        seasonal_demand = self._apply_seasonal_adjustments(
            base_demand,
            seasonality_data
        )
        
        # Calculate safety stock
        safety_stock = self._calculate_safety_stock(
            sales_history,
            seasonal_demand
        )
        
        # Calculate reorder point
        reorder_point = self._calculate_reorder_point(
            seasonal_demand,
            safety_stock,
            product
        )
        
        return {
            'optimal_stock': int(seasonal_demand + safety_stock),
            'reorder_point': int(reorder_point),
            'safety_stock': int(safety_stock),
            'expected_demand': seasonal_demand,
            'confidence': self._calculate_confidence(
                sales_history,
                seasonality_data
            )
        }
    
    def _calculate_base_demand(
        self,
        sales_history: List[Dict]
    ) -> float:
        """
        Calculate base demand using weighted average of historical sales.
        """
        if not sales_history:
            return 0.0
        
        # Calculate weighted average with more recent sales weighted higher
        total_weight = 0
        weighted_sum = 0
        
        for i, sale in enumerate(sales_history):
            weight = math.exp(-i / len(sales_history))  # Exponential decay
            weighted_sum += sale['quantity'] * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _apply_seasonal_adjustments(
        self,
        base_demand: float,
        seasonality_data: Dict
    ) -> float:
        """
        Apply seasonal adjustments to base demand.
        """
        if not seasonality_data['has_seasonality']:
            return base_demand
        
        # Get current period's seasonal factor
        current_period = datetime.now().isocalendar()[1]  # Current week
        seasonal_pattern = seasonality_data['seasonal_pattern']
        
        # Apply seasonal adjustment if available
        if (
            seasonal_pattern and
            'weekly_pattern' in seasonal_pattern and
            current_period in seasonal_pattern['weekly_pattern']['peak_days']
        ):
            # Increase demand for peak periods
            return base_demand * 1.5
        
        return base_demand
    
    def _calculate_safety_stock(
        self,
        sales_history: List[Dict],
        seasonal_demand: float
    ) -> float:
        """
        Calculate safety stock based on demand variability.
        """
        if not sales_history:
            return seasonal_demand * 0.5  # Default safety stock
        
        # Calculate demand variability
        quantities = [sale['quantity'] for sale in sales_history]
        demand_std = np.std(quantities)
        
        # Calculate service factor based on desired service level (e.g., 95%)
        service_factor = 1.645  # For 95% service level
        
        # Calculate lead time in days
        lead_time = 7  # Example lead time of 7 days
        
        # Calculate safety stock
        safety_stock = service_factor * demand_std * math.sqrt(lead_time)
        
        return safety_stock
    
    def _calculate_reorder_point(
        self,
        seasonal_demand: float,
        safety_stock: float,
        product: Dict
    ) -> float:
        """
        Calculate reorder point considering lead time and safety stock.
        """
        # Get lead time in days
        lead_time = product.get('lead_time', 7)  # Default to 7 days
        
        # Calculate daily demand
        daily_demand = seasonal_demand / 30  # Assuming monthly demand
        
        # Calculate reorder point
        reorder_point = (daily_demand * lead_time) + safety_stock
        
        return reorder_point
    
    def _calculate_confidence(
        self,
        sales_history: List[Dict],
        seasonality_data: Dict
    ) -> float:
        """
        Calculate confidence in inventory recommendations.
        """
        if not sales_history:
            return 0.0
        
        confidence_factors = []
        
        # History length confidence
        history_confidence = min(len(sales_history) / 90, 1.0)  # 90 days for full confidence
        confidence_factors.append(history_confidence)
        
        # Seasonality confidence
        if seasonality_data['has_seasonality']:
            confidence_factors.append(seasonality_data['confidence'])
        
        # Demand variability confidence
        quantities = [sale['quantity'] for sale in sales_history]
        cv = np.std(quantities) / np.mean(quantities) if np.mean(quantities) > 0 else 0
        variability_confidence = 1 - min(cv, 1.0)
        confidence_factors.append(variability_confidence)
        
        return np.mean(confidence_factors)

# Initialize all components
def initialize_recommendation_system() -> Tuple[
    RecommendationEngine,
    PersonalizationEngine,
    TrendAnalyzer,
    SeasonalityAnalyzer,
    InventoryOptimizer
]:
    """
    Initialize all components of the recommendation system.
    """
    return (
        RecommendationEngine(),
        PersonalizationEngine(),
        TrendAnalyzer(),
        SeasonalityAnalyzer(),
        InventoryOptimizer()
    )