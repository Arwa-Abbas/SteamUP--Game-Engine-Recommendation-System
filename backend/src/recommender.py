import numpy as np
import pickle
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime
import time
import math

class GameRecommender:
    """Memory-efficient recommendation system with PROPER similarity measures"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.games = []
        self.game_features = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=800, ngram_range=(1, 2), min_df=2)
        
        # Store only top-K similarities
        self.top_k_similarities = {
            'cosine': {},
            'pearson': {},
            'euclidean': {},
            'jaccard': {}
        }
        
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Settings
        self.TOP_K = 50  # Store only top 50 similarities per game
        self.CHUNK_SIZE = 500  # Process games in chunks (reduced for memory)
        
        # Load games and index on initialization
        self.load_games_chunked()
        self._build_game_index()
    
    def _build_game_index(self):
        """Build index for game titles"""
        self.game_index = {}
        for i, game in enumerate(self.games):
            if game.get('title'):
                self.game_index[game['title'].lower().strip()] = i
    
    def load_games_chunked(self, limit=None):
        """Load games from MongoDB"""
        try:
            if limit:
                self.games = list(self.db.steam_games.find({}).limit(limit))
            else:
                self.games = list(self.db.steam_games.find({}))
            
            # Ensure all required fields exist
            for game in self.games:
                # Ensure float fields
                for field in ['discounted_price', 'original_price', 'discount_percentage', 
                             'overall_sentiment_score', 'popularity_score']:
                    if field in game:
                        game[field] = float(game[field])
                
                # Ensure int fields
                for field in ['all_reviews_count', 'release_year', 'memory_gb', 'storage_gb']:
                    if field in game:
                        try:
                            game[field] = int(game[field])
                        except:
                            game[field] = 0
                
                # Ensure list fields
                for field in ['tags', 'languages', 'features', 'categories']:
                    if field not in game or not isinstance(game[field], list):
                        game[field] = []
                
                # Ensure string fields
                for field in ['developer', 'publisher', 'os_type', 'link']:
                    if field not in game:
                        game[field] = ''
            
            print(f"📊 Loaded {len(self.games)} games")
            return self.games
        except Exception as e:
            print(f"❌ Error loading games: {e}")
            self.games = []
            return []
    
    def prepare_features_sparse(self):
        """Create comprehensive sparse TF-IDF features"""
        if not self.games:
            self.load_games_chunked()
        
        print("🔄 Creating comprehensive features...")
        
        feature_strings = []
        for game in self.games:
            features = []
            
            # 1. ALL TAGS (but limit to 15 to avoid memory issues)
            if game.get('tags'):
                # Take all unique tags, limit to 15 most relevant
                unique_tags = list(set([str(tag).lower().strip() for tag in game['tags']]))
                features.extend([f"tag_{tag}" for tag in unique_tags[:15]])
            
            # 2. Developer and Publisher
            if game.get('developer'):
                dev_clean = str(game['developer']).lower().strip()
                if dev_clean:
                    features.append(f"dev_{dev_clean}")
            
            if game.get('publisher'):
                pub_clean = str(game['publisher']).lower().strip()
                if pub_clean:
                    features.append(f"pub_{pub_clean}")
            
            # 3. Release year as feature
            if game.get('release_year'):
                features.append(f"year_{game['release_year']}")
            
            # 4. Game features (single-player, multiplayer, etc.)
            if game.get('features'):
                features.extend([f"feature_{str(feat).lower().strip()}" for feat in game['features'][:5]])
            
            # 5. Price category
            price = game.get('discounted_price', 0)
            if price == 0:
                features.append("price_free")
            elif price < 10:
                features.append("price_budget")
            elif price < 30:
                features.append("price_mid")
            else:
                features.append("price_premium")
            
            # 6. Sentiment category
            sentiment = game.get('overall_sentiment_score', 0.5)
            if sentiment >= 0.8:
                features.append("sentiment_very_positive")
            elif sentiment >= 0.6:
                features.append("sentiment_positive")
            elif sentiment >= 0.4:
                features.append("sentiment_mixed")
            else:
                features.append("sentiment_negative")
            
            # 7. Popularity category
            reviews = game.get('all_reviews_count', 0)
            if reviews > 10000:
                features.append("popularity_very_high")
            elif reviews > 1000:
                features.append("popularity_high")
            elif reviews > 100:
                features.append("popularity_medium")
            else:
                features.append("popularity_low")
            
            # 8. Game categories
            if game.get('categories'):
                features.extend([f"cat_{str(cat).lower().strip()}" for cat in game['categories'][:5]])
            
            # 9. Memory requirements
            if game.get('memory_gb'):
                mem = game['memory_gb']
                if mem <= 4:
                    features.append("memory_low")
                elif mem <= 8:
                    features.append("memory_medium")
                elif mem <= 16:
                    features.append("memory_high")
                else:
                    features.append("memory_very_high")
            
            # 10. Storage requirements
            if game.get('storage_gb'):
                storage = game['storage_gb']
                if storage <= 10:
                    features.append("storage_small")
                elif storage <= 50:
                    features.append("storage_medium")
                else:
                    features.append("storage_large")
            
            # Combine all features
            feature_strings.append(' '.join(features))
        
        # Create sparse matrix
        self.game_features = self.tfidf_vectorizer.fit_transform(feature_strings)
        print(f"✅ Comprehensive features created: {self.game_features.shape}")
        print(f"   Memory: {self.game_features.data.nbytes / 1024 / 1024:.1f} MB")
        print(f"   Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return self.game_features
    
    def _normalize_similarity(self, similarity_score, method='cosine'):
        """Normalize similarity score to 0-1 range"""
        if method == 'pearson':
            # Pearson ranges from -1 to 1, normalize to 0-1
            return max(0.0, (similarity_score + 1) / 2)
        elif method == 'euclidean':
            # Euclidean similarity is already in 0-1 range (Gaussian kernel)
            return min(1.0, max(0.0, similarity_score))
        elif method == 'jaccard':
            # Jaccard is already in 0-1 range
            return min(1.0, max(0.0, similarity_score))
        else:  # cosine
            # Cosine similarity ranges from -1 to 1, but TF-IDF vectors are non-negative
            # so cosine similarity should be in 0-1 range
            return min(1.0, max(0.0, similarity_score))
    
    def calculate_cosine_topk_chunked(self):
        """Calculate top-K cosine similarities in chunks"""
        if self.game_features is None:
            self.prepare_features_sparse()
        
        print(f"🔢 Calculating top-{self.TOP_K} cosine similarities...")
        start = time.time()
        
        n_games = len(self.games)
        cosine_sims = defaultdict(list)
        
        # Process in chunks
        for i in range(0, n_games, self.CHUNK_SIZE):
            chunk_end = min(i + self.CHUNK_SIZE, n_games)
            print(f"   Processing chunk {i}-{chunk_end-1}...")
            
            chunk_features = self.game_features[i:chunk_end]
            chunk_similarities = cosine_similarity(chunk_features, self.game_features)
            
            for idx_in_chunk in range(chunk_end - i):
                game_idx = i + idx_in_chunk
                similarities = chunk_similarities[idx_in_chunk]
                
                # Normalize scores to ensure 0-1 range
                similarities = np.clip(similarities, 0.0, 1.0)
                
                # Get top-K indices (excluding self)
                top_indices = np.argsort(similarities)[-self.TOP_K-1:-1][::-1]
                top_scores = similarities[top_indices]
                
                # Filter low similarities
                valid_mask = top_scores > 0.1
                if np.any(valid_mask):
                    cosine_sims[game_idx] = list(zip(
                        top_scores[valid_mask].tolist(),
                        top_indices[valid_mask].tolist()
                    ))
        
        self.top_k_similarities['cosine'] = cosine_sims
        self._save_topk_similarities('cosine', cosine_sims)
        
        elapsed = time.time() - start
        print(f"✅ Top-{self.TOP_K} cosine calculated in {elapsed:.1f}s")
        return cosine_sims
    
    def calculate_pearson_topk_chunked(self):
        """Calculate TRUE Pearson correlation (mathematically correct)"""
        if self.game_features is None:
            self.prepare_features_sparse()
        
        print(f"📊 Calculating TRUE top-{self.TOP_K} Pearson correlations...")
        start = time.time()
        
        n_games = len(self.games)
        pearson_sims = defaultdict(list)
        
        # Convert to dense array for Pearson (needs actual values)
        features_dense = self.game_features.toarray()
        
        # Add small noise to prevent perfect correlations
        noise_scale = 0.0001
        noise = np.random.normal(0, noise_scale, features_dense.shape)
        features_with_noise = features_dense + noise
        
        # Standardize features (z-score normalization) for Pearson
        features_mean = np.mean(features_with_noise, axis=1, keepdims=True)
        features_std = np.std(features_with_noise, axis=1, keepdims=True)
        features_std[features_std == 0] = 1  # Avoid division by zero
        features_standardized = (features_with_noise - features_mean) / features_std
        
        for i in range(0, n_games, self.CHUNK_SIZE):
            chunk_end = min(i + self.CHUNK_SIZE, n_games)
            print(f"   Processing chunk {i}-{chunk_end-1}...")
            
            chunk_features = features_standardized[i:chunk_end]
            
            # Compute Pearson correlation matrix
            # Pearson = (1/n) * Σ[(x_i - μ_x)/σ_x * (y_i - μ_y)/σ_y]
            # Since data is standardized, Pearson = (1/n) * X·Y^T
            n_features = features_standardized.shape[1]
            pearson_matrix = np.dot(chunk_features, features_standardized.T) / n_features
            
            for idx_in_chunk in range(chunk_end - i):
                game_idx = i + idx_in_chunk
                correlations = pearson_matrix[idx_in_chunk]
                
                # Normalize Pearson to 0-1 range and clip
                similarities = np.clip((correlations + 1) / 2, 0.0, 1.0)
                
                # Get top-K indices (excluding self)
                top_indices = np.argsort(similarities)[-self.TOP_K-1:-1][::-1]
                top_scores = similarities[top_indices]
                
                # Filter low similarities
                valid_mask = top_scores > 0.1
                if np.any(valid_mask):
                    pearson_sims[game_idx] = list(zip(
                        top_scores[valid_mask].tolist(),
                        top_indices[valid_mask].tolist()
                    ))
        
        self.top_k_similarities['pearson'] = pearson_sims
        self._save_topk_similarities('pearson', pearson_sims)
        
        elapsed = time.time() - start
        print(f"✅ TRUE Top-{self.TOP_K} Pearson calculated in {elapsed:.1f}s")
        return pearson_sims
    
    def calculate_euclidean_topk_chunked(self):
        """Calculate TRUE Euclidean similarity (inverse of distance)"""
        if self.game_features is None:
            self.prepare_features_sparse()
        
        print(f"📏 Calculating TRUE top-{self.TOP_K} Euclidean similarities...")
        start = time.time()
        
        n_games = len(self.games)
        euclidean_sims = defaultdict(list)
        
        # Convert to dense
        features_dense = self.game_features.toarray()
        
        # Normalize features to unit length for Euclidean distance
        features_norm = np.linalg.norm(features_dense, axis=1, keepdims=True)
        features_norm[features_norm == 0] = 1
        features_normalized = features_dense / features_norm
        
        for i in range(0, n_games, self.CHUNK_SIZE):
            chunk_end = min(i + self.CHUNK_SIZE, n_games)
            print(f"   Processing chunk {i}-{chunk_end-1}...")
            
            chunk_features = features_normalized[i:chunk_end]
            
            # Compute Euclidean distances
            distances = cdist(chunk_features, features_normalized, 'euclidean')
            
            # Convert distances to similarities using Gaussian kernel
            # similarity = exp(-γ * distance^2)
            gamma = 2.0  # Adjusted for normalized features
            similarities = np.exp(-gamma * (distances ** 2))
            
            # Ensure similarities are in 0-1 range
            similarities = np.clip(similarities, 0.0, 1.0)
            
            for idx_in_chunk in range(chunk_end - i):
                game_idx = i + idx_in_chunk
                sims = similarities[idx_in_chunk]
                
                # Get top-K indices (excluding self)
                top_indices = np.argsort(sims)[-self.TOP_K-1:-1][::-1]
                top_scores = sims[top_indices]
                
                # Filter low similarities
                valid_mask = top_scores > 0.1
                if np.any(valid_mask):
                    euclidean_sims[game_idx] = list(zip(
                        top_scores[valid_mask].tolist(),
                        top_indices[valid_mask].tolist()
                    ))
        
        self.top_k_similarities['euclidean'] = euclidean_sims
        self._save_topk_similarities('euclidean', euclidean_sims)
        
        elapsed = time.time() - start
        print(f"✅ TRUE Top-{self.TOP_K} Euclidean calculated in {elapsed:.1f}s")
        return euclidean_sims
    
    def calculate_jaccard_topk_chunked(self):
        """Calculate Jaccard similarity (set-based, completely different)"""
        if self.game_features is None:
            self.prepare_features_sparse()
        
        print(f"🎭 Calculating top-{self.TOP_K} Jaccard similarities...")
        start = time.time()
        
        n_games = len(self.games)
        jaccard_sims = defaultdict(list)
        
        # Convert to binary features for Jaccard (presence/absence)
        features_dense = self.game_features.toarray()
        features_binary = (features_dense > 0).astype(int)
        
        # Process in smaller chunks due to memory constraints
        small_chunk = 200
        for i in range(0, n_games, small_chunk):
            chunk_end = min(i + small_chunk, n_games)
            print(f"   Processing chunk {i}-{chunk_end-1}...")
            
            chunk_features = features_binary[i:chunk_end]
            
            # Compute Jaccard similarity: |A ∩ B| / |A ∪ B|
            dot_products = np.dot(chunk_features, features_binary.T)
            chunk_sums = np.sum(chunk_features, axis=1, keepdims=True)
            all_sums = np.sum(features_binary, axis=1)
            
            # Avoid division by zero
            denominator = chunk_sums + all_sums - dot_products
            denominator[denominator == 0] = 1
            
            jaccard_matrix = dot_products / denominator
            
            # Ensure Jaccard is in 0-1 range
            jaccard_matrix = np.clip(jaccard_matrix, 0.0, 1.0)
            
            for idx_in_chunk in range(chunk_end - i):
                game_idx = i + idx_in_chunk
                similarities = jaccard_matrix[idx_in_chunk]
                
                # Get top-K indices (excluding self)
                top_indices = np.argsort(similarities)[-self.TOP_K-1:-1][::-1]
                top_scores = similarities[top_indices]
                
                # Filter low similarities
                valid_mask = top_scores > 0.05  # Lower threshold for Jaccard
                if np.any(valid_mask):
                    jaccard_sims[game_idx] = list(zip(
                        top_scores[valid_mask].tolist(),
                        top_indices[valid_mask].tolist()
                    ))
        
        self.top_k_similarities['jaccard'] = jaccard_sims
        self._save_topk_similarities('jaccard', jaccard_sims)
        
        elapsed = time.time() - start
        print(f"✅ Top-{self.TOP_K} Jaccard calculated in {elapsed:.1f}s")
        return jaccard_sims
    
    def _save_topk_similarities(self, method: str, similarities: dict):
        """Save top-K similarities efficiently"""
        file_path = f"{self.model_dir}/topk_{method}.pkl"
        
        compact_data = {}
        for game_idx, sim_list in similarities.items():
            if sim_list:
                scores, indices = zip(*sim_list)
                # Ensure scores are in 0-1 range
                scores = np.clip(scores, 0.0, 1.0)
                compact_data[game_idx] = {
                    'scores': np.array(scores, dtype=np.float16),
                    'indices': np.array(indices, dtype=np.uint16)
                }
        
        with open(file_path, 'wb') as f:
            pickle.dump(compact_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"💾 Saved {method} top-K: {len(compact_data)} games")
    
    def _load_topk_similarities(self, method: str):
        """Load top-K similarities"""
        file_path = f"{self.model_dir}/topk_{method}.pkl"
        
        if not os.path.exists(file_path):
            return {}
        
        with open(file_path, 'rb') as f:
            compact_data = pickle.load(f)
        
        similarities = {}
        for game_idx, data in compact_data.items():
            scores = data['scores'].tolist()
            indices = data['indices'].tolist()
            similarities[game_idx] = list(zip(scores, indices))
        
        return similarities
    
    def train_models(self):
        """Train all models efficiently"""
        print("🚀 Training all models (Cosine, Pearson, Euclidean, Jaccard)...")
        start = time.time()
        
        # Step 1: Load games
        self.load_games_chunked()
        
        # Step 2: Prepare sparse features
        self.prepare_features_sparse()
        
        # Step 3: Calculate all similarity matrices
        print("\n📊 Calculating 4 DIFFERENT similarity matrices...")
        
        self.calculate_cosine_topk_chunked()
        self.calculate_pearson_topk_chunked()
        self.calculate_euclidean_topk_chunked()
        self.calculate_jaccard_topk_chunked()
        
        # Step 4: Save base data
        self._save_base_data()
        
        # Step 5: Verify scores are in correct range
        self._verify_score_ranges()
        
        elapsed = time.time() - start
        print(f"\n✅ All models trained in {elapsed:.1f} seconds!")
        print(f"📊 Stats: {len(self.games)} games, {self.game_features.shape[1]} features")
        print("✅ Methods available: cosine, pearson, euclidean, jaccard")
        
        return True
    
    def _verify_score_ranges(self):
        """Verify that all similarity scores are in 0-1 range"""
        print("\n🔍 Verifying score ranges...")
        
        for method in ['cosine', 'pearson', 'euclidean', 'jaccard']:
            if method in self.top_k_similarities:
                all_scores = []
                for game_idx, sim_list in self.top_k_similarities[method].items():
                    scores = [score for score, _ in sim_list]
                    all_scores.extend(scores)
                
                if all_scores:
                    min_score = min(all_scores)
                    max_score = max(all_scores)
                    print(f"   {method}: {min_score:.3f} - {max_score:.3f}")
                    
                    if max_score > 1.0 or min_score < 0.0:
                        print(f"⚠️  {method} scores out of 0-1 range!")
    
    def _save_base_data(self):
        """Save essential game data"""
        try:
            game_index = {}
            game_titles = []
            
            for i, game in enumerate(self.games):
                title = game.get('title', '').strip()
                if title:
                    game_index[title.lower()] = i
                    game_titles.append(title)
            
            essential_games = []
            for game in self.games:
                essential_games.append({
                    'title': game.get('title'),
                    'developer': game.get('developer'),
                    'publisher': game.get('publisher'),
                    'discounted_price': float(game.get('discounted_price', 0)),
                    'original_price': float(game.get('original_price', game.get('discounted_price', 0))),
                    'discount_percentage': float(game.get('discount_percentage', 0)),
                    'overall_sentiment_score': float(game.get('overall_sentiment_score', 0.5)),
                    'all_reviews_count': game.get('all_reviews_count', 0),
                    'popularity_score': float(game.get('popularity_score', 0)),
                    'tags': game.get('tags', [])[:15],
                    'languages': game.get('languages', []),
                    'features': game.get('features', []),
                    'categories': game.get('categories', []),
                    'memory_gb': game.get('memory_gb'),
                    'storage_gb': game.get('storage_gb'),
                    'os_type': game.get('os_type', ''),
                    'ssd_required': game.get('ssd_required', False),
                    'link': game.get('link', '#'),
                    'release_year': game.get('release_year')
                })
            
            base_data = {
                'games': essential_games,
                'game_titles': game_titles,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'game_index': game_index,
                'trained_at': datetime.now().isoformat(),
                'methods_available': ['cosine', 'pearson', 'euclidean', 'jaccard']
            }
            
            with open(f"{self.model_dir}/base_data.pkl", 'wb') as f:
                pickle.dump(base_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"✅ Base data saved: {len(essential_games)} games")
            return True
        except Exception as e:
            print(f"❌ Error saving base data: {e}")
            return False
    
    def load_models(self):
        """Load pre-trained models"""
        print("📂 Loading models...")
        
        base_path = f"{self.model_dir}/base_data.pkl"
        if not os.path.exists(base_path):
            print("⚠️ No trained models found. Training...")
            return self.train_models()
        
        try:
            with open(base_path, 'rb') as f:
                base_data = pickle.load(f)
            
            self.games = base_data['games']
            self.tfidf_vectorizer = base_data['tfidf_vectorizer']
            self.game_index = base_data['game_index']
            
            # Load similarity matrices
            for method in ['cosine', 'pearson', 'euclidean', 'jaccard']:
                loaded = self._load_topk_similarities(method)
                if method not in self.top_k_similarities:
                    self.top_k_similarities[method] = {}
                self.top_k_similarities[method].update(loaded)
            
            print(f"✅ Models loaded: {len(self.games)} games")
            print(f"   Trained: {base_data.get('trained_at', 'Unknown')}")
            print(f"   Methods: {base_data.get('methods_available', [])}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def _get_similarities(self, method: str, game_idx: int) -> List[Tuple[float, int]]:
        """Get similarities for a game"""
        if method not in ['cosine', 'pearson', 'euclidean', 'jaccard']:
            return []
            
        return self.top_k_similarities.get(method, {}).get(game_idx, [])
    
    def get_game_index(self, title: str) -> int:
        """Get game index by title"""
        if not hasattr(self, 'game_index') or not self.game_index:
            self._build_game_index()
        
        title_lower = title.lower().strip()
        return self.game_index.get(title_lower, -1)
    
    def get_game_by_title(self, title: str):
        """Get game by title"""
        idx = self.get_game_index(title)
        if idx != -1:
            return self.games[idx]
        return None
    
    def constraint_based_recommendations(self, user_preferences: Dict[str, Any], top_n: int = 10):
        """Constraint-based filtering"""
        if not self.games:
            return {'error': 'No games available', 'recommendations': []}
        
        print("🔍 Constraint-based filtering...")
        
        # Extract preferences with defaults
        max_price = user_preferences.get('max_price', 1000.0)
        min_price = user_preferences.get('min_price', 0.0)
        required_tags = set(str(t).lower() for t in user_preferences.get('preferred_tags', []))
        min_sentiment = user_preferences.get('min_sentiment', 0.0)
        min_reviews = user_preferences.get('min_reviews', 0)
        
        # System specs
        system_specs = user_preferences.get('system_specs', {})
        max_memory = system_specs.get('memory_gb')
        min_storage = system_specs.get('storage_gb')
        os_type = system_specs.get('os_type', '').lower()
        require_ssd = system_specs.get('require_ssd', False)
        
        # Other preferences
        languages = [str(l).lower() for l in user_preferences.get('languages', [])]
        developers = [str(d).lower() for d in user_preferences.get('developers', [])]
        publishers = [str(p).lower() for p in user_preferences.get('publishers', [])]
        
        filtered_games = []
        
        for i, game in enumerate(self.games):
            game_price = game.get('discounted_price', 0)
            
            # Hard constraints
            # Price constraint
            if game_price < min_price or game_price > max_price:
                continue
            
            # Sentiment constraint
            if game.get('overall_sentiment_score', 0) < min_sentiment:
                continue
            
            # Reviews constraint
            if game.get('all_reviews_count', 0) < min_reviews:
                continue
            
            # Language constraint
            if languages:
                game_langs = [str(l).lower() for l in game.get('languages', [])]
                if not any(lang in game_langs for lang in languages):
                    continue
            
            # System requirements constraints
            if max_memory and game.get('memory_gb'):
                if game['memory_gb'] > max_memory:
                    continue
            
            if min_storage and game.get('storage_gb'):
                if game['storage_gb'] < min_storage:
                    continue
            
            if os_type and game.get('os_type'):
                game_os = str(game['os_type']).lower()
                if os_type not in game_os and game_os != 'multi':
                    continue
            
            if require_ssd and not game.get('ssd_required', False):
                continue
            
            filtered_games.append((i, game))
        
        # Score games based on soft constraints
        scored_games = []
        
        for i, game in filtered_games:
            score = 0
            explanations = []
            
            # Tag matching (40 points)
            if required_tags:
                game_tags = set(str(t).lower() for t in game.get('tags', []))
                matched = required_tags & game_tags
                if matched:
                    tag_score = min(40, (len(matched) / max(1, len(required_tags))) * 40)
                    score += tag_score
                    explanations.append(f"Matches {len(matched)} tags")
            
            # Developer matching (20 points)
            game_dev = str(game.get('developer', '')).lower()
            if developers:
                for dev in developers:
                    if dev in game_dev:
                        score += 20
                        explanations.append("Preferred developer")
                        break
            
            # Publisher matching (15 points)
            game_pub = str(game.get('publisher', '')).lower()
            if publishers:
                for pub in publishers:
                    if pub in game_pub:
                        score += 15
                        explanations.append("Preferred publisher")
                        break
            
            # Sentiment score (10 points)
            sentiment = game.get('overall_sentiment_score', 0)
            score += min(10, sentiment * 10)
            
            # Popularity/Reviews (5 points)
            reviews = game.get('all_reviews_count', 0)
            if reviews > 1000:
                score += 5
                explanations.append("Popular")
            
            # Value scoring (10 points)
            price = game.get('discounted_price', 0)
            if price == 0:
                score += 10
                explanations.append("Free")
            else:
                discount = game.get('discount_percentage', 0)
                if discount > 50:
                    score += 9
                    explanations.append(f"{discount:.0f}% off")
                elif price <= 10:
                    score += 8
                    explanations.append("Budget")
                elif price <= 20:
                    score += 6
                    explanations.append("Affordable")
                else:
                    score += 3
            
            # Ensure score is between 0-100
            score = min(100, max(0, score))
            
            scored_games.append({
                'title': game['title'],
                'developer': game.get('developer', ''),
                'publisher': game.get('publisher', ''),
                'price': float(price),
                'original_price': float(game.get('original_price', price)),
                'discount': float(game.get('discount_percentage', 0)),
                'sentiment': float(sentiment),
                'reviews': game.get('all_reviews_count', 0),
                'tags': game.get('tags', [])[:15],
                'languages': game.get('languages', []),
                'link': game.get('link', '#'),
                'release_year': game.get('release_year'),
                'memory_gb': game.get('memory_gb'),
                'storage_gb': game.get('storage_gb'),
                'os_type': game.get('os_type', ''),
                'ssd_required': game.get('ssd_required', False),
                'score': round(score, 1),
                'explanations': explanations[:3]
            })
        
        # Sort and categorize
        scored_games.sort(key=lambda x: -x['score'])
        
        perfect = [g for g in scored_games if g['score'] >= 70][:top_n]
        good = [g for g in scored_games if 50 <= g['score'] < 70][:top_n]
        partial = [g for g in scored_games if 30 <= g['score'] < 50][:top_n]
        
        print(f"✅ Found {len(perfect)} perfect, {len(good)} good, {len(partial)} partial matches")
        
        return {
            'perfect_matches': {'games': perfect, 'count': len(perfect)},
            'good_matches': {'games': good, 'count': len(good)},
            'partial_matches': {'games': partial, 'count': len(partial)},
            'total_evaluated': len(self.games)
        }
    
    def content_based_recommendations(self, cases: List[str], method: str = 'cosine', top_n: int = 10):
        """Content-based similarity recommendations"""
        if not cases:
            return {'recommendations': [], 'error': 'No cases provided'}
        
        print(f"🎯 Content-based ({method}) with {len(cases)} liked games...")
        
        # Get indices for input cases
        case_indices = []
        valid_cases = []
        
        for case in cases:
            idx = self.get_game_index(case)
            if idx != -1:
                case_indices.append(idx)
                valid_cases.append(case)
        
        if not case_indices:
            return {'recommendations': [], 'error': 'No valid cases found'}
        
        print(f"✅ Found {len(case_indices)} valid liked games")
        
        # Analyze liked games to understand user preferences
        liked_games_analysis = {
            'tags': set(),
            'developers': set(),
            'publishers': set(),
            'price_range': [],
            'sentiment_range': [],
            'categories': set()
        }
        
        for idx in case_indices:
            game = self.games[idx]
            
            # Collect tags from liked games
            if game.get('tags'):
                liked_games_analysis['tags'].update([str(t).lower() for t in game['tags']])
            
            # Collect developers
            if game.get('developer'):
                liked_games_analysis['developers'].add(str(game['developer']).lower())
            
            # Collect publishers
            if game.get('publisher'):
                liked_games_analysis['publishers'].add(str(game['publisher']).lower())
            
            # Collect categories
            if game.get('categories'):
                liked_games_analysis['categories'].update([str(c).lower() for c in game['categories']])
            
            # Collect price and sentiment
            if 'discounted_price' in game:
                liked_games_analysis['price_range'].append(game['discounted_price'])
            if 'overall_sentiment_score' in game:
                liked_games_analysis['sentiment_range'].append(game['overall_sentiment_score'])
        
        print(f"📊 Liked games analysis:")
        print(f"   - {len(liked_games_analysis['tags'])} unique tags")
        print(f"   - {len(liked_games_analysis['developers'])} developers")
        print(f"   - {len(liked_games_analysis['categories'])} categories")
        
        # Aggregate similarities with preference weighting
        game_scores = defaultdict(float)
        game_sources = defaultdict(list)
        game_tag_matches = defaultdict(int)
        game_category_matches = defaultdict(int)
        
        for case_idx, case_title in zip(case_indices, valid_cases):
            similarities = self._get_similarities(method, case_idx)
            
            for raw_score, other_idx in similarities:
                if other_idx != case_idx:
                    other_game = self.games[other_idx]
                    game_title = other_game['title']
                    
                    # Ensure raw_score is in 0-1 range
                    raw_score = max(0.0, min(1.0, raw_score))
                    
                    # Calculate enhanced score based on preferences
                    enhanced_score = raw_score
                    max_enhancement = 0.3  # Maximum 30% enhancement
                    current_enhancement = 0.0
                    
                    # Bonus for matching tags from liked games
                    if liked_games_analysis['tags'] and other_game.get('tags'):
                        other_tags = set([str(t).lower() for t in other_game['tags']])
                        tag_overlap = len(liked_games_analysis['tags'] & other_tags)
                        if tag_overlap > 0:
                            # Add bonus based on tag overlap
                            tag_bonus = min(max_enhancement - current_enhancement, tag_overlap * 0.05)
                            enhanced_score = min(1.0, enhanced_score + tag_bonus)
                            current_enhancement += tag_bonus
                            game_tag_matches[game_title] = tag_overlap
                    
                    # Bonus for matching categories
                    if liked_games_analysis['categories'] and other_game.get('categories'):
                        other_cats = set([str(c).lower() for c in other_game['categories']])
                        cat_overlap = len(liked_games_analysis['categories'] & other_cats)
                        if cat_overlap > 0:
                            cat_bonus = min(max_enhancement - current_enhancement, cat_overlap * 0.1)
                            enhanced_score = min(1.0, enhanced_score + cat_bonus)
                            current_enhancement += cat_bonus
                            game_category_matches[game_title] = cat_overlap
                    
                    # Bonus for matching developer
                    if liked_games_analysis['developers'] and other_game.get('developer'):
                        if str(other_game['developer']).lower() in liked_games_analysis['developers']:
                            dev_bonus = min(max_enhancement - current_enhancement, 0.1)
                            enhanced_score = min(1.0, enhanced_score + dev_bonus)
                            current_enhancement += dev_bonus
                    
                    # Bonus for matching publisher
                    if liked_games_analysis['publishers'] and other_game.get('publisher'):
                        if str(other_game['publisher']).lower() in liked_games_analysis['publishers']:
                            pub_bonus = min(max_enhancement - current_enhancement, 0.08)
                            enhanced_score = min(1.0, enhanced_score + pub_bonus)
                            current_enhancement += pub_bonus
                    
                    # Similar price range bonus
                    if liked_games_analysis['price_range'] and 'discounted_price' in other_game:
                        avg_liked_price = sum(liked_games_analysis['price_range']) / len(liked_games_analysis['price_range'])
                        price_diff = abs(other_game['discounted_price'] - avg_liked_price)
                        if price_diff < 10:  # Within $10 of average liked price
                            price_bonus = min(max_enhancement - current_enhancement, 0.15 * (1 - price_diff/10))
                            enhanced_score = min(1.0, enhanced_score + price_bonus)
                            current_enhancement += price_bonus
                    
                    # Similar sentiment bonus
                    if liked_games_analysis['sentiment_range'] and 'overall_sentiment_score' in other_game:
                        avg_liked_sentiment = sum(liked_games_analysis['sentiment_range']) / len(liked_games_analysis['sentiment_range'])
                        sentiment_diff = abs(other_game['overall_sentiment_score'] - avg_liked_sentiment)
                        if sentiment_diff < 0.3:  # Within 0.3 sentiment score
                            sentiment_bonus = min(max_enhancement - current_enhancement, 0.1 * (1 - sentiment_diff/0.3))
                            enhanced_score = min(1.0, enhanced_score + sentiment_bonus)
                            current_enhancement += sentiment_bonus
                    
                    # Ensure enhanced_score is in 0-1 range
                    enhanced_score = max(0.0, min(1.0, enhanced_score))
                    
                    # Only update if we have a higher enhanced score
                    if enhanced_score > game_scores.get(game_title, 0):
                        game_scores[game_title] = enhanced_score
                        game_sources[game_title] = [case_title]
                    elif enhanced_score == game_scores.get(game_title, 0):
                        if case_title not in game_sources[game_title]:
                            game_sources[game_title].append(case_title)
        
        # Prepare results with explanations
        results = []
        for title, score in sorted(game_scores.items(), key=lambda x: -x[1]):
            if title in cases:
                continue
                
            game_idx = self.get_game_index(title)
            if game_idx != -1:
                game = self.games[game_idx]
                
                # Convert similarity to percentage (0-100%)
                similarity_percentage = min(100.0, max(0.0, score * 100))
                
                # Generate explanation
                explanations = []
                tag_matches = game_tag_matches.get(title, 0)
                if tag_matches > 0:
                    explanations.append(f"Shares {tag_matches} tags with your liked games")
                
                cat_matches = game_category_matches.get(title, 0)
                if cat_matches > 0:
                    explanations.append(f"Matches {cat_matches} categories")
                
                if game_sources[title]:
                    sources = game_sources[title][:2]
                    if len(sources) == 1:
                        explanations.append(f"Similar to '{sources[0]}'")
                    else:
                        explanations.append(f"Similar to {len(sources)} liked games")
                
                results.append({
                    'title': title,
                    'developer': game.get('developer', ''),
                    'publisher': game.get('publisher', ''),
                    'price': float(game.get('discounted_price', 0)),
                    'original_price': float(game.get('original_price', game.get('discounted_price', 0))),
                    'discount': float(game.get('discount_percentage', 0)),
                    'similarity': round(similarity_percentage, 1),
                    'sentiment': float(game.get('overall_sentiment_score', 0.5)),
                    'reviews': game.get('all_reviews_count', 0),
                    'tags': game.get('tags', [])[:10],
                    'categories': game.get('categories', [])[:5],
                    'languages': game.get('languages', []),
                    'link': game.get('link', '#'),
                    'release_year': game.get('release_year'),
                    'source_cases': game_sources[title][:3],
                    'explanations': explanations[:2],
                    'method': method,
                    'enhanced': True
                })
        
        # Sort and categorize
        results.sort(key=lambda x: -x['similarity'])
        
        high = [g for g in results if g['similarity'] >= 70][:top_n]
        medium = [g for g in results if 40 <= g['similarity'] < 70][:top_n]
        low = [g for g in results if 20 <= g['similarity'] < 40][:top_n]
        
        print(f"✅ Found {len(high)} high, {len(medium)} medium, {len(low)} low similarity matches")
        
        return {
            'highly_similar': {'games': high, 'count': len(high)},
            'moderately_similar': {'games': medium, 'count': len(medium)},
            'somewhat_similar': {'games': low, 'count': len(low)},
            'method': method,
            'total_found': len(results),
            'liked_games_analysis': {
                'unique_tags_count': len(liked_games_analysis['tags']),
                'developers_count': len(liked_games_analysis['developers']),
                'categories_count': len(liked_games_analysis['categories'])
            }
        }
    
    def hybrid_recommendations(self, user_preferences: Dict, cases: List[str], 
                              method: str = 'cosine', top_n: int = 10):
        """PROPER Hybrid recommendations - Intelligently combines both approaches"""
        print(f"🤝 Hybrid recommendations ({method})...")
        
        if not cases:
            print("⚠️ No liked games provided, using constraint-based only")
            return self.constraint_based_recommendations(user_preferences, top_n)
        
        # Get both recommendations
        constraint_results = self.constraint_based_recommendations(user_preferences, top_n * 3)
        content_results = self.content_based_recommendations(cases, method, top_n * 3)
        
        # Extract all games from both methods
        constraint_games = {}
        for category in ['perfect_matches', 'good_matches', 'partial_matches']:
            if category in constraint_results:
                for game in constraint_results[category].get('games', []):
                    title = game['title']
                    if title not in constraint_games:
                        constraint_games[title] = {
                            'game': game,
                            'category': category,
                            'constraint_score': game.get('score', 0)
                        }
        
        content_games = {}
        for category in ['highly_similar', 'moderately_similar', 'somewhat_similar']:
            if category in content_results:
                for game in content_results[category].get('games', []):
                    title = game['title']
                    if title not in content_games:
                        content_games[title] = {
                            'game': game,
                            'category': category,
                            'content_score': game.get('similarity', 0)
                        }
        
        # Combine with intelligent weighting
        all_games = {}
        
        # 1. Games that appear in BOTH constraint and content results
        for title in set(constraint_games.keys()) & set(content_games.keys()):
            constraint_data = constraint_games[title]
            content_data = content_games[title]
            
            constraint_score = constraint_data['constraint_score']
            content_score = content_data['content_score']
            
            # Normalize both scores to 0-1 range
            constraint_norm = constraint_score / 100.0
            content_norm = content_score / 100.0
            
            # Dynamic weighting based on category quality
            if constraint_data['category'] == 'perfect_matches':
                constraint_weight = 0.7
            elif constraint_data['category'] == 'good_matches':
                constraint_weight = 0.6
            else:
                constraint_weight = 0.5
            
            if content_data['category'] == 'highly_similar':
                content_weight = 0.5
            elif content_data['category'] == 'moderately_similar':
                content_weight = 0.4
            else:
                content_weight = 0.3
            
            # Normalize weights
            total_weight = constraint_weight + content_weight
            constraint_weight = constraint_weight / total_weight
            content_weight = content_weight / total_weight
            
            hybrid_score_norm = (constraint_norm * constraint_weight + 
                               content_norm * content_weight)
            
            # Convert back to percentage
            hybrid_score = hybrid_score_norm * 100
            
            all_games[title] = {
                **constraint_data['game'],
                'type': 'both',
                'constraint_score': constraint_score,
                'content_score': content_score,
                'hybrid_score': round(min(100.0, max(0.0, hybrid_score)), 1),
                'reason': f"Perfect match! Fits preferences ({constraint_score:.0f}%) and similar to liked games ({content_score:.0f}%)"
            }
        
        # 2. Games only in constraint results (but highly ranked)
        for title, constraint_data in constraint_games.items():
            if title not in all_games and constraint_data['category'] in ['perfect_matches', 'good_matches']:
                constraint_score = constraint_data['constraint_score']
                hybrid_score = constraint_score * 0.9  # Slight penalty
                
                all_games[title] = {
                    **constraint_data['game'],
                    'type': 'constraint_only',
                    'constraint_score': constraint_score,
                    'content_score': 0,
                    'hybrid_score': round(min(100.0, max(0.0, hybrid_score)), 1),
                    'reason': f"Perfectly matches your preferences ({constraint_score:.0f}%)"
                }
        
        # 3. Games only in content results (but highly similar)
        for title, content_data in content_games.items():
            if title not in all_games and content_data['category'] in ['highly_similar']:
                content_score = content_data['content_score']
                hybrid_score = content_score * 0.8  # Penalty for not matching constraints
                
                all_games[title] = {
                    **content_data['game'],
                    'type': 'content_only',
                    'constraint_score': 0,
                    'content_score': content_score,
                    'hybrid_score': round(min(100.0, max(0.0, hybrid_score)), 1),
                    'reason': f"Very similar to games you like ({content_score:.0f}%)"
                }
        
        # Final list
        hybrid_list = list(all_games.values())
        hybrid_list.sort(key=lambda x: -x['hybrid_score'])
        
        print(f"✅ Hybrid: {len(hybrid_list[:top_n])} recommendations")
        print(f"   Both constraints & content: {len([g for g in hybrid_list if g['type'] == 'both'])}")
        print(f"   Constraints only: {len([g for g in hybrid_list if g['type'] == 'constraint_only'])}")
        print(f"   Content only: {len([g for g in hybrid_list if g['type'] == 'content_only'])}")
        
        return {
            'recommendations': hybrid_list[:top_n],
            'stats': {
                'constraint_only': len([g for g in hybrid_list if g['type'] == 'constraint_only']),
                'content_only': len([g for g in hybrid_list if g['type'] == 'content_only']),
                'both': len([g for g in hybrid_list if g['type'] == 'both'])
            },
            'method_used': method,
            'description': 'Intelligently combines constraint-based filtering with content-based similarity'
        }
    
    def compare_methods(self, game_title: str, top_n: int = 5):
        """Compare ALL similarity methods"""
        game_idx = self.get_game_index(game_title)
        if game_idx == -1:
            return {'error': f'Game "{game_title}" not found'}
        
        results = {}
        game = self.get_game_by_title(game_title)
        
        methods = ['cosine', 'pearson', 'euclidean', 'jaccard']
        
        for method in methods:
            similarities = self._get_similarities(method, game_idx)
            
            top_recs = []
            for score, other_idx in similarities[:top_n]:
                other_game = self.games[other_idx]
                # Ensure similarity is in percentage and bounded 0-100
                similarity_percentage = min(100.0, max(0.0, score * 100))
                top_recs.append({
                    'title': other_game['title'],
                    'similarity': round(similarity_percentage, 1),
                    'tags': other_game.get('tags', [])[:5],
                    'price': other_game.get('discounted_price', 0),
                    'developer': other_game.get('developer', ''),
                    'sentiment': other_game.get('overall_sentiment_score', 0.5)
                })
            
            results[method] = {
                'recommendations': top_recs,
                'top_similarity': top_recs[0]['similarity'] if top_recs else 0
            }
        
        # Check if methods give different results
        method_titles = {}
        for method in methods:
            method_titles[method] = set(g['title'] for g in results[method]['recommendations'])
        
        # Calculate pairwise overlaps
        overlap_matrix = {}
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                overlap = len(method_titles[m1] & method_titles[m2])
                overlap_matrix[f"{m1}_{m2}"] = overlap
        
        # Check if all are different
        all_same = all(len(set(method_titles[m1]) - set(method_titles[m2])) == 0 
                      for m1 in methods for m2 in methods)
        
        return {
            'source_game': {
                'title': game_title,
                'developer': game.get('developer', ''),
                'tags': game.get('tags', [])[:5],
                'price': game.get('discounted_price', 0),
                'sentiment': game.get('overall_sentiment_score', 0.5)
            },
            'comparisons': results,
            'methods_different': not all_same,
            'overlap_analysis': overlap_matrix,
            'method_descriptions': {
                'cosine': 'Angle between feature vectors',
                'pearson': 'Linear correlation between standardized features',
                'euclidean': 'Inverse of straight-line distance (Gaussian kernel)',
                'jaccard': 'Set overlap between binary features'
            }
        }
    
    def get_popular_recommendations(self, top_n: int = 10):
        """Get popular games as fallback"""
        popular_games = sorted(self.games, 
                              key=lambda x: x.get('popularity_score', 0) * (x.get('all_reviews_count', 0) + 1),
                              reverse=True)[:top_n]
        
        return [{
            'title': game['title'],
            'developer': game.get('developer', ''),
            'price': float(game.get('discounted_price', 0)),
            'sentiment': float(game.get('overall_sentiment_score', 0.5)),
            'reviews': game.get('all_reviews_count', 0),
            'tags': game.get('tags', [])[:5],
            'reason': 'Popular game',
            'popularity_score': game.get('popularity_score', 0)
        } for game in popular_games]