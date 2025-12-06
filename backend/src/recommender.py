import numpy as np
import pickle
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime
import time

class GameRecommender:
    """Memory-efficient recommendation system"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.games = []
        self.game_features = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        
        # Store only top-K similarities
        self.top_k_similarities = {
            'cosine': {},
            'pearson': {},
            'euclidean': {}
        }
        
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Settings
        self.TOP_K = 50  # Store only top 50 similarities per game
        self.CHUNK_SIZE = 1000  # Process games in chunks
        
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
                for field in ['tags', 'languages']:
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
        """Create sparse TF-IDF features"""
        if not self.games:
            self.load_games_chunked()
        
        print("🔄 Creating sparse features...")
        
        feature_strings = []
        for game in self.games:
            features = []
            
            # Essential features only
            if game.get('tags'):
                features.extend([str(tag).lower() for tag in game['tags'][:5]])
            
            if game.get('developer'):
                features.append(f"dev_{str(game['developer']).lower()}")
            
            if game.get('publisher'):
                features.append(f"pub_{str(game['publisher']).lower()}")
            
            if game.get('release_year'):
                features.append(f"year_{game['release_year']}")
            
            feature_strings.append(' '.join(features))
        
        # Create sparse matrix
        self.game_features = self.tfidf_vectorizer.fit_transform(feature_strings)
        print(f"✅ Sparse features created: {self.game_features.shape}")
        print(f"   Memory: {self.game_features.data.nbytes / 1024 / 1024:.1f} MB")
        
        return self.game_features
    
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
        """Calculate top-K Pearson correlations in chunks"""
        if self.game_features is None:
            self.prepare_features_sparse()
        
        print(f"📊 Calculating top-{self.TOP_K} Pearson correlations...")
        start = time.time()
        
        n_games = len(self.games)
        pearson_sims = defaultdict(list)
        
        # Convert to dense array for Pearson
        features_dense = self.game_features.toarray()
        
        for i in range(0, n_games, self.CHUNK_SIZE):
            chunk_end = min(i + self.CHUNK_SIZE, n_games)
            print(f"   Processing chunk {i}-{chunk_end-1}...")
            
            chunk_features = features_dense[i:chunk_end]
            
            # Pearson correlation
            chunk_mean = np.mean(chunk_features, axis=1, keepdims=True)
            chunk_centered = chunk_features - chunk_mean
            
            all_mean = np.mean(features_dense, axis=1, keepdims=True)
            all_centered = features_dense - all_mean
            
            chunk_norm = np.linalg.norm(chunk_centered, axis=1, keepdims=True)
            all_norm = np.linalg.norm(all_centered, axis=1, keepdims=True)
            
            chunk_norm[chunk_norm == 0] = 1
            all_norm[all_norm == 0] = 1
            
            correlations = np.dot(chunk_centered / chunk_norm, 
                                 (all_centered / all_norm).T)
            
            for idx_in_chunk in range(chunk_end - i):
                game_idx = i + idx_in_chunk
                corrs = correlations[idx_in_chunk]
                
                top_indices = np.argsort(corrs)[-self.TOP_K-1:-1][::-1]
                top_scores = corrs[top_indices]
                
                # Normalize to 0-1
                top_scores = (top_scores + 1) / 2
                valid_mask = top_scores > 0.1
                
                if np.any(valid_mask):
                    pearson_sims[game_idx] = list(zip(
                        top_scores[valid_mask].tolist(),
                        top_indices[valid_mask].tolist()
                    ))
        
        self.top_k_similarities['pearson'] = pearson_sims
        self._save_topk_similarities('pearson', pearson_sims)
        
        elapsed = time.time() - start
        print(f"✅ Top-{self.TOP_K} Pearson calculated in {elapsed:.1f}s")
        return pearson_sims
    
    def calculate_euclidean_topk_chunked(self):
        """Calculate top-K Euclidean similarities in chunks"""
        if self.game_features is None:
            self.prepare_features_sparse()
        
        print(f"📏 Calculating top-{self.TOP_K} Euclidean similarities...")
        start = time.time()
        
        n_games = len(self.games)
        euclidean_sims = defaultdict(list)
        
        # Convert to dense
        features_dense = self.game_features.toarray()
        
        # Normalize
        norms = np.linalg.norm(features_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1
        features_normalized = features_dense / norms
        
        for i in range(0, n_games, self.CHUNK_SIZE):
            chunk_end = min(i + self.CHUNK_SIZE, n_games)
            print(f"   Processing chunk {i}-{chunk_end-1}...")
            
            chunk_features = features_normalized[i:chunk_end]
            
            # Calculate distances
            distances = cdist(chunk_features, features_normalized, 'euclidean')
            
            # Convert to similarity: 1 / (1 + distance)
            similarities = 1 / (1 + distances)
            
            for idx_in_chunk in range(chunk_end - i):
                game_idx = i + idx_in_chunk
                sims = similarities[idx_in_chunk]
                
                top_indices = np.argsort(sims)[-self.TOP_K-1:-1][::-1]
                top_scores = sims[top_indices]
                
                valid_mask = top_scores > 0.1
                if np.any(valid_mask):
                    euclidean_sims[game_idx] = list(zip(
                        top_scores[valid_mask].tolist(),
                        top_indices[valid_mask].tolist()
                    ))
        
        self.top_k_similarities['euclidean'] = euclidean_sims
        self._save_topk_similarities('euclidean', euclidean_sims)
        
        elapsed = time.time() - start
        print(f"✅ Top-{self.TOP_K} Euclidean calculated in {elapsed:.1f}s")
        return euclidean_sims
    
    def _save_topk_similarities(self, method: str, similarities: dict):
        """Save top-K similarities efficiently"""
        file_path = f"{self.model_dir}/topk_{method}.pkl"
        
        compact_data = {}
        for game_idx, sim_list in similarities.items():
            if sim_list:
                scores, indices = zip(*sim_list)
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
        """Train 3 models efficiently"""
        print("🚀 Training 3 efficient models (Cosine, Pearson, Euclidean)...")
        start = time.time()
        
        # Step 1: Load games
        self.load_games_chunked()
        
        # Step 2: Prepare sparse features
        self.prepare_features_sparse()
        
        # Step 3: Calculate 3 similarity matrices
        print("\n📊 Calculating 3 similarity matrices...")
        
        self.calculate_cosine_topk_chunked()
        self.calculate_pearson_topk_chunked()
        self.calculate_euclidean_topk_chunked()
        
        # Step 4: Save base data
        self._save_base_data()
        
        elapsed = time.time() - start
        print(f"\n✅ 3 models trained in {elapsed:.1f} seconds!")
        print(f"📊 Stats: {len(self.games)} games, {self.game_features.shape[1]} features")
        print("✅ Methods available: cosine, pearson, euclidean")
        
        return True
    
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
                    'tags': game.get('tags', [])[:10],
                    'languages': game.get('languages', []),
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
                'methods_available': ['cosine', 'pearson', 'euclidean']
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
            for method in ['cosine', 'pearson', 'euclidean']:
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
        if method not in ['cosine', 'pearson', 'euclidean']:
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
        """Constraint-based filtering - FIXED VERSION"""
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
                'tags': game.get('tags', [])[:8],
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
        """Content-based recommendations"""
        if not cases:
            return {'recommendations': [], 'error': 'No cases provided'}
        
        print(f"🎯 Content-based ({method})...")
        
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
        
        # Aggregate similarities
        game_scores = defaultdict(float)
        game_sources = defaultdict(list)
        
        for case_idx, case_title in zip(case_indices, valid_cases):
            similarities = self._get_similarities(method, case_idx)
            
            for score, other_idx in similarities:
                if other_idx != case_idx:
                    other_game = self.games[other_idx]
                    game_title = other_game['title']
                    
                    # Only update if we have a higher score
                    if score > game_scores.get(game_title, 0):
                        game_scores[game_title] = score
                        game_sources[game_title] = [case_title]
                    elif score == game_scores.get(game_title, 0):
                        if case_title not in game_sources[game_title]:
                            game_sources[game_title].append(case_title)
        
        # Prepare results
        results = []
        for title, score in game_scores.items():
            if title in cases:
                continue
                
            game_idx = self.get_game_index(title)
            if game_idx != -1:
                game = self.games[game_idx]
                
                explanation = ""
                if game_sources[title]:
                    sources = game_sources[title][:2]
                    if len(sources) == 1:
                        explanation = f"Similar to '{sources[0]}'"
                    else:
                        explanation = f"Similar to {len(sources)} liked games"
                
                results.append({
                    'title': title,
                    'developer': game.get('developer', ''),
                    'publisher': game.get('publisher', ''),
                    'price': float(game.get('discounted_price', 0)),
                    'original_price': float(game.get('original_price', game.get('discounted_price', 0))),
                    'discount': float(game.get('discount_percentage', 0)),
                    'similarity': round(score * 100, 1),
                    'sentiment': float(game.get('overall_sentiment_score', 0.5)),
                    'reviews': game.get('all_reviews_count', 0),
                    'tags': game.get('tags', [])[:5],
                    'languages': game.get('languages', []),
                    'link': game.get('link', '#'),
                    'release_year': game.get('release_year'),
                    'source_cases': game_sources[title][:3],
                    'explanation': explanation,
                    'method': method
                })
        
        # Sort and categorize
        results.sort(key=lambda x: -x['similarity'])
        
        high = [g for g in results if g['similarity'] >= 70][:top_n]
        medium = [g for g in results if 40 <= g['similarity'] < 70][:top_n]
        low = [g for g in results if 20 <= g['similarity'] < 40][:top_n]
        
        print(f"✅ Found {len(high)} high, {len(medium)} medium, {len(low)} low")
        
        return {
            'highly_similar': {'games': high, 'count': len(high)},
            'moderately_similar': {'games': medium, 'count': len(medium)},
            'somewhat_similar': {'games': low, 'count': len(low)},
            'method': method,
            'total_found': len(results)
        }
    
    def hybrid_recommendations(self, user_preferences: Dict, cases: List[str], 
                              method: str = 'cosine', top_n: int = 10):
        """Hybrid recommendations"""
        print(f"🤝 Hybrid recommendations ({method})...")
        
        # Get both recommendations
        constraint_results = self.constraint_based_recommendations(user_preferences, top_n * 2)
        content_results = self.content_based_recommendations(cases, method, top_n * 2)
        
        # Combine
        all_games = {}
        
        # Add constraint matches
        for category in ['perfect_matches', 'good_matches', 'partial_matches']:
            for game in constraint_results.get(category, {}).get('games', []):
                title = game['title']
                if title not in all_games:
                    all_games[title] = {
                        **game,
                        'type': f'constraint_{category.split("_")[0]}',
                        'constraint_score': game.get('score', 0),
                        'content_score': 0,
                        'hybrid_score': game.get('score', 0)
                    }
        
        # Add content matches
        for category in ['highly_similar', 'moderately_similar', 'somewhat_similar']:
            for game in content_results.get(category, {}).get('games', []):
                title = game['title']
                content_score = game.get('similarity', 0)
                
                if title in all_games:
                    all_games[title]['type'] = 'both'
                    all_games[title]['content_score'] = content_score
                    all_games[title]['hybrid_score'] = (
                        all_games[title]['constraint_score'] * 0.6 + 
                        content_score * 0.4
                    )
                else:
                    all_games[title] = {
                        **game,
                        'type': f'content_{category.split("_")[0]}',
                        'constraint_score': 0,
                        'content_score': content_score,
                        'hybrid_score': content_score
                    }
        
        # Final list
        hybrid_list = list(all_games.values())
        hybrid_list.sort(key=lambda x: -x['hybrid_score'])
        
        # Add explanations
        for game in hybrid_list[:top_n]:
            if game['type'] == 'both':
                game['reason'] = f"Perfect match! ({game['constraint_score']:.0f}% constraints + {game['content_score']:.0f}% similarity)"
            elif 'constraint' in game['type']:
                game['reason'] = f"Matches your preferences ({game['constraint_score']:.0f}%)"
            else:
                game['reason'] = f"Similar to games you like ({game['content_score']:.0f}%)"
        
        print(f"✅ Hybrid: {len(hybrid_list[:top_n])} recommendations")
        
        return {
            'recommendations': hybrid_list[:top_n],
            'stats': {
                'constraint_only': len([g for g in hybrid_list if 'constraint' in g['type'] and g['type'] != 'both']),
                'content_only': len([g for g in hybrid_list if 'content' in g['type'] and g['type'] != 'both']),
                'both': len([g for g in hybrid_list if g['type'] == 'both'])
            }
        }
    
    def compare_methods(self, game_title: str, top_n: int = 5):
        """Compare similarity methods"""
        game_idx = self.get_game_index(game_title)
        if game_idx == -1:
            return {'error': f'Game "{game_title}" not found'}
        
        results = {}
        game = self.get_game_by_title(game_title)
        
        for method in ['cosine', 'pearson', 'euclidean']:
            similarities = self._get_similarities(method, game_idx)
            
            top_recs = []
            for score, other_idx in similarities[:top_n]:
                other_game = self.games[other_idx]
                top_recs.append({
                    'title': other_game['title'],
                    'similarity': round(score * 100, 1),
                    'tags': other_game.get('tags', [])[:3],
                    'price': other_game.get('discounted_price', 0),
                    'developer': other_game.get('developer', ''),
                    'sentiment': other_game.get('overall_sentiment_score', 0.5)
                })
            
            results[method] = {
                'recommendations': top_recs,
                'top_similarity': top_recs[0]['similarity'] if top_recs else 0
            }
        
        return {
            'source_game': {
                'title': game_title,
                'developer': game.get('developer', ''),
                'tags': game.get('tags', [])[:5],
                'price': game.get('discounted_price', 0),
                'sentiment': game.get('overall_sentiment_score', 0.5)
            },
            'comparisons': results
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
            'tags': game.get('tags', [])[:3],
            'reason': 'Popular game'
        } for game in popular_games]