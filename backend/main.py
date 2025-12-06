from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
from bson import json_util
from contextlib import asynccontextmanager
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your config and modules
try:
    from config import PORT
    from src.db import db
    from src.recommender import GameRecommender
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have the following structure:")
    print("  - src/db.py with MongoDB connection")
    print("  - src/recommender.py with GameRecommender class")
    print("  - config.py with PORT variable")
    raise

# Initialize recommender
recommender = GameRecommender(db)

# Pydantic Models
class SystemSpecs(BaseModel):
    memory_gb: Optional[int] = Field(None, ge=1, le=128, description="RAM in GB")
    storage_gb: Optional[int] = Field(None, ge=1, le=2000, description="Storage in GB")
    os_type: Optional[str] = Field(None, description="Operating system (windows, linux, mac)")
    require_ssd: Optional[bool] = Field(False, description="SSD required")

class UserPreferences(BaseModel):
    max_price: float = Field(50.0, ge=0, le=1000, description="Maximum price in USD")
    min_price: float = Field(0.0, ge=0, le=1000, description="Minimum price in USD")
    preferred_tags: List[str] = Field(default_factory=list, description="Preferred game tags")
    languages: List[str] = Field(default_factory=list, description="Required languages")
    developers: List[str] = Field(default_factory=list, description="Preferred developers")
    publishers: List[str] = Field(default_factory=list, description="Preferred publishers")
    system_specs: Optional[SystemSpecs] = Field(None, description="System specifications")
    min_sentiment: float = Field(0.0, ge=0, le=1, description="Minimum sentiment score (0-1)")
    min_reviews: int = Field(0, ge=0, description="Minimum number of reviews")

class ContentRequest(BaseModel):
    cases: List[str] = Field(..., description="List of game titles to find similar games")
    method: str = Field("cosine", description="Similarity method: cosine, pearson, euclidean, jaccard")
    limit: int = Field(10, ge=1, le=50, description="Number of recommendations")

class ConstraintRequest(BaseModel):
    preferences: UserPreferences
    limit: int = Field(10, ge=1, le=50, description="Number of recommendations")

class HybridRequest(BaseModel):
    preferences: UserPreferences
    cases: List[str] = Field(default_factory=list, description="Game titles to find similar games")
    method: str = Field("cosine", description="Similarity method")
    limit: int = Field(10, ge=1, le=50, description="Number of recommendations")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    print("🚀 Starting Game Recommendation API...")
    print("📊 Loading recommendation models...")
    success = recommender.load_models()
    if success:
        print("✅ All models loaded successfully!")
    else:
        print("⚠️ Model loading failed")
    yield
    print("👋 Shutting down Recommendation API...")

app = FastAPI(
    title="Game Recommendation API",
    description="Professional game recommendation system with constraint-based, content-based, and hybrid methods",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_mongo_data(data):
    """Convert MongoDB data to JSON serializable format"""
    return json.loads(json_util.dumps(data))

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Game Recommendation API v1.0",
        "description": "Professional RS with constraint-based, content-based, and hybrid methods",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "methods": {
            "knowledge_based": "Constraint-based filtering with hard/soft constraints",
            "content_based": "Content similarity using 3 different metrics",
            "hybrid": "Combines both constraint and content-based approaches"
        },
        "similarity_methods": [
            {"name": "cosine", "description": "Cosine similarity between feature vectors"},
            {"name": "pearson", "description": "Pearson correlation coefficient"},
            {"name": "euclidean", "description": "Euclidean distance-based similarity"}
        ],
        "endpoints": {
            "GET /health": "Health check with model status",
            "GET /games": "Browse all games with pagination",
            "GET /games/search": "Search games by title",
            "GET /games/{title}": "Get specific game details",
            "POST /recommend/constraint": "Knowledge-based constraint recommendations",
            "POST /recommend/content": "Content-based similarity recommendations",
            "POST /recommend/hybrid": "Hybrid recommendations",
            "GET /similar/{title}": "Find similar games to a specific game",
            "GET /compare/{title}": "Compare all similarity methods",
            "GET /stats": "Database and model statistics",
            "GET /tags": "Get all available tags",
            "GET /languages": "Get all supported languages",
            "GET /developers": "Get all developers",
            "GET /publishers": "Get all publishers",
            "POST /retrain": "Retrain all models"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "games_loaded": len(recommender.games),
        "models_status": {
            "cosine": len(recommender.top_k_similarities.get('cosine', {})) > 0,
            "pearson": len(recommender.top_k_similarities.get('pearson', {})) > 0,
            "euclidean": len(recommender.top_k_similarities.get('euclidean', {})) > 0,
            "game_features": recommender.game_features is not None
        },
        "model_shape": {
            "games": len(recommender.games),
            "features": recommender.game_features.shape[1] if recommender.game_features is not None else 0
        }
    }

@app.get("/games")
def get_games(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("popularity_score", description="Field to sort by"),
    sort_order: int = Query(-1, description="Sort order: 1 for ascending, -1 for descending")
):
    """Get paginated list of games"""
    skip = (page - 1) * limit
    
    # Validate sort field
    valid_sort_fields = ["title", "popularity_score", "discounted_price", "release_year", 
                         "overall_sentiment_score", "all_reviews_count"]
    if sort_by not in valid_sort_fields:
        sort_by = "popularity_score"
    
    games = list(db.steam_games.find({}, {"_id": 0})
                 .sort(sort_by, sort_order)
                 .skip(skip)
                 .limit(limit))
    
    total = db.steam_games.count_documents({})
    
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": (total + limit - 1) // limit,
        "sort": {"by": sort_by, "order": sort_order},
        "games": convert_mongo_data(games)
    }

@app.get("/games/search")
def search_games(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=50, description="Maximum results")
):
    """Search games by title"""
    games = list(db.steam_games.find(
        {"$or": [
            {"title": {"$regex": q, "$options": "i"}},
            {"title_lower": {"$regex": q.lower()}}
        ]},
        {"_id": 0}
    ).limit(limit))
    
    return {
        "query": q,
        "count": len(games),
        "games": convert_mongo_data(games)
    }

@app.get("/games/{title}")
def get_game_by_title(title: str):
    """Get specific game by title"""
    game = db.steam_games.find_one(
        {"$or": [
            {"title": title},
            {"title_lower": title.lower()}
        ]},
        {"_id": 0}
    )
    
    if not game:
        raise HTTPException(status_code=404, detail=f"Game '{title}' not found")
    
    return {"game": convert_mongo_data(game)}

@app.post("/recommend/constraint")
def constraint_based_recommendations(request: ConstraintRequest):
    """KNOWLEDGE-BASED: Constraint-based recommendations"""
    try:
        print(f"🔍 Processing constraint-based recommendation...")
        
        # Convert preferences to dict
        prefs_dict = request.preferences.dict()
        
        # Call constraint-based recommender
        results = recommender.constraint_based_recommendations(
            user_preferences=prefs_dict,
            top_n=request.limit
        )
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        # Handle new response format
        perfect_matches_games = results.get('perfect_matches', {}).get('games', [])
        good_matches_games = results.get('good_matches', {}).get('games', [])
        partial_matches_games = results.get('partial_matches', {}).get('games', [])
        
        return {
            "method": "knowledge_based_constraint",
            "timestamp": datetime.now().isoformat(),
            "results": {
                "perfect_matches": {
                    "count": len(perfect_matches_games),
                    "description": "Games that perfectly match all requirements (70%+ match)",
                    "games": perfect_matches_games
                },
                "good_matches": {
                    "count": len(good_matches_games),
                    "description": "Games that match most requirements (50-69% match)",
                    "games": good_matches_games
                },
                "partial_matches": {
                    "count": len(partial_matches_games),
                    "description": "Games that partially match requirements (30-49% match)",
                    "games": partial_matches_games
                }
            },
            "summary": {
                "total_evaluated": results.get('total_evaluated', 0),
                "perfect_match_percentage": round(len(perfect_matches_games) / max(1, results.get('total_evaluated', 1)) * 100, 1)
            }
        }
    except Exception as e:
        print(f"❌ Error in constraint-based recommendation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recommend/content")
def content_based_recommendations(request: ContentRequest):
    """CONTENT-BASED: Similarity-based recommendations"""
    try:
        print(f"🎯 Processing content-based recommendation with {request.method} similarity...")
        
        if not request.cases:
            # Fallback to popular games
            popular = recommender.get_popular_recommendations(request.limit)
            return {
                "method": "content_based",
                "similarity_method": request.method,
                "warning": "No cases provided. Showing popular games instead.",
                "popular_recommendations": popular
            }
        
        # Call content-based recommender
        results = recommender.content_based_recommendations(
            cases=request.cases,
            method=request.method,
            top_n=request.limit
        )
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        # Handle new response format
        highly_similar_games = results.get('highly_similar', {}).get('games', [])
        moderately_similar_games = results.get('moderately_similar', {}).get('games', [])
        somewhat_similar_games = results.get('somewhat_similar', {}).get('games', [])
        
        return {
            "method": "content_based",
            "timestamp": datetime.now().isoformat(),
            "similarity_method": request.method,
            "results": {
                "highly_similar": {
                    "count": len(highly_similar_games),
                    "description": "Games very similar to your cases (70%+ similarity)",
                    "games": highly_similar_games
                },
                "moderately_similar": {
                    "count": len(moderately_similar_games),
                    "description": "Games somewhat similar (40-69% similarity)",
                    "games": moderately_similar_games
                },
                "somewhat_similar": {
                    "count": len(somewhat_similar_games),
                    "description": "Games with some similarity (20-39% similarity)",
                    "games": somewhat_similar_games
                }
            },
            "summary": {
                "total_found": results.get('total_found', 0),
                "method": request.method
            }
        }
    except Exception as e:
        print(f"❌ Error in content-based recommendation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recommend/hybrid")
def hybrid_recommendations(request: HybridRequest):
    """HYBRID: Combine constraint-based and content-based approaches"""
    try:
        print(f"🤝 Processing hybrid recommendation with {request.method} similarity...")
        
        # Convert preferences to dict
        prefs_dict = request.preferences.dict()
        
        # Call hybrid recommender
        results = recommender.hybrid_recommendations(
            user_preferences=prefs_dict,
            cases=request.cases,
            method=request.method,
            top_n=request.limit
        )
        
        return {
            "method": "hybrid",
            "timestamp": datetime.now().isoformat(),
            "similarity_method": request.method,
            "description": "Combines knowledge-based constraint matching and content-based similarity",
            "recommendations": results.get('recommendations', []),
            "statistics": results.get('statistics', {})
        }
    except Exception as e:
        print(f"❌ Error in hybrid recommendation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/similar/{game_title}")
def get_similar_games(
    game_title: str,
    method: str = Query("cosine", description="Similarity method"),
    limit: int = Query(10, ge=1, le=20, description="Number of similar games")
):
    """Get games similar to a specific game"""
    try:
        # Use content-based with single case
        results = recommender.content_based_recommendations(
            cases=[game_title],
            method=method,
            top_n=limit
        )
        
        if 'error' in results:
            raise HTTPException(status_code=404, detail=results['error'])
        
        # Combine all similarity levels
        highly_similar = results.get('highly_similar', {}).get('games', [])
        moderately_similar = results.get('moderately_similar', {}).get('games', [])
        somewhat_similar = results.get('somewhat_similar', {}).get('games', [])
        
        all_similar = highly_similar + moderately_similar + somewhat_similar
        
        return {
            "source_game": game_title,
            "similarity_method": method,
            "count": len(all_similar),
            "similar_games": all_similar[:limit],
            "similarity_distribution": {
                "highly_similar": len(highly_similar),
                "moderately_similar": len(moderately_similar),
                "somewhat_similar": len(somewhat_similar)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare/{game_title}")
def compare_similarity_methods(
    game_title: str,
    limit: int = Query(5, ge=1, le=10, description="Recommendations per method")
):
    """Compare all similarity methods for a single game"""
    try:
        results = recommender.compare_similarity_methods(game_title, limit)
        
        if 'error' in results:
            raise HTTPException(status_code=404, detail=results['error'])
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """Get database and model statistics"""
    try:
        # Basic counts
        total_games = db.steam_games.count_documents({})
        
        # Price statistics
        price_stats = list(db.steam_games.aggregate([
            {"$group": {
                "_id": None,
                "avg_price": {"$avg": "$discounted_price"},
                "max_price": {"$max": "$discounted_price"},
                "min_price": {"$min": "$discounted_price"},
                "free_games": {"$sum": {"$cond": [{"$eq": ["$discounted_price", 0]}, 1, 0]}}
            }}
        ]))
        
        # Tag statistics
        tag_stats = list(db.steam_games.aggregate([
            {"$unwind": "$tags"},
            {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 20}
        ]))
        
        # Developer statistics
        dev_stats = list(db.steam_games.aggregate([
            {"$group": {"_id": "$developer", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 15}
        ]))
        
        return {
            "database": {
                "total_games": total_games,
                "price_statistics": price_stats[0] if price_stats else {},
                "top_tags": tag_stats,
                "top_developers": dev_stats
            },
            "model": {
                "games_loaded": len(recommender.games),
                "feature_dimensions": recommender.game_features.shape[1] if recommender.game_features is not None else 0,
                "similarity_methods_loaded": {
                    "cosine": len(recommender.top_k_similarities.get('cosine', {})) > 0,
                    "pearson": len(recommender.top_k_similarities.get('pearson', {})) > 0,
                    "euclidean": len(recommender.top_k_similarities.get('euclidean', {})) > 0
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tags")
def get_tags(
    limit: int = Query(100, ge=1, le=500, description="Maximum tags to return"),
    min_count: int = Query(1, ge=1, description="Minimum occurrences")
):
    """Get all available tags with counts"""
    pipeline = [
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$match": {"count": {"$gte": min_count}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    
    tags = list(db.steam_games.aggregate(pipeline))
    
    return {
        "tags": convert_mongo_data(tags),
        "total_unique": len(tags)
    }

@app.get("/languages")
def get_languages():
    """Get all supported languages"""
    languages = db.steam_games.distinct("languages")
    return {
        "languages": sorted([lang for lang in languages if lang]),
        "count": len(languages)
    }

@app.get("/developers")
def get_developers(
    limit: int = Query(100, ge=1, le=500, description="Maximum developers to return")
):
    """Get all developers"""
    developers = db.steam_games.distinct("developer")
    filtered = [d for d in developers if d]
    return {
        "developers": sorted(filtered)[:limit],
        "count": len(filtered)
    }

@app.get("/publishers")
def get_publishers(
    limit: int = Query(100, ge=1, le=500, description="Maximum publishers to return")
):
    """Get all publishers"""
    publishers = db.steam_games.distinct("publisher")
    filtered = [p for p in publishers if p]
    return {
        "publishers": sorted(filtered)[:limit],
        "count": len(filtered)
    }

@app.post("/retrain")
def retrain_model():
    """Retrain all recommendation models"""
    try:
        print("🔄 Retraining all models...")
        success = recommender.train_models()
        
        if success:
            return {
                "status": "success",
                "message": "All models retrained successfully",
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "games_count": len(recommender.games),
                    "models_trained": ["cosine", "pearson", "euclidean"]
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Model training failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)