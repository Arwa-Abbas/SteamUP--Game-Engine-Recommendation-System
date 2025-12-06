import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('explore');
  const [games, setGames] = useState([]);
  const [likedGames, setLikedGames] = useState([]);
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('popularity');
  
  // Available options from backend
  const [availableTags, setAvailableTags] = useState([]);
  const [availableLanguages, setAvailableLanguages] = useState([]);
  const [availableDevelopers, setAvailableDevelopers] = useState([]);
  const [availablePublishers, setAvailablePublishers] = useState([]);
  
  // Similarity method for each recommendation type
  const [contentMethod, setContentMethod] = useState('cosine');
  const [hybridMethod, setHybridMethod] = useState('cosine');
  
  // User preferences for constraint-based
  const [preferences, setPreferences] = useState({
    max_price: 50,
    min_price: 0,
    preferred_tags: [],
    languages: [],
    developers: [],
    publishers: [],
    system_specs: {
      memory_gb: null,
      storage_gb: null,
      os_type: '',
      require_ssd: false
    },
    min_sentiment: 0.0,
    min_reviews: 0
  });

  // Extract Steam app ID from URL
  const extractSteamAppId = (url) => {
    if (!url || typeof url !== 'string') return null;
    try {
      const patterns = [
        /store\.steampowered\.com\/app\/(\d+)/,
        /\/app\/(\d+)/,
        /appid=(\d+)/,
        /\/(\d+)\/?$/
      ];
      for (const pattern of patterns) {
        const match = url.match(pattern);
        if (match && match[1]) return match[1];
      }
      return null;
    } catch (error) {
      console.error('Error extracting app ID:', error);
      return null;
    }
  };

  const getGamePosterUrl = (game) => {
    if (!game) return null;
    const appId = extractSteamAppId(game.link);
    if (appId) {
      return `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/header.jpg`;
    }
    return null;
  };

  const getPlaceholderImage = (title) => {
    if (!title) return '';
    const colors = ['8b5cf6', 'ec4899', '06b6d4', '10b981', 'f59e0b'];
    let hash = 0;
    for (let i = 0; i < title.length; i++) {
      hash = title.charCodeAt(i) + ((hash << 5) - hash);
    }
    const colorIndex = Math.abs(hash) % colors.length;
    const color = colors[colorIndex];
    const shortTitle = title.length > 30 ? title.substring(0, 27) + '...' : title;
    const encodedTitle = encodeURIComponent(shortTitle);
    return `https://via.placeholder.com/460x215/${color}/ffffff?text=${encodedTitle}`;
  };

  // Load liked games from storage on mount
  useEffect(() => {
    const stored = localStorage.getItem('likedGames');
    if (stored) {
      try {
        setLikedGames(JSON.parse(stored));
      } catch (e) {
        console.error('Error loading liked games:', e);
      }
    }
    loadInitialData();
  }, []);

  // Save liked games to storage whenever they change
  useEffect(() => {
    localStorage.setItem('likedGames', JSON.stringify(likedGames));
  }, [likedGames]);

  const loadInitialData = async () => {
    try {
      console.log("Loading initial data...");
      
      const statsRes = await fetch(`${API_BASE}/stats`);
      const statsData = await statsRes.json();
      console.log("Stats loaded:", statsData);
      setStats(statsData);
      
      const tagsRes = await fetch(`${API_BASE}/tags?limit=100&min_count=2`);
      const tagsData = await tagsRes.json();
      console.log("Tags loaded:", tagsData.tags?.length);
      setAvailableTags(tagsData.tags?.map(tag => tag._id) || []);
      
      const langsRes = await fetch(`${API_BASE}/languages`);
      const langsData = await langsRes.json();
      console.log("Languages loaded:", langsData.languages?.length);
      setAvailableLanguages(langsData.languages || []);
      
      const devsRes = await fetch(`${API_BASE}/developers?limit=100`);
      const devsData = await devsRes.json();
      console.log("Developers loaded:", devsData.developers?.length);
      setAvailableDevelopers(devsData.developers || []);
      
      const pubsRes = await fetch(`${API_BASE}/publishers?limit=100`);
      const pubsData = await pubsRes.json();
      console.log("Publishers loaded:", pubsData.publishers?.length);
      setAvailablePublishers(pubsData.publishers || []);
      
      await loadGames();
    } catch (error) {
      console.error('Error loading initial data:', error);
    }
  };

  const loadGames = async (page = 1) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/games?page=${page}&limit=24&sort_by=popularity_score&sort_order=-1`);
      const data = await response.json();
      console.log("Games loaded:", data.games?.length);
      setGames(data.games || []);
    } catch (error) {
      console.error('Error loading games:', error);
      setGames([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      loadGames();
      return;
    }
    
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/games/search?q=${encodeURIComponent(searchQuery)}&limit=24`);
      const data = await response.json();
      console.log("Search results:", data.games?.length);
      setGames(data.games || []);
    } catch (error) {
      console.error('Search error:', error);
      setGames([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (games) => {
    const sorted = [...games];
    switch (sortBy) {
      case 'alphabetical':
        return sorted.sort((a, b) => a.title.localeCompare(b.title));
      case 'price-low':
        return sorted.sort((a, b) => (a.discounted_price || 0) - (b.discounted_price || 0));
      case 'price-high':
        return sorted.sort((a, b) => (b.discounted_price || 0) - (a.discounted_price || 0));
      case 'rating':
        return sorted.sort((a, b) => (b.overall_sentiment_score || 0) - (a.overall_sentiment_score || 0));
      default:
        return sorted;
    }
  };

  const toggleLike = (game) => {
    const isLiked = likedGames.some(g => g.title === game.title);
    if (isLiked) {
      setLikedGames(likedGames.filter(g => g.title !== game.title));
    } else {
      setLikedGames([...likedGames, game]);
    }
  };

  const isGameLiked = (game) => {
    return likedGames.some(g => g.title === game.title);
  };

  const getContentRecommendations = async () => {
    if (likedGames.length === 0) {
      alert('Please like some games first to get content-based recommendations');
      return;
    }
    
    try {
      setLoading(true);
      console.log("Getting content recommendations for:", likedGames.map(g => g.title));
      
      const response = await fetch(`${API_BASE}/recommend/content`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cases: likedGames.map(g => g.title),
          method: contentMethod,
          limit: 24
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Content recommendations received:", data);
      
      // FIXED: Properly handle the API response format
      const recommendationsData = {
        results: {
          highly_similar: {
            games: data.results?.highly_similar?.games || []
          },
          moderately_similar: {
            games: data.results?.moderately_similar?.games || []
          },
          somewhat_similar: {
            games: data.results?.somewhat_similar?.games || []
          }
        }
      };
      
      setRecommendations({ 
        method: 'content', 
        data: recommendationsData, 
        similarity: contentMethod 
      });
    } catch (error) {
      console.error('Content recommendation error:', error);
      alert('Error getting content-based recommendations: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const getConstraintRecommendations = async () => {
    try {
      setLoading(true);
      console.log("Getting constraint recommendations with preferences:", preferences);
      
      // Prepare preferences properly
      const requestPreferences = {
        max_price: preferences.max_price,
        min_price: preferences.min_price,
        preferred_tags: preferences.preferred_tags,
        languages: preferences.languages,
        developers: preferences.developers,
        publishers: preferences.publishers,
        system_specs: preferences.system_specs,
        min_sentiment: preferences.min_sentiment,
        min_reviews: preferences.min_reviews
      };
      
      const response = await fetch(`${API_BASE}/recommend/constraint`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preferences: requestPreferences,
          limit: 24
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Constraint recommendations received:", data);
      
      // FIXED: Properly handle the API response format
      const recommendationsData = {
        results: {
          perfect_matches: {
            games: data.results?.perfect_matches?.games || []
          },
          good_matches: {
            games: data.results?.good_matches?.games || []
          },
          partial_matches: {
            games: data.results?.partial_matches?.games || []
          }
        }
      };
      
      setRecommendations({ 
        method: 'constraint', 
        data: recommendationsData, 
        similarity: 'constraint' 
      });
    } catch (error) {
      console.error('Constraint recommendation error:', error);
      alert('Error getting constraint-based recommendations: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const getHybridRecommendations = async () => {
    if (likedGames.length === 0) {
      alert('Please like some games first to get hybrid recommendations');
      return;
    }
    
    try {
      setLoading(true);
      console.log("Getting hybrid recommendations...");
      
      // Prepare preferences properly
      const requestPreferences = {
        max_price: preferences.max_price,
        min_price: preferences.min_price,
        preferred_tags: preferences.preferred_tags,
        languages: preferences.languages,
        developers: preferences.developers,
        publishers: preferences.publishers,
        system_specs: preferences.system_specs,
        min_sentiment: preferences.min_sentiment,
        min_reviews: preferences.min_reviews
      };
      
      const response = await fetch(`${API_BASE}/recommend/hybrid`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preferences: requestPreferences,
          cases: likedGames.map(g => g.title),
          method: hybridMethod,
          limit: 24
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Hybrid recommendations received:", data);
      
      // FIXED: Convert recommendations to proper format
      const recommendationsArray = data.recommendations || [];
      
      // Convert to game objects with proper fields
      const formattedRecommendations = recommendationsArray.map(rec => ({
        title: rec.title,
        developer: rec.developer,
        discounted_price: rec.price || rec.discounted_price || 0,
        original_price: rec.original_price || rec.price || 0,
        discount_percentage: rec.discount || rec.discount_percentage || 0,
        overall_sentiment_score: rec.sentiment || rec.overall_sentiment_score || 0.5,
        all_reviews_count: rec.reviews || rec.all_reviews_count || 0,
        tags: rec.tags || [],
        link: rec.link || '#',
        release_year: rec.release_year,
        score: rec.hybrid_score || rec.similarity || rec.score || 0
      }));
      
      setRecommendations({ 
        method: 'hybrid', 
        data: { recommendations: formattedRecommendations }, 
        similarity: hybridMethod 
      });
    } catch (error) {
      console.error('Hybrid recommendation error:', error);
      alert('Error getting hybrid recommendations: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const addTag = (tag) => {
    if (tag && !preferences.preferred_tags.includes(tag)) {
      setPreferences(prev => ({
        ...prev,
        preferred_tags: [...prev.preferred_tags, tag]
      }));
    }
  };

  const removeTag = (tag) => {
    setPreferences(prev => ({
      ...prev,
      preferred_tags: prev.preferred_tags.filter(t => t !== tag)
    }));
  };

  const addItem = (list, item) => {
    if (item && !preferences[list].includes(item)) {
      setPreferences(prev => ({
        ...prev,
        [list]: [...prev[list], item]
      }));
    }
  };

  const removeItem = (list, item) => {
    setPreferences(prev => ({
      ...prev,
      [list]: prev[list].filter(i => i !== item)
    }));
  };

  const renderGameCard = (game, index, showScore = false) => {
    if (!game) return null;
    
    const posterUrl = getGamePosterUrl(game);
    const liked = isGameLiked(game);
    
    // Ensure all fields exist
    const gameData = {
      title: game.title || 'Unknown Game',
      developer: game.developer || game.developerName || 'Unknown',
      discounted_price: game.discounted_price || game.price || game.discountedPrice || 0,
      original_price: game.original_price || game.originalPrice || game.discounted_price || 0,
      discount_percentage: game.discount_percentage || game.discount || game.discountPercentage || 0,
      overall_sentiment_score: game.overall_sentiment_score || game.sentiment || game.overallSentimentScore || 0.5,
      all_reviews_count: game.all_reviews_count || game.reviews || game.allReviewsCount || 0,
      tags: game.tags || game.genres || [],
      link: game.link || game.url || '#',
      release_year: game.release_year || game.year || game.releaseYear,
      score: game.score || game.similarity || game.hybrid_score || 0
    };
    
    const score = showScore ? gameData.score : 0;
    
    return (
      <div key={index} className="game-card" style={{ animationDelay: `${index * 0.05}s` }}>
        <div className="game-poster">
          <img 
            src={posterUrl || getPlaceholderImage(gameData.title)}
            alt={gameData.title}
            className="poster-image"
            onError={(e) => { 
              e.target.onerror = null; 
              e.target.src = getPlaceholderImage(gameData.title); 
            }}
          />
          
          <button 
            className={`like-btn ${liked ? 'liked' : ''}`}
            onClick={(e) => { e.stopPropagation(); toggleLike(game); }}
            title={liked ? 'Remove from liked' : 'Add to liked'}
          >
            {liked ? '❤️' : '🤍'}
          </button>
          
          {gameData.discount_percentage > 0 && gameData.discount_percentage < 100 && (
            <div className="discount-badge">-{Math.round(gameData.discount_percentage)}%</div>
          )}
          
          {(gameData.discounted_price === 0 || gameData.discount_percentage === 100) && (
            <div className="free-badge">FREE</div>
          )}
        </div>
        
        {showScore && score > 0 && (
          <div className="match-badge">{Math.round(score)}% Match</div>
        )}
        
        <div className="card-content">
          <h3 title={gameData.title}>{gameData.title}</h3>
          
          <div className="game-meta">
            <span className="price">
              {gameData.discounted_price === 0 ? 'FREE' : `$${gameData.discounted_price.toFixed(2)}`}
            </span>
            {gameData.discount_percentage > 0 && gameData.discount_percentage < 100 && gameData.original_price > gameData.discounted_price && (
              <span className="original-price">${gameData.original_price.toFixed(2)}</span>
            )}
          </div>
          
          <div className="game-stats">
            <div className="stat" title="Rating">
              <span className="stat-icon">⭐</span>
              <span>{Math.round((gameData.overall_sentiment_score || 0.5) * 100)}%</span>
            </div>
            <div className="stat" title="Reviews">
              <span className="stat-icon">👥</span>
              <span>{(gameData.all_reviews_count || 0).toLocaleString()}</span>
            </div>
          </div>
          
          {gameData.tags && gameData.tags.length > 0 && (
            <div className="game-tags">
              {gameData.tags.slice(0, 3).map((tag, i) => (
                <span key={i} className="tag" title={tag}>
                  {tag.length > 15 ? tag.substring(0, 12) + '...' : tag}
                </span>
              ))}
              {gameData.tags.length > 3 && (
                <span className="tag" title={gameData.tags.slice(3).join(', ')}>
                  +{gameData.tags.length - 3}
                </span>
              )}
            </div>
          )}
          
          <div className="game-footer">
            <span className="developer" title={gameData.developer}>
              {gameData.developer.length > 20 ? gameData.developer.substring(0, 18) + '...' : gameData.developer}
            </span>
            {gameData.release_year && (
              <span className="year" title="Release Year">{gameData.release_year}</span>
            )}
          </div>
          
          {gameData.link && gameData.link !== '#' && (
            <a 
              href={gameData.link} 
              target="_blank" 
              rel="noopener noreferrer"
              className="view-link"
              onClick={(e) => e.stopPropagation()}
              title={`View ${gameData.title} on Steam`}
            >
              View on Steam →
            </a>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="app">
      <div className="animated-bg">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
      </div>

      <header className="header">
        <div className="header-content">
          <div className="logo">
            <h1>GameFinder</h1>
          </div>
          
          <nav className="nav-tabs">
            <button className={`nav-tab ${activeTab === 'explore' ? 'active' : ''}`} onClick={() => setActiveTab('explore')}>
              Explore
            </button>
            <button className={`nav-tab ${activeTab === 'liked' ? 'active' : ''}`} onClick={() => setActiveTab('liked')}>
              Liked ({likedGames.length})
            </button>
            <button className={`nav-tab ${activeTab === 'content' ? 'active' : ''}`} onClick={() => setActiveTab('content')}>
              Content-Based
            </button>
            <button className={`nav-tab ${activeTab === 'constraint' ? 'active' : ''}`} onClick={() => setActiveTab('constraint')}>
              Constraint-Based
            </button>
            <button className={`nav-tab ${activeTab === 'hybrid' ? 'active' : ''}`} onClick={() => setActiveTab('hybrid')}>
              Hybrid
            </button>
            <button className={`nav-tab ${activeTab === 'stats' ? 'active' : ''}`} onClick={() => setActiveTab('stats')}>
              Stats
            </button>
          </nav>
        </div>
      </header>

      <main className="main-content">
        
        {/* EXPLORE TAB */}
        {activeTab === 'explore' && (
          <div className="tab-content">
            <div className="section-header">
              <h2>Discover Games</h2>
              <div className="controls">
                <form onSubmit={handleSearch} className="search-form">
                  <input
                    type="text"
                    placeholder="Search games..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="search-input"
                  />
                  <button type="submit" className="search-btn">🔍</button>
                </form>
                
                <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className="sort-select">
                  <option value="popularity">Sort by Popularity</option>
                  <option value="alphabetical">Sort A-Z</option>
                  <option value="price-low">Price: Low to High</option>
                  <option value="price-high">Price: High to Low</option>
                  <option value="rating">Rating</option>
                </select>
              </div>
            </div>
            
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Loading games...</p>
              </div>
            ) : (
              <div className="games-grid">
                {handleSort(games).map((game, index) => renderGameCard(game, index))}
              </div>
            )}
          </div>
        )}

        {/* LIKED GAMES TAB */}
        {activeTab === 'liked' && (
          <div className="tab-content">
            <div className="section-header">
              <h2>Your Liked Games</h2>
              <p className="subtitle">Games you've marked as favorites</p>
            </div>
            
            {likedGames.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">💔</div>
                <h3>No liked games yet</h3>
                <p>Start exploring and click the heart icon on games you like!</p>
                <button onClick={() => setActiveTab('explore')} className="btn-primary">
                  Explore Games
                </button>
              </div>
            ) : (
              <div className="games-grid">
                {likedGames.map((game, index) => renderGameCard(game, index))}
              </div>
            )}
          </div>
        )}

        {/* CONTENT-BASED TAB */}
        {activeTab === 'content' && (
          <div className="tab-content">
            <div className="section-header">
              <h2>Content-Based Recommendations</h2>
              <div className="method-selector">
                <label>Similarity Method:</label>
                <button className={`method-btn ${contentMethod === 'cosine' ? 'active' : ''}`} onClick={() => setContentMethod('cosine')}>Cosine</button>
                <button className={`method-btn ${contentMethod === 'pearson' ? 'active' : ''}`} onClick={() => setContentMethod('pearson')}>Pearson</button>
                <button className={`method-btn ${contentMethod === 'euclidean' ? 'active' : ''}`} onClick={() => setContentMethod('euclidean')}>Euclidean</button>
              </div>
            </div>
            
            <button onClick={getContentRecommendations} className="btn-primary" disabled={likedGames.length === 0}>
              🔍 Get Recommendations Based on Liked Games
            </button>
            
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Getting recommendations...</p>
              </div>
            ) : recommendations && recommendations.method === 'content' ? (
              <div className="recommendations-container">
                {recommendations.data.results.highly_similar?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3>🎯 Highly Similar (70%+ match)</h3>
                    <div className="games-grid">
                      {recommendations.data.results.highly_similar.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.moderately_similar?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3>👍 Moderately Similar (40-69% match)</h3>
                    <div className="games-grid">
                      {recommendations.data.results.moderately_similar.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.somewhat_similar?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3>✨ Somewhat Similar (20-39% match)</h3>
                    <div className="games-grid">
                      {recommendations.data.results.somewhat_similar.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.highly_similar?.games?.length === 0 && 
                 recommendations.data.results.moderately_similar?.games?.length === 0 && 
                 recommendations.data.results.somewhat_similar?.games?.length === 0 && (
                  <div className="empty-state">
                    <div className="empty-icon">🔍</div>
                    <h3>No similar games found</h3>
                    <p>Try liking different games or change similarity method.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">🔍</div>
                <h3>No recommendations yet</h3>
                <p>Like some games and click the button above to get personalized recommendations!</p>
              </div>
            )}
          </div>
        )}

        {/* CONSTRAINT-BASED TAB */}
        {activeTab === 'constraint' && (
          <div className="tab-content">
            <div className="section-header">
              <h2>Constraint-Based Recommendations</h2>
            </div>
            
            <div className="filters-panel">
              <div className="filter-grid">
                <div className="filter-group">
                  <label>Price Range: ${preferences.min_price} - ${preferences.max_price}</label>
                  <input 
                    type="range" 
                    min="0" 
                    max="100" 
                    value={preferences.max_price}
                    onChange={(e) => setPreferences(prev => ({ ...prev, max_price: parseInt(e.target.value) }))}
                    className="slider" 
                  />
                </div>
                
                <div className="filter-group">
                  <label>Preferred Tags</label>
                  <div className="chips-container">
                    {preferences.preferred_tags.map(tag => (
                      <span key={tag} className="chip">
                        {tag} <button onClick={() => removeTag(tag)}>×</button>
                      </span>
                    ))}
                  </div>
                  <select onChange={(e) => { addTag(e.target.value); e.target.value = ''; }}>
                    <option value="">Select tag...</option>
                    {availableTags.map(tag => (
                      <option key={tag} value={tag}>{tag}</option>
                    ))}
                  </select>
                </div>
                
                <div className="filter-group">
                  <label>Languages</label>
                  <div className="chips-container">
                    {preferences.languages.map(lang => (
                      <span key={lang} className="chip">
                        {lang} <button onClick={() => removeItem('languages', lang)}>×</button>
                      </span>
                    ))}
                  </div>
                  <select onChange={(e) => { addItem('languages', e.target.value); e.target.value = ''; }}>
                    <option value="">Select language...</option>
                    {availableLanguages.map(lang => (
                      <option key={lang} value={lang}>{lang}</option>
                    ))}
                  </select>
                </div>
                
                <div className="filter-group">
                  <label>Developers</label>
                  <div className="chips-container">
                    {preferences.developers.map(dev => (
                      <span key={dev} className="chip">
                        {dev} <button onClick={() => removeItem('developers', dev)}>×</button>
                      </span>
                    ))}
                  </div>
                  <select onChange={(e) => { addItem('developers', e.target.value); e.target.value = ''; }}>
                    <option value="">Select developer...</option>
                    {availableDevelopers.map(dev => (
                      <option key={dev} value={dev}>{dev}</option>
                    ))}
                  </select>
                </div>
                
                <div className="filter-group">
                  <label>Publishers</label>
                  <div className="chips-container">
                    {preferences.publishers.map(pub => (
                      <span key={pub} className="chip">
                        {pub} <button onClick={() => removeItem('publishers', pub)}>×</button>
                      </span>
                    ))}
                  </div>
                  <select onChange={(e) => { addItem('publishers', e.target.value); e.target.value = ''; }}>
                    <option value="">Select publisher...</option>
                    {availablePublishers.map(pub => (
                      <option key={pub} value={pub}>{pub}</option>
                    ))}
                  </select>
                </div>
                
                <div className="filter-group">
                  <label>System Requirements</label>
                  <div className="specs-grid">
                    <div className="spec-item">
                      <label>RAM (GB):</label>
                      <select 
                        value={preferences.system_specs.memory_gb || ''}
                        onChange={(e) => setPreferences(prev => ({
                          ...prev,
                          system_specs: {
                            ...prev.system_specs,
                            memory_gb: e.target.value ? parseInt(e.target.value) : null
                          }
                        }))}
                      >
                        <option value="">Any</option>
                        <option value="2">2 GB</option>
                        <option value="4">4 GB</option>
                        <option value="8">8 GB</option>
                        <option value="16">16 GB</option>
                        <option value="32">32 GB</option>
                      </select>
                    </div>
                    
                    <div className="spec-item">
                      <label>Storage (GB):</label>
                      <select 
                        value={preferences.system_specs.storage_gb || ''}
                        onChange={(e) => setPreferences(prev => ({
                          ...prev,
                          system_specs: {
                            ...prev.system_specs,
                            storage_gb: e.target.value ? parseInt(e.target.value) : null
                          }
                        }))}
                      >
                        <option value="">Any</option>
                        <option value="10">10 GB</option>
                        <option value="20">20 GB</option>
                        <option value="50">50 GB</option>
                        <option value="100">100 GB</option>
                      </select>
                    </div>
                    
                    <div className="spec-item">
                      <label>OS:</label>
                      <select 
                        value={preferences.system_specs.os_type || ''}
                        onChange={(e) => setPreferences(prev => ({
                          ...prev,
                          system_specs: {
                            ...prev.system_specs,
                            os_type: e.target.value || ''
                          }
                        }))}
                      >
                        <option value="">Any</option>
                        <option value="windows">Windows</option>
                        <option value="linux">Linux</option>
                        <option value="mac">Mac</option>
                      </select>
                    </div>
                    
                    <div className="spec-item">
                      <label className="checkbox-label">
                        <input 
                          type="checkbox" 
                          checked={preferences.system_specs.require_ssd}
                          onChange={(e) => setPreferences(prev => ({
                            ...prev,
                            system_specs: {
                              ...prev.system_specs,
                              require_ssd: e.target.checked
                            }
                          }))}
                        />
                        SSD Required
                      </label>
                    </div>
                  </div>
                </div>
                
                <div className="filter-group">
                  <label>Min Sentiment: {Math.round(preferences.min_sentiment * 100)}%</label>
                  <input 
                    type="range" 
                    min="0" 
                    max="100" 
                    value={preferences.min_sentiment * 100}
                    onChange={(e) => setPreferences(prev => ({ ...prev, min_sentiment: parseInt(e.target.value) / 100 }))}
                    className="slider" 
                  />
                </div>
                
                <div className="filter-group">
                  <label>Min Reviews: {preferences.min_reviews}</label>
                  <input 
                    type="range" 
                    min="0" 
                    max="10000" 
                    step="100" 
                    value={preferences.min_reviews}
                    onChange={(e) => setPreferences(prev => ({ ...prev, min_reviews: parseInt(e.target.value) }))}
                    className="slider" 
                  />
                </div>
              </div>
              
              <button onClick={getConstraintRecommendations} className="btn-primary">
                🎯 Get Constraint-Based Recommendations
              </button>
            </div>
            
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Getting recommendations...</p>
              </div>
            ) : recommendations && recommendations.method === 'constraint' ? (
              <div className="recommendations-container">
                {recommendations.data.results.perfect_matches?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3>🎯 Perfect Matches (70%+ match)</h3>
                    <div className="games-grid">
                      {recommendations.data.results.perfect_matches.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.good_matches?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3>👍 Good Matches (50-69% match)</h3>
                    <div className="games-grid">
                      {recommendations.data.results.good_matches.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.partial_matches?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3>✨ Partial Matches (30-49% match)</h3>
                    <div className="games-grid">
                      {recommendations.data.results.partial_matches.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.perfect_matches?.games?.length === 0 && 
                 recommendations.data.results.good_matches?.games?.length === 0 && 
                 recommendations.data.results.partial_matches?.games?.length === 0 && (
                  <div className="empty-state">
                    <div className="empty-icon">🎯</div>
                    <h3>No matches found</h3>
                    <p>Try relaxing your constraints or select different preferences.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">🎯</div>
                <h3>No recommendations yet</h3>
                <p>Set your preferences and click the button above!</p>
              </div>
            )}
          </div>
        )}

        {/* HYBRID TAB */}
        {activeTab === 'hybrid' && (
          <div className="tab-content">
            <div className="section-header">
              <h2>Hybrid Recommendations</h2>
              <div className="method-selector">
                <label>Similarity Method:</label>
                <button className={`method-btn ${hybridMethod === 'cosine' ? 'active' : ''}`} onClick={() => setHybridMethod('cosine')}>Cosine</button>
                <button className={`method-btn ${hybridMethod === 'pearson' ? 'active' : ''}`} onClick={() => setHybridMethod('pearson')}>Pearson</button>
                <button className={`method-btn ${hybridMethod === 'euclidean' ? 'active' : ''}`} onClick={() => setHybridMethod('euclidean')}>Euclidean</button>
              </div>
            </div>
            
            <button onClick={getHybridRecommendations} className="btn-primary" disabled={likedGames.length === 0}>
              🤝 Get Hybrid Recommendations
            </button>
            
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Getting recommendations...</p>
              </div>
            ) : recommendations && recommendations.method === 'hybrid' ? (
              <div className="recommendation-category">
                <h3>🤝 Hybrid Recommendations</h3>
                <div className="games-grid">
                  {recommendations.data.recommendations.map((game, index) => 
                    renderGameCard(game, index, true)
                  )}
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">🤝</div>
                <h3>No recommendations yet</h3>
                <p>Like some games and set constraints, then click the button above!</p>
              </div>
            )}
          </div>
        )}

        {/* STATS TAB */}
        {activeTab === 'stats' && stats && (
          <div className="tab-content">
            <h2>Platform Statistics</h2>
            
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-icon">🎮</div>
                <div className="stat-value">{stats.database.total_games}</div>
                <div className="stat-label">Total Games</div>
              </div>
              
              <div className="stat-card">
                <div className="stat-icon">💰</div>
                <div className="stat-value">${stats.database.price_statistics?.avg_price?.toFixed(2) || '0.00'}</div>
                <div className="stat-label">Average Price</div>
              </div>
              
              <div className="stat-card">
                <div className="stat-icon">🆓</div>
                <div className="stat-value">{stats.database.price_statistics?.free_games || 0}</div>
                <div className="stat-label">Free Games</div>
              </div>
              
              <div className="stat-card">
                <div className="stat-icon">💎</div>
                <div className="stat-value">${stats.database.price_statistics?.max_price?.toFixed(2) || '0.00'}</div>
                <div className="stat-label">Highest Price</div>
              </div>
            </div>
            
            <div className="model-stats">
              <h3>Model Information</h3>
              <div className="stats-cards">
                <div className="model-stat">
                  <span className="label">Games Loaded:</span>
                  <span className="value">{stats.model.games_loaded}</span>
                </div>
                <div className="model-stat">
                  <span className="label">Features:</span>
                  <span className="value">{stats.model.feature_dimensions}</span>
                </div>
              </div>
            </div>
            
            {stats.database.top_tags && (
              <div className="tags-cloud">
                <h3>Popular Tags</h3>
                <div className="cloud">
                  {stats.database.top_tags.slice(0, 30).map((tag, index) => (
                    <span key={index} className="cloud-tag"
                      style={{ fontSize: `${0.9 + (tag.count / stats.database.top_tags[0].count) * 0.8}em` }}>
                      {tag._id} ({tag.count})
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>GameFinder Pro • Advanced Recommendation System</p>
      </footer>
    </div>
  );
}

export default App;