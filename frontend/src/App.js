import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('explore');
  const [games, setGames] = useState([]);
  const [likedGames, setLikedGames] = useState([]);
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [initialLoading, setInitialLoading] = useState(true);
  const [stats, setStats] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('popularity');
  
  // Game details modal
  const [selectedGame, setSelectedGame] = useState(null);
  const [showGameDetails, setShowGameDetails] = useState(false);
  
  // Available options from backend
  const [availableTags, setAvailableTags] = useState([]);
  const [availableLanguages, setAvailableLanguages] = useState([]);
  const [availableDevelopers, setAvailableDevelopers] = useState([]);
  const [availablePublishers, setAvailablePublishers] = useState([]);
  
  // Similarity method for each recommendation type
  const [contentMethod, setContentMethod] = useState('cosine');
  const [hybridMethod, setHybridMethod] = useState('cosine');
  
  // Game type filter
  const [gameType, setGameType] = useState('all');
  
  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalGames, setTotalGames] = useState(0);
  
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

  // State for custom tag input
  const [customTag, setCustomTag] = useState('');
  const [customDeveloper, setCustomDeveloper] = useState('');
  const [customPublisher, setCustomPublisher] = useState('');

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

  // Get better game images
  const getSteamImages = (game) => {
    const appId = extractSteamAppId(game?.link);
    if (!appId) return { header: null, capsule: null };
    
    return {
      header: `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/header.jpg`,
      capsule: `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/capsule_231x87.jpg`,
      capsule_616: `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/capsule_616x353.jpg`,
      library: `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/library_600x900.jpg`
    };
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
      setInitialLoading(true);
      console.log("Loading initial data...");
      
      // Load stats
      const statsRes = await fetch(`${API_BASE}/stats`);
      const statsData = await statsRes.json();
      console.log("Stats loaded:", statsData);
      setStats(statsData);
      
      // Load ALL tags without limit
      const tagsRes = await fetch(`${API_BASE}/tags?min_count=1`);
      const tagsData = await tagsRes.json();
      const allTags = tagsData.tags?.map(tag => tag._id) || [];
      console.log("Tags loaded:", allTags.length);
      // Sort tags alphabetically
      setAvailableTags(allTags.sort((a, b) => a.localeCompare(b)));
      
      // Load languages
      const langsRes = await fetch(`${API_BASE}/languages`);
      const langsData = await langsRes.json();
      const allLanguages = langsData.languages || [];
      console.log("Languages loaded:", allLanguages.length);
      setAvailableLanguages(allLanguages.sort((a, b) => a.localeCompare(b)));
      
      // Load ALL developers without limit
      const devsRes = await fetch(`${API_BASE}/developers`);
      const devsData = await devsRes.json();
      const allDevelopers = devsData.developers || [];
      console.log("Developers loaded:", allDevelopers.length);
      // Sort developers alphabetically
      setAvailableDevelopers(allDevelopers.sort((a, b) => a.localeCompare(b)));
      
      // Load ALL publishers without limit
      const pubsRes = await fetch(`${API_BASE}/publishers`);
      const pubsData = await pubsRes.json();
      const allPublishers = pubsData.publishers || [];
      console.log("Publishers loaded:", allPublishers.length);
      // Sort publishers alphabetically
      setAvailablePublishers(allPublishers.sort((a, b) => a.localeCompare(b)));
      
      await loadGames();
    } catch (error) {
      console.error('Error loading initial data:', error);
    } finally {
      setInitialLoading(false);
      setLoading(false);
    }
  };

  const loadGames = async (page = 1, type = gameType) => {
    try {
      setLoading(true);
      let url = `${API_BASE}/games?page=${page}&limit=24&sort_by=popularity_score&sort_order=-1`;
      
      // Add game type filter if not 'all'
      if (type !== 'all') {
        url += `&game_type=${type}`;
      }
      
      // Add sort parameter
      if (sortBy !== 'popularity') {
        const sortMap = {
          'alphabetical': 'title',
          'price-low': 'discounted_price',
          'price-high': 'discounted_price',
          'rating': 'overall_sentiment_score'
        };
        const orderMap = {
          'alphabetical': 1,
          'price-low': 1,
          'price-high': -1,
          'rating': -1
        };
        url += `&sort_by=${sortMap[sortBy]}&sort_order=${orderMap[sortBy]}`;
      }
      
      console.log("Loading games from:", url);
      const response = await fetch(url);
      const data = await response.json();
      console.log("Games loaded:", data.games?.length, "Type:", type, "Page:", page, "Total:", data.total);
      
      setGames(data.games || []);
      setTotalGames(data.total || 0);
      setTotalPages(data.total_pages || 1);
      setCurrentPage(page);
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
      loadGames(1, gameType);
      return;
    }
    
    try {
      setLoading(true);
      let url = `${API_BASE}/games/search?q=${encodeURIComponent(searchQuery)}&limit=24`;
      
      // Add game type filter if not 'all'
      if (gameType !== 'all') {
        url += `&game_type=${gameType}`;
      }
      
      const response = await fetch(url);
      const data = await response.json();
      console.log("Search results:", data.games?.length);
      setGames(data.games || []);
      setTotalGames(data.count || 0);
      setTotalPages(1);
      setCurrentPage(1);
    } catch (error) {
      console.error('Search error:', error);
      setGames([]);
    } finally {
      setLoading(false);
    }
  };

  // Handle game type change
  const handleGameTypeChange = (type) => {
    setGameType(type);
    setCurrentPage(1);
    loadGames(1, type);
  };

  // Handle sort change
  const handleSortChange = (e) => {
    const newSortBy = e.target.value;
    setSortBy(newSortBy);
    setCurrentPage(1);
    loadGames(1, gameType);
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

  const viewGameDetails = async (game) => {
    try {
      setLoading(true);
      
      // Try to get full game details from backend
      const response = await fetch(`${API_BASE}/games/${encodeURIComponent(game.title)}`);
      
      if (response.ok) {
        const data = await response.json();
        setSelectedGame(data.game);
      } else {
        // If API fails, use the basic game data
        setSelectedGame(game);
      }
      
      setShowGameDetails(true);
    } catch (error) {
      console.error('Error loading game details:', error);
      // Fallback to basic game data
      setSelectedGame(game);
      setShowGameDetails(true);
    } finally {
      setLoading(false);
    }
  };

  const closeGameDetails = () => {
    setShowGameDetails(false);
    setSelectedGame(null);
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
      
      // FIXED: Process hybrid recommendations properly
      const recommendationsArray = data.recommendations || [];
      
      const formattedRecommendations = recommendationsArray.map(rec => {
        // Ensure we have correct price data
        const price = rec.price || rec.discounted_price || 0;
        const originalPrice = rec.original_price || rec.price || price;
        
        return {
          title: rec.title,
          developer: rec.developer,
          publisher: rec.publisher,
          discounted_price: price,
          original_price: originalPrice,
          discount_percentage: rec.discount || rec.discount_percentage || 0,
          overall_sentiment_score: rec.sentiment || rec.overall_sentiment_score || 0.5,
          all_reviews_count: rec.reviews || rec.all_reviews_count || 0,
          tags: rec.tags || [],
          categories: rec.categories || [],
          features: rec.features || [],
          languages: rec.languages || [],
          link: rec.link || '#',
          release_year: rec.release_year,
          memory_gb: rec.memory_gb,
          storage_gb: rec.storage_gb,
          os_type: rec.os_type,
          ssd_required: rec.ssd_required,
          score: rec.hybrid_score || rec.similarity || rec.score || 0
        };
      });
      
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
    if (tag && tag.trim() && !preferences.preferred_tags.includes(tag.trim())) {
      setPreferences(prev => ({
        ...prev,
        preferred_tags: [...prev.preferred_tags, tag.trim()]
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
    if (item && item.trim() && !preferences[list].includes(item.trim())) {
      setPreferences(prev => ({
        ...prev,
        [list]: [...prev[list], item.trim()]
      }));
    }
  };

  const removeItem = (list, item) => {
    setPreferences(prev => ({
      ...prev,
      [list]: prev[list].filter(i => i !== item)
    }));
  };

  // Handle custom tag input
  const handleCustomTagSubmit = (e) => {
    e.preventDefault();
    if (customTag.trim()) {
      addTag(customTag);
      setCustomTag('');
    }
  };

  // Handle custom developer input
  const handleCustomDeveloperSubmit = (e) => {
    e.preventDefault();
    if (customDeveloper.trim()) {
      addItem('developers', customDeveloper);
      setCustomDeveloper('');
    }
  };

  // Handle custom publisher input
  const handleCustomPublisherSubmit = (e) => {
    e.preventDefault();
    if (customPublisher.trim()) {
      addItem('publishers', customPublisher);
      setCustomPublisher('');
    }
  };

  // Helper function to get accurate game type
  const getAccurateGameType = (game) => {
    if (!game) return 'paid';
    
    const discounted_price = game.discounted_price || game.price || 0;
    const discount_percentage = game.discount_percentage || game.discount || 0;
    
    if (discounted_price === 0) {
      return 'free';
    } else if (discount_percentage > 0) {
      return 'discount';
    } else {
      return 'paid';
    }
  };

  const renderGameCard = (game, index, showScore = false) => {
    if (!game) return null;
    
    const posterUrl = getGamePosterUrl(game);
    const liked = isGameLiked(game);
    const gameTypeAccurate = getAccurateGameType(game);
    
    const gameData = {
      title: game.title || 'Unknown Game',
      developer: game.developer || 'Unknown',
      publisher: game.publisher || 'Unknown',
      discounted_price: game.discounted_price || game.price || 0,
      original_price: game.original_price || game.discounted_price || game.price || 0,
      discount_percentage: game.discount_percentage || game.discount || 0,
      overall_sentiment_score: game.overall_sentiment_score || game.sentiment || 0.5,
      all_reviews_count: game.all_reviews_count || game.reviews || 0,
      tags: game.tags || [],
      categories: game.categories || [],
      features: game.features || [],
      languages: game.languages || [],
      link: game.link || '#',
      release_year: game.release_year,
      memory_gb: game.memory_gb,
      storage_gb: game.storage_gb,
      os_type: game.os_type,
      ssd_required: game.ssd_required,
      score: game.score || game.similarity || game.hybrid_score || 0
    };
    
    const score = showScore ? gameData.score : 0;
    const hasDiscount = gameTypeAccurate === 'discount' && gameData.discount_percentage > 0;
    
    return (
      <div key={index} className="game-card">
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
          
          {hasDiscount && gameData.discount_percentage > 0 && (
            <div className="discount-badge">-{Math.round(gameData.discount_percentage)}%</div>
          )}
          
          {gameTypeAccurate === 'free' && (
            <div className="free-badge">FREE</div>
          )}
          
          <div className={`game-type-badge ${gameTypeAccurate}`}>
            {gameTypeAccurate === 'free' ? 'Free' : 
             gameTypeAccurate === 'discount' ? 'Discount' : 
             'Paid'}
          </div>
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
            {hasDiscount && gameData.original_price > gameData.discounted_price && (
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
          
          <div className="game-actions">
            <button 
              onClick={(e) => {
                e.stopPropagation();
                viewGameDetails(game);
              }}
              className="view-details-btn"
              title={`View ${gameData.title} details`}
            >
              View Details
            </button>
            
            {gameData.link && gameData.link !== '#' && (
              <a 
                href={gameData.link} 
                target="_blank" 
                rel="noopener noreferrer"
                className="view-steam-btn"
                onClick={(e) => e.stopPropagation()}
                title={`View ${gameData.title} on Steam`}
              >
                View on Steam →
              </a>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Render Game Details Modal
  const renderGameDetailsModal = () => {
    if (!selectedGame) return null;
    
    const gameImages = getSteamImages(selectedGame);
    const gameTypeAccurate = getAccurateGameType(selectedGame);
    const hasDiscount = gameTypeAccurate === 'discount' && selectedGame.discount_percentage > 0;
    
    return (
      <div className={`game-details-modal ${showGameDetails ? 'show' : ''}`}>
        <div className="modal-backdrop" onClick={closeGameDetails}></div>
        
        <div className="modal-content">
          <div className="modal-header">
            <h2>{selectedGame.title}</h2>
            <button className="close-modal-btn" onClick={closeGameDetails}>✕</button>
          </div>
          
          <div className="modal-body">
            <div className="game-images">
              <img 
                src={gameImages.header || getPlaceholderImage(selectedGame.title)}
                alt={selectedGame.title}
                className="main-game-image"
                onError={(e) => { 
                  e.target.onerror = null; 
                  e.target.src = getPlaceholderImage(selectedGame.title); 
                }}
              />
              
              <div className="game-badges">
                {gameTypeAccurate === 'free' && (
                  <span className="free-badge">FREE</span>
                )}
                {hasDiscount && (
                  <span className="discount-badge-large">-{Math.round(selectedGame.discount_percentage || 0)}% OFF</span>
                )}
              </div>
            </div>
            
            <div className="game-info-grid">
              <div className="game-basic-info">
                <div className="price-section">
                  <div className="current-price">
                    {selectedGame.discounted_price === 0 ? 'FREE' : `$${selectedGame.discounted_price?.toFixed(2) || '0.00'}`}
                  </div>
                  {hasDiscount && selectedGame.original_price > selectedGame.discounted_price && (
                    <div className="original-price-large">
                      ${selectedGame.original_price?.toFixed(2) || '0.00'}
                    </div>
                  )}
                </div>
                
                <div className="game-stats-detailed">
                  <div className="stat-item">
                    <span className="stat-label">Rating:</span>
                    <span className="stat-value rating">
                      ⭐ {Math.round((selectedGame.overall_sentiment_score || 0.5) * 100)}%
                    </span>
                  </div>
                  
                  <div className="stat-item">
                    <span className="stat-label">Reviews:</span>
                    <span className="stat-value">
                      {(selectedGame.all_reviews_count || 0).toLocaleString()}
                    </span>
                  </div>
                  
                  <div className="stat-item">
                    <span className="stat-label">Release Year:</span>
                    <span className="stat-value">
                      {selectedGame.release_year || 'N/A'}
                    </span>
                  </div>
                  
                  <div className="stat-item">
                    <span className="stat-label">Developer:</span>
                    <span className="stat-value">{selectedGame.developer || 'Unknown'}</span>
                  </div>
                  
                  <div className="stat-item">
                    <span className="stat-label">Publisher:</span>
                    <span className="stat-value">{selectedGame.publisher || 'Unknown'}</span>
                  </div>
                </div>
              </div>
              
              <div className="game-details-section">
                {selectedGame.tags && selectedGame.tags.length > 0 && (
                  <div className="details-group">
                    <h3>Tags</h3>
                    <div className="tags-list">
                      {selectedGame.tags.map((tag, index) => (
                        <span key={index} className="tag-large">{tag}</span>
                      ))}
                    </div>
                  </div>
                )}
                
                {selectedGame.features && selectedGame.features.length > 0 && (
                  <div className="details-group">
                    <h3>Features</h3>
                    <div className="features-list">
                      {selectedGame.features.map((feature, index) => (
                        <span key={index} className="feature-item">✓ {feature}</span>
                      ))}
                    </div>
                  </div>
                )}
                
                {selectedGame.languages && selectedGame.languages.length > 0 && (
                  <div className="details-group">
                    <h3>Languages</h3>
                    <div className="languages-list">
                      {selectedGame.languages.map((lang, index) => (
                        <span key={index} className="language-item">{lang}</span>
                      ))}
                    </div>
                  </div>
                )}
                
                <div className="details-group">
                  <h3>System Requirements</h3>
                  <div className="system-requirements">
                    {selectedGame.memory_gb && (
                      <div className="requirement">
                        <span className="req-label">Memory:</span>
                        <span className="req-value">{selectedGame.memory_gb} GB</span>
                      </div>
                    )}
                    
                    {selectedGame.storage_gb && (
                      <div className="requirement">
                        <span className="req-label">Storage:</span>
                        <span className="req-value">{selectedGame.storage_gb} GB</span>
                      </div>
                    )}
                    
                    {selectedGame.os_type && (
                      <div className="requirement">
                        <span className="req-label">OS:</span>
                        <span className="req-value">{selectedGame.os_type}</span>
                      </div>
                    )}
                    
                    {selectedGame.ssd_required && (
                      <div className="requirement">
                        <span className="req-label">SSD:</span>
                        <span className="req-value">Required</span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="game-actions-large">
                  <button 
                    className={`like-btn-large ${isGameLiked(selectedGame) ? 'liked' : ''}`}
                    onClick={() => toggleLike(selectedGame)}
                  >
                    {isGameLiked(selectedGame) ? '❤️ Liked' : '🤍 Like this Game'}
                  </button>
                  
                  {selectedGame.link && selectedGame.link !== '#' && (
                    <a 
                      href={selectedGame.link} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="steam-btn"
                    >
                      View on Steam →
                    </a>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Render pagination controls
  const renderPagination = () => {
    if (totalPages <= 1) return null;
    
    const pageNumbers = [];
    const maxVisiblePages = 5;
    
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    
    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
      pageNumbers.push(i);
    }
    
    return (
      <div className="pagination">
        <button 
          onClick={() => loadGames(currentPage - 1, gameType)}
          disabled={currentPage === 1}
          className="page-btn prev-btn"
        >
          « Previous
        </button>
        
        {startPage > 1 && (
          <>
            <button 
              onClick={() => loadGames(1, gameType)}
              className={`page-btn ${currentPage === 1 ? 'active' : ''}`}
            >
              1
            </button>
            {startPage > 2 && <span className="page-dots">...</span>}
          </>
        )}
        
        {pageNumbers.map(page => (
          <button
            key={page}
            onClick={() => loadGames(page, gameType)}
            className={`page-btn ${currentPage === page ? 'active' : ''}`}
          >
            {page}
          </button>
        ))}
        
        {endPage < totalPages && (
          <>
            {endPage < totalPages - 1 && <span className="page-dots">...</span>}
            <button 
              onClick={() => loadGames(totalPages, gameType)}
              className={`page-btn ${currentPage === totalPages ? 'active' : ''}`}
            >
              {totalPages}
            </button>
          </>
        )}
        
        <button 
          onClick={() => loadGames(currentPage + 1, gameType)}
          disabled={currentPage === totalPages}
          className="page-btn next-btn"
        >
          Next »
        </button>
      </div>
    );
  };

  // Initial loading screen
  if (initialLoading) {
    return (
      <div className="initial-loading-screen">
        <div className="loading-content">
          <div className="loading-logo">
            <h1>SteamMind</h1>
            <p>Your Ultimate Game Discovery Platform</p>
          </div>
          <div className="loading-spinner">
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
          </div>
          <div className="loading-text">
            Loading your gaming universe...
          </div>
          <div className="loading-progress">
            <div className="progress-bar">
              <div className="progress-fill"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
            <h1>SteamMind</h1>
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
                
                <div className="game-type-selector">
                  <button 
                    className={`game-type-btn ${gameType === 'all' ? 'active' : ''}`}
                    onClick={() => handleGameTypeChange('all')}
                  >
                    All Games ({totalGames.toLocaleString()})
                  </button>
                  <button 
                    className={`game-type-btn ${gameType === 'free' ? 'active' : ''}`}
                    onClick={() => handleGameTypeChange('free')}
                  >
                    Free
                  </button>
                  <button 
                    className={`game-type-btn ${gameType === 'discount' ? 'active' : ''}`}
                    onClick={() => handleGameTypeChange('discount')}
                  >
                    Discount
                  </button>
                  <button 
                    className={`game-type-btn ${gameType === 'paid' ? 'active' : ''}`}
                    onClick={() => handleGameTypeChange('paid')}
                  >
                    Paid
                  </button>
                </div>
                
                <select value={sortBy} onChange={handleSortChange} className="sort-select">
                  <option value="popularity">Sort by Popularity</option>
                  <option value="alphabetical">Sort A-Z</option>
                  <option value="price-low">Price: Low to High</option>
                  <option value="price-high">Price: High to Low</option>
                  <option value="rating">Rating</option>
                </select>
              </div>
            </div>
            
            <div className="filter-info">
              <p>
                Showing <strong>{games.length}</strong> of <strong>{totalGames.toLocaleString()}</strong> games
                {gameType !== 'all' && ` (Filtered: ${gameType})`}
                {currentPage > 1 && ` • Page ${currentPage} of ${totalPages}`}
              </p>
            </div>
            
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Loading games...</p>
              </div>
            ) : (
              <>
                <div className="games-grid">
                  {handleSort(games).map((game, index) => renderGameCard(game, index))}
                </div>
                
                {games.length === 0 && (
                  <div className="empty-state">
                    <div className="empty-icon">🎮</div>
                    <h3>No games found</h3>
                    <p>Try a different search or game type filter</p>
                  </div>
                )}
                
                {renderPagination()}
              </>
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
                <button className={`method-btn ${contentMethod === 'jaccard' ? 'active' : ''}`} onClick={() => setContentMethod('jaccard')}>Jaccard</button>
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
                    <h3>Highly Similar</h3>
                    <div className="games-grid">
                      {recommendations.data.results.highly_similar.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.moderately_similar?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3>Moderately Similar</h3>
                    <div className="games-grid">
                      {recommendations.data.results.moderately_similar.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.somewhat_similar?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3>Somewhat Similar </h3>
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
                  <div className="input-with-button">
                    <form onSubmit={handleCustomTagSubmit} className="custom-input-form">
                      <input
                        type="text"
                        placeholder="Add custom tag..."
                        value={customTag}
                        onChange={(e) => setCustomTag(e.target.value)}
                        className="custom-input"
                      />
                      <button type="submit" className="add-btn">+</button>
                    </form>
                    <select 
                      onChange={(e) => { 
                        if (e.target.value) {
                          addTag(e.target.value); 
                          e.target.value = ''; 
                        }
                      }}
                      className="dropdown-select"
                    >
                      <option value="">Or choose from list...</option>
                      {availableTags.map(tag => (
                        <option key={tag} value={tag}>{tag}</option>
                      ))}
                    </select>
                  </div>
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
                  <select 
                  onChange={(e) => { if (e.target.value) { addItem('languages', e.target.value); e.target.value = ''; } }}
                  className="dropdown-select"  // ADD THIS CLASS
                >
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
                  <div className="input-with-button">
                    <form onSubmit={handleCustomDeveloperSubmit} className="custom-input-form">
                      <input
                        type="text"
                        placeholder="Add custom developer..."
                        value={customDeveloper}
                        onChange={(e) => setCustomDeveloper(e.target.value)}
                        className="custom-input"
                      />
                      <button type="submit" className="add-btn">+</button>
                    </form>
                    <select 
                      onChange={(e) => { 
                        if (e.target.value) {
                          addItem('developers', e.target.value); 
                          e.target.value = ''; 
                        }
                      }}
                      className="dropdown-select"
                    >
                      <option value="">Or choose from list...</option>
                      {availableDevelopers.map(dev => (
                        <option key={dev} value={dev}>{dev}</option>
                      ))}
                    </select>
                  </div>
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
                  <div className="input-with-button">
                    <form onSubmit={handleCustomPublisherSubmit} className="custom-input-form">
                      <input
                        type="text"
                        placeholder="Add custom publisher..."
                        value={customPublisher}
                        onChange={(e) => setCustomPublisher(e.target.value)}
                        className="custom-input"
                      />
                      <button type="submit" className="add-btn">+</button>
                    </form>
                    <select 
                      onChange={(e) => { 
                        if (e.target.value) {
                          addItem('publishers', e.target.value); 
                          e.target.value = ''; 
                        }
                      }}
                      className="dropdown-select"
                    >
                      <option value="">Or choose from list...</option>
                      {availablePublishers.map(pub => (
                        <option key={pub} value={pub}>{pub}</option>
                      ))}
                    </select>
                  </div>
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
                
                {/* <div className="filter-group">
                  <label>Min Sentiment: {Math.round(preferences.min_sentiment * 100)}%</label>
                  <input 
                    type="range" 
                    min="0" 
                    max="100" 
                    value={preferences.min_sentiment * 100}
                    onChange={(e) => setPreferences(prev => ({ ...prev, min_sentiment: parseInt(e.target.value) / 100 }))}
                    className="slider" 
                  />
                </div> */}
{/*                 
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
                </div> */}
              </div>
              
              <button onClick={getConstraintRecommendations} className="btn-primary">
                Get Constraint-Based Recommendations
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
                    <h3> Best Matches </h3>
                    <div className="games-grid">
                      {recommendations.data.results.perfect_matches.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.good_matches?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3> What you make like  </h3>
                    <div className="games-grid">
                      {recommendations.data.results.good_matches.games.map((game, index) => 
                        renderGameCard(game, index, true)
                      )}
                    </div>
                  </div>
                )}
                
                {recommendations.data.results.partial_matches?.games?.length > 0 && (
                  <div className="recommendation-category">
                    <h3>Partial Matches</h3>
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
                    <div className="empty-icon"> </div>
                    <h3>No matches found</h3>
                    <p>Try relaxing your constraints or select different preferences.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon"> </div>
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
                 <button className={`method-btn ${hybridMethod === 'jaccard' ? 'active' : ''}`} onClick={() => setHybridMethod('jaccard')}>Jaccard</button>
              </div>
            </div>
            
            <button onClick={getHybridRecommendations} className="btn-primary" disabled={likedGames.length === 0}>
              Get Hybrid Recommendations
            </button>
            
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Getting recommendations...</p>
              </div>
            ) : recommendations && recommendations.method === 'hybrid' ? (
              <div className="recommendation-category">
                <h3>Hybrid Recommendations</h3>
                <div className="games-grid">
                  {recommendations.data.recommendations.map((game, index) => 
                    renderGameCard(game, index, true)
                  )}
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon"></div>
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

      {/* Game Details Modal */}
      {renderGameDetailsModal()}
    </div>
  );
}

export default App;