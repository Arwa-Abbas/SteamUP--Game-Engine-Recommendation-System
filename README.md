# SteamMind – Game Engine Recommendation System #

SteamMind is an advanced game recommendation platform that leverages both **knowledge-based (constraint-driven)** and **content-based filtering** techniques to provide personalized game suggestions. Built with **FastAPI** and **MongoDB**, the system combines sophisticated similarity metrics, user preferences, and game metadata to deliver accurate and explainable recommendations for gamers.

---

**Dataset Used:** [Steam Games Dataset](https://www.kaggle.com/datasets/nikatomashvili/steam-games-dataset)


## 📋 Features

### 🧠 Knowledge-Based Recommendations
- Constraint-based filtering (hard & soft constraints)
- Price range filtering (min/max)
- System requirements matching (RAM, storage, OS)
- Language preferences
- Developer/publisher preferences
- Sentiment score filtering
- Tag-based matching
- Score-based ranking with explanations

### 🎯 Content-Based Recommendations
Four similarity metrics:
- **Cosine Similarity** – Angle between feature vectors
- **Pearson Correlation** – Linear correlation coefficient
- **Euclidean Distance** – Inverse distance with Gaussian kernel
- **Jaccard Similarity** – Set overlap between binary features


### 🤝 Hybrid Recommendations
- Intelligent combination of constraint + content-based systems
- Dynamic weighting based on match quality

---

## System Architecture

```bash
Game Recommendation System
 ├──backend
   📁 src/
   │ ├── recommender.py
   │ ├── db.py
   │ └── data_loader.py
   ├── 📁 models/
   ├── 📁 dataset/
   ├── main.py
   ├── requirements.txt
   └── config.py
   └── .env

```


## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Arwa-Abbas/SteamMind--Game-Engine-Recommendation-System.git
cd SteamMind--Game-Engine-Recommendation-System
cd backend
```
### 2. Create virtual environment in backend
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
###3. Load Dataset 
```bash
python -m src.loader
```
### 4. Train the Model
 ```bash
from src.recommender import GameRecommender
from src.db import db
recommender = GameRecommender(db)
recommender.train_models
```
### 5. Run the Server
```bash
python main.py
```
API will be live at:
```bash
http://localhost:8000
```
### 6. Run frontend
```bash
npm start
```

---

## Game Data Structure

```bash
{
  "title": "Game Name",
  "developer": "Developer Name",
  "publisher": "Publisher Name",
  "discounted_price": 29.99,
  "original_price": 59.99,
  "discount_percentage": 50.0,
  "overall_sentiment_score": 0.85,
  "all_reviews_count": 100000,
  "popularity_score": 0.92,
  "tags": ["action", "rpg"],
  "languages": ["english", "french"],
  "features": ["single-player", "cloud saves"],
  "categories": ["rpg", "action"],
  "memory_gb": 8,
  "storage_gb": 70,
  "os_type": "windows",
  "ssd_required": true,
  "link": "https://store.steampowered.com/app/...",
  "release_year": 2020
}

```

---

## Similarity Methods

### Cosine Similarity
Measures angle between vectors (0 → 1).
### Pearson Correlation
Measures linear correlation (-1 → 1).
### Euclidean Similarity
Gaussian-based inverse distance.
### Jaccard Similarity
Set overlap for binary features.

### Memory Efficiency

Sparse TF-IDF (max 800 features)
Top-K similarity storage
Chunked processing
Float16 / UInt16 compression
Fallback mechanisms for missing data

---

