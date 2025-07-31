# Product Semantic Analysis Project

## Overview
This project performs comprehensive semantic analysis on the Shein product: **Spring-Summer French Minimalist Style White Camisole Dress** (Product ID: 81105517).

## Product Information
- **Product Link**: [Shein Camisole Dress](https://euqs.shein.com/Spring-Summer-French-Minimalist-Style-White-Camisole-Dress-Women-High-End-Sense-Slimming-Square-Neck-Waisted-Sleeveless-Backless-Vintage-Fitted-Sexy-Slip-Dress-p-81105517.html)
- **Product Name**: Spring-Summer French Minimalist Style White Camisole Dress
- **Category**: Women's Clothing
- **Style**: Camisole Dress
- **Color**: White
- **Price Range**: High-End

## Analysis Results

### 1. Keyword Analysis
The most frequent keywords identified in the product descriptions:
- **minimalist** (3 occurrences)
- **vintage** (3 occurrences) 
- **elegant** (3 occurrences)
- **french** (2 occurrences)
- **high-end** (2 occurrences)
- **slimming** (2 occurrences)
- **waisted** (2 occurrences)
- **backless** (2 occurrences)
- **fitted** (2 occurrences)
- **sexy** (2 occurrences)

### 2. Sentiment Analysis
- **Average Sentiment Score**: 0.324
- **Sentiment Interpretation**: Positive
- The product descriptions convey a positive emotional tone, emphasizing luxury, elegance, and sophistication.

### 3. Topic Modeling
Three main topics were identified:
1. **Topic 1**: Elegant, slip, dress, waisted, style, design, slimming, fitted, flattering, silhouette
2. **Topic 2**: Definition, enhanced, figure, spring, occasions, perfect, summer, vintage-inspired, elements, design
3. **Topic 3**: Premium, quality, sophisticated, vintage, backless, sexy, sense, high-end, minimalist, look

### 4. Product Characteristics
- **Neckline**: Square Neck
- **Sleeve Type**: Sleeveless
- **Fit Type**: Fitted
- **Season**: Spring-Summer
- **Material**: Silk/Satin (inferred)

### 5. Similarity Analysis
- **Average Similarity**: 0.150 between product descriptions
- This indicates moderate similarity across different aspects of the product descriptions.

## Files Generated

### Data Files
- `product_dataset.csv` - Contains structured product data with 10 different aspects/descriptions
- `semantic_analysis.py` - Main analysis script

### Visualizations
- `semantic_analysis_results.png` - Comprehensive analysis charts including:
  - Keyword frequency bar chart
  - Sentiment distribution histogram
  - Product style distribution pie chart
  - Season distribution bar chart
- `product_keywords_wordcloud.png` - Word cloud visualization of product keywords

### Dependencies
- `requirements.txt` - Python package dependencies
- `venv/` - Virtual environment for isolated package management

## Technical Implementation

### Libraries Used
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms (TF-IDF, LDA, Cosine Similarity)
- **nltk** - Natural language processing
- **matplotlib & seaborn** - Data visualization
- **wordcloud** - Word cloud generation

### Analysis Techniques
1. **TF-IDF Vectorization** - Text feature extraction
2. **Cosine Similarity** - Measuring text similarity
3. **Latent Dirichlet Allocation (LDA)** - Topic modeling
4. **VADER Sentiment Analysis** - Sentiment scoring
5. **Keyword Frequency Analysis** - Text pattern identification

## Setup and Usage

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Activate virtual environment
source venv/bin/activate

# Run semantic analysis
python semantic_analysis.py
```

## Key Insights

### Marketing Implications
1. **Luxury Positioning**: The product is clearly positioned as high-end with premium quality
2. **Seasonal Appeal**: Strong spring-summer seasonal messaging
3. **Style Keywords**: Emphasis on "minimalist", "vintage", and "elegant" appeals to sophisticated consumers
4. **Fit Focus**: Multiple mentions of "slimming", "fitted", and "flattering" indicate body-conscious design

### Consumer Appeal
- **Target Demographics**: Fashion-conscious women seeking elegant, vintage-inspired pieces
- **Price Sensitivity**: High-end positioning suggests premium pricing strategy
- **Occasion Usage**: Versatile for spring-summer occasions
- **Style Preferences**: Appeals to minimalist and vintage fashion enthusiasts

## Future Enhancements
- Expand dataset with customer reviews
- Include competitor analysis
- Add price sensitivity analysis
- Implement real-time sentiment monitoring
- Create recommendation system based on semantic similarity

## Project Structure
```
practiceproject1.py/
├── project.py                 # Basic Python project file
├── semantic_analysis.py       # Main analysis script
├── product_dataset.csv        # Product data
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── venv/                      # Virtual environment
├── semantic_analysis_results.png    # Analysis visualizations
└── product_keywords_wordcloud.png   # Keyword word cloud
```

## Contact
For questions or contributions, please refer to the project documentation and code comments. 