import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class ProductSemanticAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.sia = SentimentIntensityAnalyzer()
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def analyze_keywords(self):
        """Analyze keyword frequency and patterns"""
        all_keywords = []
        for keywords in self.df['keywords'].dropna():
            all_keywords.extend(keywords.split(','))
        
        keyword_counts = Counter(all_keywords)
        return keyword_counts
    
    def perform_tfidf_analysis(self):
        """Perform TF-IDF analysis on product descriptions"""
        descriptions = self.df['description'].apply(self.preprocess_text)
        tfidf_matrix = self.vectorizer.fit_transform(descriptions)
        feature_names = self.vectorizer.get_feature_names_out()
        
        return tfidf_matrix, feature_names
    
    def calculate_similarity_matrix(self):
        """Calculate similarity between different product aspects"""
        tfidf_matrix, feature_names = self.perform_tfidf_analysis()
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix, feature_names
    
    def sentiment_analysis(self):
        """Perform sentiment analysis on product descriptions"""
        sentiments = []
        for description in self.df['description']:
            if pd.notna(description):
                sentiment_scores = self.sia.polarity_scores(description)
                sentiments.append(sentiment_scores)
            else:
                sentiments.append({'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0})
        
        return sentiments
    
    def topic_modeling(self, n_topics=3):
        """Perform topic modeling using LDA"""
        tfidf_matrix, feature_names = self.perform_tfidf_analysis()
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf_matrix)
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            topics.append(top_words)
        
        return topics
    
    def generate_wordcloud(self):
        """Generate word cloud from keywords"""
        all_keywords = ' '.join(self.df['keywords'].dropna().str.replace(',', ' '))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)
        return wordcloud
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Keyword frequency
        keyword_counts = self.analyze_keywords()
        top_keywords = dict(keyword_counts.most_common(10))
        axes[0, 0].bar(top_keywords.keys(), top_keywords.values())
        axes[0, 0].set_title('Top 10 Keywords')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Sentiment distribution
        sentiments = self.sentiment_analysis()
        compound_scores = [s['compound'] for s in sentiments]
        axes[0, 1].hist(compound_scores, bins=10, alpha=0.7)
        axes[0, 1].set_title('Sentiment Distribution')
        axes[0, 1].set_xlabel('Compound Sentiment Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Style analysis
        style_counts = self.df['style'].value_counts()
        axes[1, 0].pie(style_counts.values, labels=style_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Product Style Distribution')
        
        # 4. Season analysis
        season_counts = self.df['season'].value_counts()
        axes[1, 1].bar(season_counts.index, season_counts.values)
        axes[1, 1].set_title('Season Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('semantic_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("=" * 60)
        print("PRODUCT SEMANTIC ANALYSIS REPORT")
        print("=" * 60)
        
        # Basic statistics
        print(f"\n1. DATASET OVERVIEW:")
        print(f"   Total products analyzed: {len(self.df)}")
        print(f"   Product ID: {self.df['product_id'].iloc[0]}")
        print(f"   Product Name: {self.df['product_name'].iloc[0]}")
        
        # Keyword analysis
        print(f"\n2. KEYWORD ANALYSIS:")
        keyword_counts = self.analyze_keywords()
        print("   Top 10 most frequent keywords:")
        for keyword, count in keyword_counts.most_common(10):
            print(f"   - {keyword.strip()}: {count} occurrences")
        
        # Sentiment analysis
        print(f"\n3. SENTIMENT ANALYSIS:")
        sentiments = self.sentiment_analysis()
        avg_compound = np.mean([s['compound'] for s in sentiments])
        print(f"   Average sentiment score: {avg_compound:.3f}")
        print(f"   Sentiment interpretation: {'Positive' if avg_compound > 0 else 'Negative' if avg_compound < 0 else 'Neutral'}")
        
        # Topic modeling
        print(f"\n4. TOPIC MODELING:")
        topics = self.topic_modeling()
        for i, topic in enumerate(topics):
            print(f"   Topic {i+1}: {', '.join(topic)}")
        
        # Style and category analysis
        print(f"\n5. PRODUCT CHARACTERISTICS:")
        print(f"   Category: {self.df['category'].iloc[0]}")
        print(f"   Style: {self.df['style'].iloc[0]}")
        print(f"   Color: {self.df['color'].iloc[0]}")
        print(f"   Neckline: {self.df['neckline'].iloc[0]}")
        print(f"   Sleeve Type: {self.df['sleeve_type'].iloc[0]}")
        print(f"   Fit Type: {self.df['fit_type'].iloc[0]}")
        print(f"   Season: {self.df['season'].iloc[0]}")
        print(f"   Price Range: {self.df['price_range'].iloc[0]}")
        
        # Similarity analysis
        print(f"\n6. SIMILARITY ANALYSIS:")
        similarity_matrix, _ = self.calculate_similarity_matrix()
        avg_similarity = np.mean(similarity_matrix)
        print(f"   Average similarity between descriptions: {avg_similarity:.3f}")
        
        print("\n" + "=" * 60)

def main():
    # Initialize analyzer
    analyzer = ProductSemanticAnalyzer('product_dataset.csv')
    
    # Generate comprehensive report
    analyzer.generate_report()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Generate word cloud
    wordcloud = analyzer.generate_wordcloud()
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Product Keywords Word Cloud')
    plt.savefig('product_keywords_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete! Check the generated visualizations:")
    print("- semantic_analysis_results.png")
    print("- product_keywords_wordcloud.png")

if __name__ == "__main__":
    main() 