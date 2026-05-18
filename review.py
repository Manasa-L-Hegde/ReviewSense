"""
Restaurant Reviews Sentiment Analysis
======================================
A machine learning project to classify restaurant reviews as positive or negative
using TF-IDF vectorization and Naive Bayes classification.

Author: Data Science Team
Version: 1.0
"""

import logging
import string
import os
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==================== Configuration ====================
CONFIG = {
    'data_file': 'Restaurant_Reviews.tsv',
    'delimiter': '\t',
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 1500,
    'sentiment_mapping': {1: 'Positive', 0: 'Negative'}
}

# ==================== Logging Setup ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Helper Functions ====================
def download_nltk_resources() -> None:
    """
    Download required NLTK resources (stopwords).
    Handles errors gracefully if resources are already available.
    """
    try:
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK stopwords already available")
    except LookupError:
        logger.info("Downloading NLTK stopwords...")
        try:
            nltk.download('stopwords', quiet=True)
            logger.info("Successfully downloaded NLTK stopwords")
        except Exception as e:
            logger.error(f"Failed to download NLTK resources: {e}")
            raise


def load_data(filepath: str, delimiter: str = '\t') -> Optional[pd.DataFrame]:
    """
    Load restaurant reviews data from TSV file.
    
    Args:
        filepath: Path to the data file
        delimiter: Delimiter used in the file
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Data file not found: {filepath}")
            return None
            
        df = pd.read_csv(filepath, delimiter=delimiter)
        logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate required columns
        if 'Review' not in df.columns or 'Liked' not in df.columns:
            logger.error("Required columns 'Review' and 'Liked' not found in data")
            return None
            
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Processed DataFrame
    """
    try:
        # Add sentiment labels
        df['sentiment'] = df['Liked'].map(CONFIG['sentiment_mapping'])
        
        # Rename columns for consistency
        df.rename(columns={'Review': 'review'}, inplace=True)
        
        # Select relevant columns
        df = df[['review', 'sentiment']].copy()
        
        # Remove rows with missing values
        df = df.dropna()
        
        logger.info(f"Data prepared: {df.shape[0]} rows remaining after cleaning")
        return df
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise


def plot_sentiment_distribution(df: pd.DataFrame, ax) -> None:
    """
    Plot sentiment distribution bar chart.
    
    Args:
        df: DataFrame containing sentiment column
        ax: Matplotlib axis object
    """
    try:
        df['sentiment'].value_counts().plot(kind='bar', color=['#ff6b6b', '#4ecdc4'], ax=ax)
        ax.set_title("Sentiment Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sentiment", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        logger.info("Sentiment distribution plot prepared")
    except Exception as e:
        logger.error(f"Error plotting sentiment distribution: {e}")


def preprocess_text(text: str, stemmer: PorterStemmer, stop_words: set) -> str:
    """
    Clean and preprocess review text.
    
    Args:
        text: Raw review text
        stemmer: PorterStemmer instance
        stop_words: Set of English stopwords
        
    Returns:
        Cleaned and preprocessed text
    """
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into words and filter
        words = text.split()
        words = [stemmer.stem(w) for w in words if w and w not in stop_words]
        
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return ""


def build_vectorizer_and_encode(df: pd.DataFrame) -> Tuple[TfidfVectorizer, object, object]:
    """
    Build TF-IDF vectorizer and encode sentiments.
    
    Args:
        df: DataFrame with 'cleaned_review' and 'sentiment' columns
        
    Returns:
        Tuple of (vectorizer, X, y)
    """
    try:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=CONFIG['max_features'])
        X = vectorizer.fit_transform(df['cleaned_review'])
        logger.info(f"TF-IDF vectorization complete: {X.shape[1]} features")
        
        # Encode sentiments
        le = LabelEncoder()
        y = le.fit_transform(df['sentiment'])
        logger.info(f"Sentiment classes: {le.classes_}")
        
        return vectorizer, X, y, le
    except Exception as e:
        logger.error(f"Error in vectorization/encoding: {e}")
        raise


def train_model(X_train: object, y_train: object) -> MultinomialNB:
    """
    Train Naive Bayes classification model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    try:
        model = MultinomialNB()
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


def evaluate_model(model: MultinomialNB, X_test: object, y_test: object, 
                  le: LabelEncoder) -> Tuple[object, float, str]:
    """
    Evaluate model performance and display results.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        le: LabelEncoder instance
        
    Returns:
        Tuple of (y_pred, accuracy, classification_report_str)
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                       target_names=le.classes_)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        return y_pred, accuracy, report
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


def plot_confusion_matrix(y_test: object, y_pred: object, le: LabelEncoder, ax) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        le: LabelEncoder instance
        ax: Matplotlib axis object
    """
    try:
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_,
                    yticklabels=le.classes_,
                    cbar_kws={'label': 'Count'},
                    ax=ax)
        
        ax.set_title("Confusion Matrix", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        logger.info("Confusion matrix plot prepared")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")


# ==================== Main Execution ====================
def main() -> None:
    """
    Main execution function for the sentiment analysis pipeline.
    """
    logger.info("Starting Restaurant Reviews Sentiment Analysis")
    
    try:
        # Download NLTK resources
        download_nltk_resources()
        
        # Load data
        logger.info(f"Loading data from {CONFIG['data_file']}")
        df = load_data(CONFIG['data_file'], CONFIG['delimiter'])
        if df is None:
            logger.error("Failed to load data. Exiting.")
            return
        
        # Prepare data
        df = prepare_data(df)
        print(f"\n{'='*80}")
        print(f"RESTAURANT REVIEWS SENTIMENT ANALYSIS - REPORT")
        print(f"{'='*80}\n")
        print(f"Dataset Overview:")
        print(f"  - Total Reviews: {df.shape[0]}")
        print(f"  - Columns: {df.shape[1]}\n")
        print(f"Sample Reviews:")
        print(f"{'-'*80}")
        print(df.head().to_string(index=False))
        
        # Preprocess text
        logger.info("Preprocessing reviews...")
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        df['cleaned_review'] = df['review'].apply(
            lambda x: preprocess_text(x, stemmer, stop_words)
        )
        logger.info("Text preprocessing completed")
        
        # Vectorize and encode
        vectorizer, X, y, le = build_vectorizer_and_encode(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state']
        )
        logger.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        y_pred, accuracy, report = evaluate_model(model, X_test, y_test, le)

        # Create a focused dashboard with two clear plots (bar + heatmap)
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle('Restaurant Reviews Sentiment Analysis - Summary', 
                     fontsize=16, fontweight='bold', y=0.98)

        # layout: single row, two columns
        gs = fig.add_gridspec(1, 2, wspace=0.35, left=0.06, right=0.98, top=0.88, bottom=0.12)

        # Left: Sentiment Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        plot_sentiment_distribution(df, ax1)

        # Right: Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        plot_confusion_matrix(y_test, y_pred, le, ax2)

        # Save figure for presentation use
        out_path = 'sentiment_summary.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        # Also save the two plots individually for easier reuse
        try:
            # Sentiment distribution alone
            fig_dist, ax_dist = plt.subplots(figsize=(7, 6))
            plot_sentiment_distribution(df, ax_dist)
            fig_dist.savefig('sentiment_distribution.png', dpi=150, bbox_inches='tight')
            plt.close(fig_dist)

            # Confusion matrix alone
            fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
            plot_confusion_matrix(y_test, y_pred, le, ax_cm)
            fig_cm.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close(fig_cm)
        except Exception as e:
            logger.error(f"Failed to save individual plots: {e}")
        plt.show()

        # Print classification report to console
        print('\n' + '='*60)
        print('CLASSIFICATION REPORT')
        print('='*60)
        print(report)
        print('='*60 + '\n')

        # Prepare sentiment counts for final summary
        sentiment_counts = df['sentiment'].value_counts()
        
        # Print final summary
        accuracy_percent = accuracy * 100
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE ✓")
        print(f"{'='*80}")
        print(f"\nFinal Results:")
        print(f"  - Model Accuracy: {accuracy:.4f} ({accuracy_percent:.2f}%)")
        print(f"  - Total Reviews Analyzed: {df.shape[0]}")
        print(f"  - Training Samples: {X_train.shape[0]}")
        print(f"  - Testing Samples: {X_test.shape[0]}")
        print(f"\nSentiment Breakdown:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {sentiment}: {count} reviews ({percentage:.1f}%)")
        print(f"{'='*80}\n")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}")
        raise


if __name__ == "__main__":
    main()