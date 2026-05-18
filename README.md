# Restaurant Reviews Sentiment Analysis

A machine learning project that classifies restaurant reviews as positive or negative using Natural Language Processing and Naive Bayes classification.

## Overview

This project analyzes restaurant reviews and predicts customer sentiment (Positive/Negative) using:
- **Text Preprocessing**: Tokenization, stopword removal, stemming
- **Feature Extraction**: TF-IDF Vectorization
- **Classification**: Multinomial Naive Bayes
- **Visualization**: Confusion Matrix and Sentiment Distribution

## Project Structure

```
ReviewSense/
├── review.py                    # Main analysis script
├── Restaurant_Reviews.tsv       # Dataset (TSV format)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
```

## Requirements

- Python 3.7+
- See `requirements.txt` for package dependencies

## Installation

1. **Clone/Download the project**
   ```bash
   cd ReviewSense
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure dataset is present**
   - Place `Restaurant_Reviews.tsv` in the project directory

## Usage

Run the analysis:
```bash
python review.py
```

The script will:
1. Load and validate the restaurant reviews dataset
2. Display data shape and sample reviews
3. Plot sentiment distribution
4. Preprocess all reviews (lowercase, remove punctuation, stemming)
5. Apply TF-IDF vectorization
6. Train Naive Bayes classifier
7. Evaluate model performance
8. Display confusion matrix and classification metrics

## Output

The script generates:
- **Console Output**: Model accuracy and classification report
- **Visualizations**: 
  - Sentiment distribution bar chart
  - Confusion matrix heatmap

## Dataset Format

The `Restaurant_Reviews.tsv` file should have:
- `Review`: Text of the restaurant review
- `Liked`: Binary sentiment (1 = Positive, 0 = Negative)

Example:
```
Review                              Liked
Wow... Loved this place.            1
Crust is not good.                  0
```

## Features

✅ **Robust Error Handling** - Try-catch blocks for all critical operations
✅ **Logging** - Detailed logging for debugging and monitoring
✅ **Data Validation** - Checks for missing values and required columns
✅ **Configuration Management** - Centralized CONFIG dictionary
✅ **Modular Functions** - Well-organized, reusable functions
✅ **Documentation** - Docstrings and comments throughout
✅ **Best Practices** - Professional code structure and standards

## Model Performance

The Multinomial Naive Bayes model typically achieves:
- **Accuracy**: 75-85% (depends on dataset)
- **Metrics**: Precision, Recall, F1-Score per class

## Key Functions

| Function | Purpose |
|----------|---------|
| `load_data()` | Load and validate TSV data |
| `prepare_data()` | Clean and prepare dataset |
| `preprocess_text()` | Clean individual reviews |
| `build_vectorizer_and_encode()` | TF-IDF and label encoding |
| `train_model()` | Train classifier |
| `evaluate_model()` | Test and display metrics |
| `plot_confusion_matrix()` | Visualize prediction results |

## Future Improvements

- [ ] Test with multiple classifiers (SVM, Random Forest)
- [ ] Add hyperparameter tuning
- [ ] Implement cross-validation
- [ ] Add sentiment intensity prediction
- [ ] Create a REST API for predictions
- [ ] Save/load trained models

## Author

@Manasa-L-Hegde

## License

MIT License

## Run Output (2026-05-18)

The project was executed locally and produced the following console output and visual summary image.

```
2026-05-18 19:14:38,506 - INFO - Starting Restaurant Reviews Sentiment Analysis
2026-05-18 19:14:38,522 - INFO - NLTK stopwords already available
2026-05-18 19:14:38,522 - INFO - Loading data from Restaurant_Reviews.tsv
2026-05-18 19:14:38,555 - INFO - Successfully loaded data: 1000 rows, 2 columns
2026-05-18 19:14:38,563 - INFO - Data prepared: 1000 rows remaining after cleaning

================================================================================
RESTAURANT REVIEWS SENTIMENT ANALYSIS - REPORT
================================================================================

Dataset Overview:
   - Total Reviews: 1000
   - Columns: 2

Sample Reviews:
--------------------------------------------------------------------------------
                                                                                                                         review sentiment
                                                                                              Wow... Loved this place.  Positive
                                                                                                       Crust is not good.  Negative
                                                                     Not tasty and the texture was just nasty.  Negative
Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.  Positive
                                          The selection on the menu was great and so were the prices.  Positive
2026-05-18 19:14:38,571 - INFO - Preprocessing reviews...
2026-05-18 19:14:38,843 - INFO - Text preprocessing completed
2026-05-18 19:14:38,875 - INFO - TF-IDF vectorization complete: 1500 features
2026-05-18 19:14:38,877 - INFO - Sentiment classes: ['Negative' 'Positive']
2026-05-18 19:14:38,881 - INFO - Data split: 800 train, 200 test
2026-05-18 19:14:38,886 - INFO - Model training completed
2026-05-18 19:14:38,911 - INFO - Model accuracy: 0.7700
2026-05-18 19:14:40,899 - INFO - Sentiment distribution plot prepared
2026-05-18 19:14:41,254 - INFO - Confusion matrix plot prepared

============================================================
CLASSIFICATION REPORT
============================================================
                     precision    recall  f1-score   support

      Negative       0.75      0.79      0.77        96
      Positive       0.80      0.75      0.77       104

      accuracy                           0.77       200
    macro avg       0.77      0.77      0.77       200
weighted avg       0.77      0.77      0.77       200

============================================================


================================================================================
ANALYSIS COMPLETE ✓
================================================================================

Final Results:
   - Model Accuracy: 0.7700 (77.00%)
   - Total Reviews Analyzed: 1000
   - Training Samples: 800
   - Testing Samples: 200

Sentiment Breakdown:
   - Positive: 500 reviews (50.0%)
   - Negative: 500 reviews (50.0%)
================================================================================

The visual summary was saved as `sentiment_summary.png` and is included in this repository.
```
