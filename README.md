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
