Sentiment Analyzer

OVERVIEW:

A complete NLP (Natural Language Processing) project that classifies text as Positive or Negative sentiment. Unlike the previous twoprojects which worked with structured numbers, this project works with raw human language movie reviews, product feedback, social media posts, or any free-form text.

The system processes raw text through a full NLP pipeline(cleaning → vectorization → classification) and returns a sentiment prediction with confidence score and word-level explanations.

WHERE I GOT THE DATA:

Dataset: IMDB Movie Reviews Dataset

Source:  HuggingFace Datasets library (original source: Stanford AI Lab)

Size:    50,000 reviews (25,000 train + 25,000 test)

Balance: Perfectly balanced — 50% positive, 50% negative
         
Target:  Sentiment (positive/negative)

Sample reviews:

Positive: "One of the best films I've ever seen. The acting was superb and the story was gripping."

Negative: "Complete waste of time. Boring plot, terrible acting, and a nonsensical ending."

WHAT I DID — STEP BY STEP:

PHASE 1: EDA & Text Analysis

- Loaded 50,000 reviews from HuggingFace datasets library
  
- Analyzed review length distribution:
  
  Average words per review: 231
  
  Maximum words: 2,470
  
  Minimum words: 4
  
- Found dataset is perfectly balanced (no SMOTE needed)
  
- Analyzed most common words in positive vs negative reviews
  
- Key insight: "good" appears in BOTH positive and negative top-10 lists — negative reviews say "not good", showing 
  why word context (bigrams) matters for sentiment

PHASE 2: Text Cleaning Pipeline

Applied a 5-step cleaning process to every review:

  Step 1: Remove HTML tags (<br />, <b>, etc.) IMDB reviews contain raw HTML formatting
  
  Step 2: Lowercase all text "BRILLIANT" and "brilliant" are the same word
  
  Step 3: Remove punctuation and numbers, Keeps only alphabetic characters
  
  Step 4: Remove stopwords (150+ common English words), "the", "is", "and", "a" carry no sentiment
  
  Step 5: Remove words shorter than 3 characters, Filters noise like "ok", "hi"
  
  Result: Average words reduced from 231 → 118 per review(Removed 49% of words while keeping all sentiment signal)

PHASE 3: TF-IDF Vectorization

This is the key step that converts text into numbers that a machine learning model can process.

TF-IDF = Term Frequency × Inverse Document Frequency

- TF  = how often a word appears in THIS review
  
- IDF = how rare the word is across ALL reviews
  
- Effect: Common words like "movie" get LOW scores rare meaningful words like "masterpiece" get HIGH scores

Settings used:

- max_features=50,000  (keep top 50K most important words)
  
- ngram_range=(1,2)    (unigrams AND bigrams)
  
- min_df=2             (word must appear in ≥2 reviews)
  
- max_df=0.95          (ignore words in >95% of reviews)
  
- sublinear_tf=True    (log normalization)

WHY BIGRAMS MATTER:

Without bigrams: "not" + "good" → both neutral

With bigrams:    "not good" → clearly negative signal. This is critical for capturing negation in sentiment!

Result: Each review becomes a sparse vector of 50,000 numbers (only ~0.23% are non-zero values)

PHASE 4: Model Training & Evaluation

- Trained 3 models suited for high-dimensional text:(Note: Random Forest excluded — too slow with 50K features)

  Model                | Accuracy | F1     | ROC-AUC
  
  Linear SVM           | 89.8%    | 0.898  | 0.963
  
  Logistic Regression  | 89.2%    | 0.892  | 0.962
  
  Naive Bayes          | 88.3%    | 0.883  | 0.952

WHY LINEAR SVM WON:

Support Vector Machines find the optimal hyperplane that maximizes the margin between positive and negative reviews in 50,000-dimensional space. For high-dimensional text data this approach is proven to be most effective.

Top words for POSITIVE sentiment (model learned):excellent (2.949), great (2.919), amazing (2.627),entertaining (2.468), perfect (2.347), fun (2.183)

Top words for NEGATIVE sentiment (model learned):worst (-3.936), awful (-3.713), bad (-3.580),fails (-3.095), poor (-2.873), boring (-2.808)

PHASE 5: Web Application

- Side-by-side layout — Single Review AND Batch Analysis visible simultaneously
  
- Single Review mode:
  
  → Confidence gauge (50-100%)
  
  → Word-level highlighting (green=positive, red=negative)
  
  → Word count statistics
  
- Batch Analysis mode:
  
  → Results table with sentiment, confidence per review
  
  → Overall verdict (Mostly Positive/Negative)
  
  → Pie chart breakdown
  
  → Word highlighting per review
  
- Sample buttons for quick testing
  
- Deployed to Streamlit Cloud

RESULTS

Best Model:    Linear SVM

Accuracy:      89.8% (correctly classifies 9/10 reviews)

ROC-AUC:       0.963 (exceptional — near-perfect ranking)

F1 Score:      0.898

TF-IDF Features: 50,000

Training Size: 20,000 reviews

Test Size:     5,000 reviews

TECH STACK

Language:   Python 

Libraries:  scikit-learn, NLTK, HuggingFace datasets,pandas, numpy, plotly, streamlit

NLP Tools:  TF-IDF Vectorizer, NLTK stopwords, LinearSVC classifier

Deployment: Streamlit Cloud

LIVE DEMO: https://sentiment-analysis-for-movie-reviews-mgtikd2ah3ydausbnehmjw.streamlit.app/

