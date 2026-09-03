# Data Flow Diagram: Scraping to Model Deployment

## Executive Summary
This document presents a comprehensive flow diagram of your house price prediction system, from initial data scraping from rumah123.com to final model deployment. The system uses a hybrid approach combining Fuzzy Logic, Random Forest, and Genetic Algorithm optimization.

## Main Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           HOUSE PRICE PREDICTION SYSTEM                                 │
│                        Yogyakarta Real Estate Data Pipeline                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   DATA SOURCE    │───▶│  DATA SCRAPING   │───▶│ DATA PROCESSING  │───▶│  MODEL TRAINING  │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
                                                                              │
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐         │
│  WEB INTERFACE   │◀───│   PREDICTION     │◀───│  MODEL STORAGE   │◀────────┘
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Detailed Flow Breakdown

### 1. Data Source & Web Scraping

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA SCRAPING PHASE                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   rumah123.com  │ 
│   Yogyakarta    │
│   Property      │
│   Listings      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐      ┌──────────────────────────┐
│  scraper.py     │────▶ │  Raw Data Fields:        │
│                 │      │  • title                 │
│ Features:       │      │  • price                 │
│ • Random UA     │      │  • bedroom               │
│ • Rate limiting │      │  • bathroom              │
│ • Error handling│      │  • carport               │
│ • Retry logic   │      │  • LT (land area)        │
│ • Multi-page    │      │  • LB (building area)    │
│   scraping      │      │  • badges                │
└─────────────────┘      │  • agent                 │
                         │  • updated               │
                         │  • location              │
                         │  • link                  │
                         │  • description           │
                         └──────────────────────────┘
                                     │
                                     ▼
                         ┌──────────────────────────┐
                         │     houses.csv           │
                         │   (Raw Dataset)          │
                         └──────────────────────────┘
```

### 2. Data Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPROCESSING PHASE                        │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   houses.csv     │
│  (Raw Data)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      ┌─────────────────────────────────────────────┐
│ preprocessing.py │────▶ │           DATA CLEANING STEPS               │
│                  │      │                                             │
│ Main Functions:  │      │ 1. Type Conversion:                         │
│ • preprocess_    │      │    • convert_to_numeric() - Price parsing   │
│   data()         │      │    • preprocess_area() - Area extraction    │
│ • apply_fuzzy_   │      │    • preprocess_updated() - Date parsing    │
│   logic()        │      │    • preprocess_location() - Location split │
│ • engineer_      │      │                                             │
│   features()     │      │ 2. Data Quality Checks:                     │
│ • remove_        │      │    • Remove duplicates                      │
│   outliers()     │      │    • Filter non-residential properties      │
│                  │      │    • Remove listings >60 days old           │
│                  │      │    • Validate price ranges                  │
│                  │      │                                             │
│                  │      │ 3. Missing Value Handling:                  │
│                  │      │    • Median imputation for numeric          │
│                  │      │    • Mode imputation for categorical        │
└──────────────────┘      └─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐      ┌─────────────────────────────────────────────┐
│ Feature          │────▶ │         FEATURE ENGINEERING                 │
│ Engineering      │      │                                             │
│                  │      │ Core Features:                              │
│                  │      │ • price_per_m2_land = price / LT            │
│                  │      │ • price_per_m2_building = price / LB        │
│                  │      │ • building_efficiency = LB / LT             │
│                  │      │ • listing_age_days = days since update      │
│                  │      │ • room_density = total_rooms / LB           │
│                  │      │ • bathroom_bedroom_ratio                    │
│                  │      │                                             │
│                  │      │ Categorical Encoding:                       │
│                  │      │ • kecamatan_encoded (Label Encoding)        │
│                  │      │ • kecamatan_freq (Frequency Encoding)       │
└──────────────────┘      └─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐      ┌─────────────────────────────────────────────┐
│ Fuzzy Logic      │────▶ │            FUZZY LOGIC SYSTEM               │
│ Application      │      │                                             │
│                  │      │ 1. Price Quality Assessment:                │
│                  │      │    • Very Low/Low/Medium/High/Very High     │
│                  │      │    • Membership functions for price ranges  │
│                  │      │                                             │
│                  │      │ 2. Room Quality System:                     │
│                  │      │    • Bedroom count fuzzy sets               │
│                  │      │    • Bathroom count fuzzy sets              │
│                  │      │    • Combined room quality score            │
│                  │      │                                             │
│                  │      │ 3. Area Quality System:                     │
│                  │      │    • Land area (LT) fuzzy sets              │
│                  │      │    • Building area (LB) fuzzy sets          │
│                  │      │    • Combined area quality score            │
│                  │      │                                             │
│                  │      │ 4. Carport Quality System:                  │
│                  │      │    • None/Few/Many categories               │
│                  │      │                                             │
│                  │      │ Output Features:                            │
│                  │      │ • fuzzy_area_quality                        │
│                  │      │ • fuzzy_room_quality                        │
│                  │      │ • fuzzy_quality_score (composite)           │
│                  │      │ • efficiency_category                       │
│                  │      │ • fuzzy_value_assessment                    │
└──────────────────┘      └─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐      ┌─────────────────────────────────────────────┐
│ Outlier Removal  │────▶ │           OUTLIER DETECTION                 │
│                  │      │                                             │
│                  │      │ Methods Available:                          │
│                  │      │ • IQR Method (Interquartile Range)          │
│                  │      │ • Z-Score Method                            │
│                  │      │ • Combined Approach                         │
│                  │      │                                             │
│                  │      │ Configurable Thresholds:                    │
│                  │      │ • price: IQR=2.0, Z=3.0                     │
│                  │      │ • LT/LB: IQR=1.75, Z=3.0                    │
│                  │      │ • bedroom/bathroom: IQR=1.5, Z=2.5          │
└──────────────────┘      └─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Preprocessed Dataset    │
│  • Clean numerical data  │
│  • Engineered features   │
│  • Fuzzy logic features  │
│  • Encoded categoricals  │
│  • Outliers removed      │
└──────────────────────────┘
```

### 3. Model Training & Optimization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL TRAINING PHASE                            │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│  Preprocessed Data   │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐      ┌─────────────────────────────────────────┐
│   train_test_split   │────▶ │        DATA SPLITTING                   │
│                      │      │                                         │
│   • 80% Training     │      │  X_train, X_test, y_train, y_test       │
│   • 20% Testing      │      │  Random state = 42                      │
│   • Stratified       │      │                                         │
└──────────────────────┘      └─────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐      ┌─────────────────────────────────────────┐
│  Base Random Forest  │────▶ │      BASE MODEL TRAINING                │
│  Training            │      │                                         │
│                      │      │  Parameters:                            │
│  rf_model.py:        │      │  • n_estimators = 100                   │
│  • train_base_rf_    │      │  • max_depth = None                     │
│    model()           │      │  • min_samples_split = 2                │
│  • evaluate_rf_      │      │  • min_samples_leaf = 1                 │
│    model()           │      │  • max_features = 'sqrt'                │
│                      │      │  • bootstrap = True                     │
│                      │      │  • random_state = 42                    │
│                      │      │  • n_jobs = -1                          │
│                      │      │                                         │
│                      │      │  Evaluation Metrics:                    │
│                      │      │  • MAPE (Mean Absolute % Error)         │
│                      │      │  • R² Score                             │
│                      │      │  • RMSE (Root Mean Square Error)        │
│                      │      │  • MAE (Mean Absolute Error)            │
│                      │      │  • Cross-validation (5-fold)            │
└──────────────────────┘      └─────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐      ┌─────────────────────────────────────────┐
│  Genetic Algorithm   │────▶ │    HYPERPARAMETER OPTIMIZATION          │
│  Optimization        │      │                                         │
│                      │      │  GA Parameters:                         │
│  ga_optimizer.py:    │      │  • Population size: 20                  │
│  • GAOptimizer       │      │  • Generations: 10                      │
│    class             │      │  • Mutation rate: 0.2 (adaptive)        │
│  • genetic_          │      │  • Tournament selection                 │
│    algorithm()       │      │  • Blend crossover (BLX-α)              │
│  • fitness_          │      │  • Gaussian mutation                    │
│    function()        │      │  • Elitism (best individual preserved)  │
│                      │      │                                         │
│                      │      │  Optimization Bounds:                   │
│                      │      │  • n_estimators: [50, 200]              │
│                      │      │  • max_depth: [5, 30]                   │
│                      │      │  • min_samples_split: [2, 20]           │
│                      │      │  • min_samples_leaf: [1, 10]            │
│                      │      │  • max_features: [0.1, 1.0]             │
│                      │      │                                         │
│                      │      │  Fitness Function:                      │
│                      │      │  • Minimize MAPE through CV             │
│                      │      │  • 5-fold cross-validation              │
│                      │      │  • Caching for efficiency               │
└──────────────────────┘      └─────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐      ┌─────────────────────────────────────────┐
│  Optimized Model     │────▶ │      OPTIMIZED MODEL TRAINING           │
│  Training            │      │                                         │
│                      │      │  Best Parameters from GA:               │
│                      │      │  • Automatically tuned hyperparameters  │
│                      │      │  • Improved performance metrics         │
│                      │      │                                         │
│                      │      │  Model Evaluation:                      │
│                      │      │  • Cross-validation performance         │
│                      │      │  • Test set evaluation                  │
│                      │      │  • Feature importance analysis          │
│                      │      │  • Permutation importance               │
│                      │      │                                         │
│                      │      │  Model Artifacts:                       │
│                      │      │  • optimized_rf_model.pkl               │
│                      │      │  • model_metadata.json                  │
│                      │      │  • feature_importance.png               │
│                      │      │  • model_comparison.png                 │
└──────────────────────┘      └─────────────────────────────────────────┘
```

### 4. Model Evaluation & Validation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MODEL EVALUATION PHASE                           │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐      ┌─────────────────────────────────────────┐
│   Model Performance  │────▶ │         PERFORMANCE METRICS             │
│   Assessment         │      │                                         │
│                      │      │  Primary Metrics:                       │
│                      │      │  • MAPE: Mean Absolute Percentage Error │
│                      │      │  • R²: Coefficient of Determination     │
│                      │      │  • RMSE: Root Mean Square Error         │
│                      │      │  • MAE: Mean Absolute Error             │
│                      │      │                                         │
│                      │      │  Cross-Validation:                      │
│                      │      │  • K-Fold CV (k=5)                      │
│                      │      │  • Stratified sampling                  │
│                      │      │  • Shuffle enabled                      │
│                      │      │                                         │
│                      │      │  Comparison Analysis:                   │
│                      │      │  • Base vs Optimized model              │
│                      │      │  • Performance improvement %            │
│                      │      │  • Statistical significance tests       │
└──────────────────────┘      └─────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐      ┌─────────────────────────────────────────┐
│   Feature Analysis   │────▶ │        FEATURE IMPORTANCE               │
│                      │      │                                         │
│                      │      │  Analysis Methods:                      │
│                      │      │  • Mean Decrease in Impurity (MDI)      │
│                      │      │  • Permutation Importance               │
│                      │      │                                         │
│                      │      │  Key Insights:                          │
│                      │      │  • Most predictive features             │
│                      │      │  • Feature ranking and scores           │
│                      │      │  • Visualization charts                 │
│                      │      │                                         │
│                      │      │  Expected Important Features:           │
│                      │      │  1. price_per_m2_land                   │
│                      │      │  2. price_per_m2_building               │
│                      │      │  3. LT (land area)                      │
│                      │      │  4. LB (building area)                  │
│                      │      │  5. fuzzy_quality_score                 │
│                      │      │  6. building_efficiency                 │
│                      │      │  7. kecamatan_encoded                   │
└──────────────────────┘      └─────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐
│   Visualization      │
│   Generation         │
│                      │
│   Outputs:           │
│   • Performance      │
│     comparison charts│
│   • Feature          │
│     importance plots │
│   • Residual plots   │
│   • Prediction       │
│     scatter plots    │
└──────────────────────┘
```

### 5. Model Storage & Deployment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       MODEL STORAGE & DEPLOYMENT                        │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐      ┌─────────────────────────────────────────┐
│   Model Artifacts    │────▶ │           MODEL PERSISTENCE             │
│   Serialization      │      │                                         │
│                      │      │  Saved Files:                           │
│                      │      │  • optimized_rf_model.pkl               │
│                      │      │    └─ Trained RandomForest object       │
│                      │      │                                         │
│                      │      │  • optimized_rf_model_metadata.json     │
│                      │      │    ├─ Model parameters                  │
│                      │      │    ├─ Feature names                     │
│                      │      │    ├─ Feature importance                │
│                      │      │    ├─ Evaluation metrics                │
│                      │      │    ├─ Training date                     │
│                      │      │    └─ Model type                        │
│                      │      │                                         │
│                      │      │  • optimized_rf_model_params.txt        │
│                      │      │    └─ Human-readable parameters         │
│                      │      │                                         │
│                      │      │  • Evaluation visualizations:           │
│                      │      │    ├─ feature_importance.png            │
│                      │      │    ├─ rf_model_comparison.png           │
│                      │      │    ├─ prediction_scatter.png            │
│                      │      │    └─ residual_plot.png                 │
└──────────────────────┘      └─────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐      ┌─────────────────────────────────────────┐
│  Web Interface       │────▶ │         STREAMLIT INTERFACE             │
│  Deployment          │      │                                         │
│                      │      │  Pages Structure:                       │
│  Streamlit App:      │      │  • Home.py - Main landing page          │
│  • Home.py           │      │  • 1_Dataset.py - Raw data view         │
│  • pages/            │      │  • 2_Preprocessing.py - Data pipeline   │
│    ├─ 1_Dataset.py   │      │  • 3_Model.py - Model statistics        │
│    ├─ 2_Preprocessing│      │  • 4_Prediction.py - Price prediction   │
│    ├─ 3_Model.py     │      │                                         │
│    └─ 4_Prediction.py│      │  Key Features:                          │
│                      │      │  • Interactive data exploration         │
│                      │      │  • Real-time predictions                │
│                      │      │  • Model performance visualization      │
│                      │      │  • Fuzzy logic integration              │
│                      │      │  • Confidence scoring                   │
│                      │      │  • Location-based analysis              │
└──────────────────────┘      └─────────────────────────────────────────┘
```

### 6. Prediction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PREDICTION PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐      ┌─────────────────────────────────────────┐
│   User Input         │────▶ │           INPUT PROCESSING              │
│                      │      │                                         │
│   Form Fields:       │      │  Required Inputs:                       │
│   • Bedroom count    │      │  • bedroom: int [1-10]                  │
│   • Bathroom count   │      │  • bathroom: int [1-10]                 │
│   • Land area (LT)   │      │  • LT: float [20-2000] m²               │
│   • Building area    │      │  • LB: float [20-1000] m²               │
│   • Carport count    │      │  • carport: int [0-5]                   │
│   • Kecamatan        │      │  • kecamatan: string (dropdown)         │
│   • Listing age      │      │  • listing_age: int [0-365] days        │
└──────────────────────┘      └─────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐      ┌─────────────────────────────────────────┐
│   Feature            │────▶ │       FEATURE ENGINEERING               │
│   Engineering        │      │                                         │
│                      │      │  Computed Features:                     │
│   Real-time          │      │  • building_efficiency = LB / LT        │
│   calculation        │      │  • price_per_m2_land (estimated)        │
│                      │      │  • price_per_m2_building (estimated)    │
│                      │      │  • kecamatan_encoded (lookup)           │
│                      │      │                                         │
│                      │      │  Fuzzy Logic Features:                  │
│                      │      │  • fuzzy_area_quality                   │
│                      │      │    └─ Real-time fuzzy inference         │
│                      │      │  • fuzzy_room_quality                   │
│                      │      │    └─ Bedroom/bathroom assessment       │
│                      │      │  • fuzzy_quality_score                  │
│                      │      │    └─ Composite quality metric          │
└──────────────────────┘      └─────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐      ┌─────────────────────────────────────────┐
│   Model Inference    │────▶ │          PRICE PREDICTION               │
│                      │      │                                         │
│   predict_price_rf() │      │  Process:                               │
│                      │      │  1. Feature vector assembly             │
│                      │      │  2. Missing feature imputation          │
│                      │      │  3. Feature ordering/alignment          │
│                      │      │  4. Random Forest prediction            │
│                      │      │  5. Confidence score calculation        │
│                      │      │                                         │
│                      │      │  Output:                                │
│                      │      │  • Predicted price (Rupiah)             │
│                      │      │  • Price per m² land                    │
│                      │      │  • Price per m² building                │
│                      │      │  • Confidence score [0-1]               │
│                      │      │  • Comparison with area averages        │
└──────────────────────┘      └─────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐      ┌─────────────────────────────────────────┐
│   Result             │────▶ │          RESULT PRESENTATION            │
│   Presentation       │      │                                         │
│                      │      │  Display Elements:                      │
│                      │      │  • Formatted price (Rp X,XXX,XXX)       │
│                      │      │  • Confidence indicator                 │
│                      │      │    ├─ High (>85%): Green                │
│                      │      │    ├─ Medium (70-85%): Yellow           │
│                      │      │    └─ Low (<70%): Red                   │
│                      │      │  • Price per m² metrics                 │
│                      │      │  • Area comparison gauge                │
│                      │      │  • Feature contribution analysis        │
│                      │      │  • Debug information (expandable)       │
└──────────────────────┘      └─────────────────────────────────────────┘
```

## Technology Stack

### Core Libraries & Frameworks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            TECHNOLOGY STACK                            │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   WEB SCRAPING       │  │   DATA PROCESSING    │  │   MACHINE LEARNING   │
│                      │  │                      │  │                      │
│ • BeautifulSoup      │  │ • Pandas             │  │ • scikit-learn       │
│ • urllib             │  │ • NumPy              │  │ • scikit-fuzzy       │
│ • requests           │  │ • datetime           │  │ • Random Forest      │
│ • random UA          │  │ • re (regex)         │  │ • Genetic Algorithm  │
│ • tqdm (progress)    │  │ • json               │  │ • Cross-validation   │
│ • colorama           │  │ • pickle             │  │                      │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   VISUALIZATION      │  │   WEB INTERFACE      │  │   UTILITIES          │
│                      │  │                      │  │                      │
│ • Matplotlib         │  │ • Streamlit          │  │ • logging            │
│ • Seaborn            │  │ • Plotly             │  │ • os/sys             │
│ • Plotly Express     │  │ • HTML/CSS           │  │ • typing             │
│ • PIL (Images)       │  │ • Markdown           │  │ • argparse           │
│                      │  │                      │  │ • traceback          │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

## Data Flow Metrics & Quality

### Pipeline Performance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE METRICS                               │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│   DATA QUALITY       │
│                      │
│ Input Validation:    │
│ • Required columns   │
│ • Data type checks   │
│ • Range validation   │
│ • Null handling      │
│                      │
│ Quality Thresholds:  │
│ • Price: 100M-20B    │
│ • LT: 15-5000 m²     │
│ • LB: 10-2000 m²     │
│ • Rooms: 1-20        │
└──────────────────────┘

┌──────────────────────┐
│   PROCESSING STATS   │
│                      │
│ Typical Pipeline:    │
│ • Raw data: ~10K     │
│ • After cleaning:    │
│   ~7-8K records      │
│ • Outlier removal:   │
│   5-15% reduction    │
│ • Features: 10-15    │
│   engineered         │
│                      │
│ Processing Time:     │
│ • Scraping: ~2-5 min │
│ • Preprocessing:     │
│   ~30-60 seconds     │
│ • Training: ~2-10    │
│   minutes            │
└──────────────────────┘

┌──────────────────────┐
│   MODEL PERFORMANCE  │
│                      │
│ Target Metrics:      │
│ • MAPE: <15%         │
│ • R²: >0.75          │
│ • RMSE: Minimized    │
│                      │
│ GA Optimization:     │
│ • 20 individuals     │
│ • 10 generations     │
│ • 5-fold CV          │
│ • Early stopping     │
│                      │
│ Typical Improvement: │
│ • 5-20% MAPE         │
│   reduction          │
│ • 0.05-0.15 R²       │
│   increase           │
└──────────────────────┘
```

## File Structure & Organization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PROJECT STRUCTURE                            │
└─────────────────────────────────────────────────────────────────────────┘

rumah/
├── Home.py                          # Main Streamlit app
├── pages/                           # Streamlit pages
│   ├── 1_Dataset.py                 # Raw data visualization
│   ├── 2_Preprocessing.py           # Preprocessing dashboard
│   ├── 3_Model.py                   # Model statistics
│   └── 4_Prediction.py              # Prediction interface
├── dataset/
│   ├── houses.csv                   # Raw scraped data
│   └── dataset-scraper/
│       ├── scraper.py               # Web scraping script
│       └── ua.txt                   # User agent list
├── utils/
│   ├── preprocessing.py             # Main preprocessing module
│   └── helper.py                    # Utility functions
├── train/
│   ├── preprocess.py               # Preprocessing script
│   ├── train.py                    # Model training script
│   ├── rf_model.py                 # Random Forest implementation
│   └── ga_optimizer.py             # Genetic Algorithm optimizer
├── model/                          # Trained models
│   ├── optimized_rf_model.pkl      # Serialized model
│   ├── optimized_rf_model_metadata.json
│   ├── optimized_rf_model_params.txt
│   └── evaluation/                 # Evaluation plots
├── processed_data/                 # Preprocessed datasets
├── reports/                        # Analysis reports
└── logs/                          # System logs
```

## Key Innovation Points

### 1. Hybrid Intelligence Approach
- **Fuzzy Logic**: Handles uncertainty in property valuation
- **Random Forest**: Robust ensemble learning for price prediction  
- **Genetic Algorithm**: Automated hyperparameter optimization

### 2. Comprehensive Feature Engineering
- **Domain-specific features**: price_per_m2, building_efficiency
- **Temporal features**: listing_age_days
- **Fuzzy-derived features**: quality scores and assessments
- **Location encoding**: Multiple encoding strategies for kecamatan

### 3. Advanced Data Quality Pipeline
- **Multi-stage cleaning**: Type conversion, validation, outlier removal
- **Configurable thresholds**: Different outlier detection parameters
- **Quality metrics tracking**: Comprehensive pipeline monitoring

### 4. Production-Ready Deployment
- **Interactive web interface**: Streamlit-based dashboard
- **Real-time predictions**: On-demand price estimation
- **Confidence scoring**: Prediction reliability assessment
- **Comprehensive logging**: Full pipeline traceability

This system represents a complete end-to-end machine learning pipeline specifically designed for Yogyakarta real estate price prediction, incorporating domain expertise through fuzzy logic and achieving optimization through genetic algorithms.

