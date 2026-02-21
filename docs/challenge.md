# Challenge Documentation

## Part I

### Model Selection

**Chosen Model**: Logistic Regression with Top 10 Features and Class Balancing

**Justification**:
- The Data Scientist evaluated 6 different model configurations in the notebook:
  1. XGBoost (baseline)
  2. Logistic Regression (baseline)
  3. XGBoost with top 10 features + class balancing
  4. XGBoost with top 10 features (no balancing)
  5. Logistic Regression with top 10 features + class balancing
  6. Logistic Regression with top 10 features (no balancing)

- According to the notebook conclusions:
  - No noticeable difference in results between XGBoost and Logistic Regression
  - Reducing features to the top 10 does not decrease model performance
  - Class balancing improves model performance by increasing recall of class "1" (delayed flights)

- **Why Logistic Regression over XGBoost**:
  - Simpler model architecture, easier to operationalize and maintain
  - Faster inference times
  - Better interpretability for stakeholders
  - Lower memory footprint
  - Similar performance metrics as demonstrated in the notebook
  - No external dependencies beyond scikit-learn (XGBoost requires additional package)

### Feature Engineering

The following feature engineering logic was transcribed from the notebook:

#### 1. Period of Day (`period_day`) - `_get_period_day`
- Categorizes flight time into three periods based on `Fecha-I`:
  - **Morning (mañana)**: 05:00 - 11:59
  - **Afternoon (tarde)**: 12:00 - 18:59
  - **Night (noche)**: 19:00 - 23:59 or 00:00 - 04:59

**Note**: Fixed boundary logic to use inclusive comparisons (`<=` instead of `<`) to properly include boundary times.

#### 2. High Season (`high_season`) - `_is_high_season`
- Binary feature (1 or 0) indicating if the flight date falls within high season periods:
  - Dec 15 - Dec 31
  - Jan 1 - Mar 3
  - Jul 15 - Jul 31
  - Sep 11 - Sep 30

#### 3. Difference in Minutes (`min_diff`) - `_get_min_diff`
- Calculates the difference in minutes between `Fecha-O` (actual operation time) and `Fecha-I` (scheduled time)
- Formula: `((Fecha-O - Fecha-I).total_seconds()) / 60`

#### 4. Delay Target (`delay`)
- Binary target variable (1 or 0)
- Threshold: 15 minutes
- `delay = 1` if `min_diff > 15`, else `delay = 0`

### Preprocessing Pipeline

The `preprocess` method implements the following steps:

1. **Input Validation**:
   - Validates presence of required columns: `Fecha-I`, `OPERA`, `TIPOVUELO`, `MES`
   - Creates a copy of input data to avoid side effects

2. **Target Computation** (if `target_column="delay"` is provided):
   - Checks if `delay` column exists in data
   - If not, computes `min_diff` from `Fecha-O` and `Fecha-I`
   - Creates `delay` column using 15-minute threshold

3. **Feature Encoding**:
   - Creates one-hot encoded features for:
     - `OPERA` (airline) → `OPERA_*` columns
     - `TIPOVUELO` (flight type) → `TIPOVUELO_*` columns
     - `MES` (month) → `MES_*` columns

4. **Feature Selection**:
   - Selects only the top 10 features identified by the Data Scientist:
     - `OPERA_Latin American Wings`
     - `MES_7`
     - `MES_10`
     - `OPERA_Grupo LATAM`
     - `MES_12`
     - `TIPOVUELO_I`
     - `MES_4`
     - `MES_11`
     - `OPERA_Sky Airline`
     - `OPERA_Copa Air`
   - Missing features are added with value 0 (handles cases where new data doesn't contain certain categories)

5. **Output**:
   - Returns features DataFrame (for prediction)
   - Returns (features, target) tuple (for training when `target_column` is provided)

### Model Training

The `fit` method implements:

1. **Input Validation**:
   - Ensures features and target have matching row counts
   - Validates target is a single-column DataFrame
   - Checks for presence of positive samples (class 1)

2. **Class Balancing**:
   - Calculates class distribution: `n_y0` (class 0) and `n_y1` (class 1)
   - Computes class weights using the formula from the notebook:
     - `class_weight[1] = n_y0 / len(target)` (weight for delayed flights)
     - `class_weight[0] = n_y1 / len(target)` (weight for on-time flights)
   - This weighting scheme increases the importance of the minority class (delayed flights)

3. **Model Training**:
   - Initializes `LogisticRegression` with:
     - `class_weight`: computed weights for class balancing
     - `random_state=42`: for reproducibility
     - `max_iter=1000`: to ensure convergence
   - Fits the model on the provided features and target

### Prediction

The `predict` method:
- Validates that the model has been trained
- Returns predictions as a list of integers (0 or 1)
- Uses the trained model to predict on new feature data

### Bugs Fixed

1. **Type Hint Syntax Error**:
   - Fixed `Union(Tuple[...])` to `Union[Tuple[...]]` (correct Python type hint syntax)

2. **Period of Day Boundary Logic**:
   - Original notebook used exclusive comparisons (`>` and `<`)
   - Fixed to use inclusive comparisons (`<=` and `>=`) to properly include boundary times
   - Ensures times like 05:00, 11:59, 12:00, etc. are correctly classified

3. **Missing Feature Handling**:
   - Added logic to handle cases where one-hot encoding doesn't produce all expected features
   - Missing features are initialized to 0, ensuring consistent feature space

## Part II: API Implementation

### Framework
FastAPI (as required by challenge constraints)

### Endpoints

**GET `/health`**: Returns `{"status": "OK"}` for health checks
```python
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}
```

**POST `/predict`**: Predicts flight delays
- **Request**: `{"flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}]}`
- **Response**: `{"predict": [0]}`
- **Status Codes**: 200 (success), 400 (validation error), 500 (server error)

### Implementation Details

**Model Loading**: 
- Lazy loading via `get_model()` function (singleton pattern with global `_model` variable)
- Model loads and trains on first `/predict` request

**Input Validation** (`validate_flight()` function):
- Manual validation without Pydantic
- **OPERA**: Must be one of 22 valid airline names in `VALID_OPERAS` list
- **TIPOVUELO**: Must be "I" (International) or "N" (National)
- **MES**: Must be integer between 1-12
- Raises `HTTPException(400)` for invalid data

**Request Processing** (`post_predict()` endpoint):
1. Parse JSON body using `Request` object with `await req.json()`
2. Validate request structure (must have "flights" as non-empty list)
3. Validate each flight using `validate_flight()`
4. Convert flights to pandas DataFrame
5. Add dummy `Fecha-I` column (required by preprocess, not used for prediction)
6. Get model via `get_model()` (loads if not already loaded)
7. Preprocess using `DelayModel.preprocess()`
8. Predict using `DelayModel.predict()`
9. Return `{"predict": predictions}` as JSON

**Error Handling**:
- `HTTPException`: Re-raised to preserve status codes
- `ValueError`: Converted to 400 (validation errors)
- Other exceptions: Converted to 500 (internal server errors)
