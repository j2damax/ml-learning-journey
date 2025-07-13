# Polynomial Regression

## Overview
This directory contains my exploration and implementation of polynomial regression, a form of regression analysis where the relationship between the independent variable x and dependent variable y is modeled as an nth degree polynomial.

## Key Concepts Learned

### What is Polynomial Regression?
- Extension of linear regression where we fit a polynomial equation to the data
- Used when the relationship between variables is non-linear
- Formula: `y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε`
- **Key Insight**: "Polynomial Regression is still technically a form of linear regression. It works by first creating new, polynomial features from your original input variable"

### When to Use Polynomial Regression
- ✅ When data shows curved/non-linear patterns
- ✅ When linear regression underfits the data  
- ✅ For modeling complex relationships with simple polynomial terms
- ✅ To capture feature interactions (e.g., TV × Radio synergy effects)
- ❌ Avoid high degrees (>4) to prevent overfitting
- ❌ Not suitable for extrapolation beyond training data range - (Extrapolation in the context of polynomial regression means making predictions outside the range of data that the model was trained on)
- ❌ Be careful with numerical instability at very high degrees

## Implementation Details

### Key Steps Implemented

#### Part 1: Basic Polynomial Regression with NumPy
- Single feature polynomial regression using `np.polyfit()`
- Comparison between linear (degree=1) and cubic (degree=3) fits
- Manual implementation of polynomial predictions
- Visualization of different polynomial curves

#### Part 2: Multiple Linear Regression with Sklearn
- Multi-feature linear regression baseline model
- Train/test split with proper evaluation
- Residual analysis and model diagnostics
- Model persistence using joblib
- **Baseline Performance**: RMSE = 1.65

#### Part 3: Polynomial Regression with PolynomialFeatures
- Feature transformation using `PolynomialFeatures`
- Systematic degree selection (1-9)
- Comparison of polynomial vs linear performance
- **Best Performance**: RMSE = 1.2 (degree 2-3)

#### Part 4: Production Best Practices
- Feature scaling considerations
- Model pipeline development
- Cross-validation recommendations
- Regularization strategies

4. **Visualization**
   - Original data scatter plot
   - Polynomial fits of different degrees
   - Residual plots
   - Validation curves

## Key Insights & Learnings


### Technical Insights
- ✅ Higher degree polynomials can capture more complex patterns but risk overfitting
- ✅ Feature scaling becomes critical with higher degree polynomials due to vastly different magnitudes
- ✅ Degree 2-3 provided optimal balance between complexity and generalization
- ✅ PolynomialFeatures creates both polynomial terms (x², x³) and interaction terms (TV×Radio)
- ✅ Feature engineering expanded 3 features to 19 features with degree=3
- ✅ Residual analysis confirmed model assumptions (normality, zero-centered errors)

### Practical Lessons
- ✅ Always start with linear regression as baseline (RMSE: 1.65)
- ✅ Systematically test degrees 1-5 to find optimal complexity
- ✅ Monitor both training and validation performance to detect overfitting
- ✅ Save both model and transformer for consistent predictions
- ✅ Use proper train/test splits (67%/33%) for reliable evaluation
- ✅ Document feature transformations for production deployment

### Common Pitfalls Encountered
- ✅ **Overfitting**: Degrees 4+ showed test RMSE explosion (up to 7.8) while training error decreased
- ✅ **Feature scaling**: Polynomial features create vastly different scales (TV=230 vs TV³=12M)
- ✅ **Transform vs Fit-Transform**: Must use `transform()` only on new data, not `fit_transform()`
- ✅ **Model persistence**: Need to save both PolynomialFeatures transformer and the model
- ✅ **Numerical instability**: Very high degrees can cause computational issues

### **Common Split Ratios in ML:**
| Split Ratio | Use Case | Pros | Cons |
|-------------|----------|------|------|
| **80%/20%** | Large datasets (>10k samples) | More training data | Smaller test set |
| **70%/30%** | Medium datasets (1k-10k) | Balanced approach | Standard choice |
| **67%/33%** | Small-medium datasets | Good balance | Your choice |
| **60%/40%** | Small datasets (<1k) | Larger test set | Less training data |

### **Your Dataset Context:**
- **Advertising dataset**: ~200 samples
- **67%/33%** = 134 training samples + 66 test samples
- This gives you enough training data while keeping a substantial test set

## Code Snippets & References

### Essential Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load
import scipy.stats as stats
```

### Basic NumPy Polynomial Regression
```python
# Simple polynomial fit using NumPy
m, c = np.polyfit(x, y, deg=1)  # Linear regression
a, b, c, d = np.polyfit(x, y, deg=3)  # Cubic regression

# Generate predictions
potential_spend = np.linspace(0, 500, 100)
predicted_sales = potential_spend * m + c  # Linear
# For cubic: y = ax³ + bx² + cx + d
predicted_sales_cubic = (a*potential_spend**3) + (b*potential_spend**2) + (c*potential_spend**1) + d
```

### Sklearn Polynomial Pipeline
```python
# Create polynomial features and fit model
poly_converter = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_converter.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Systematic Degree Selection
```python
train_rmse_errors = []
test_rmse_errors = []

for degree in range(1, 10):
    poly_converter = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly_converter.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        poly_features, Y, test_size=0.33, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    train_rmse_errors.append(train_rmse)
    test_rmse_errors.append(test_rmse)
```

### Model Persistence for Production
```python
# Save both transformer and model
dump(poly_converter, 'Final_Converter.joblib')
dump(final_model, 'Final_model.joblib')

# Load and predict on new data
loaded_converter = load('Final_Converter.joblib')
loaded_model = load('Final_model.joblib')

# For new campaign: [TV=149, Radio=22, Newspaper=12]
campaign = [[149, 22, 12]]
poly_features = loaded_converter.transform(campaign)  # Only transform!
prediction = loaded_model.predict(poly_features)
```

## Results Summary

### Best Performing Model
- **Optimal Degree**: 2-3 (from systematic evaluation of degrees 1-9)
- **Best RMSE**: 1.2 (Polynomial) vs 1.65 (Linear baseline)
- **Improvement**: 27% reduction in RMSE over linear regression
- **Feature Expansion**: 3 → 9 features (degree 2) or 3 → 19 features (degree 3)
- **Dataset**: Advertising data (200 samples, TV/Radio/Newspaper → Sales)

### Performance Comparison
| Degree | Train RMSE | Test RMSE | Features | Notes |
|--------|------------|-----------|----------|-------|
| 1      | ~1.7       | ~1.65     | 3        | Linear baseline |
| 2      | ~1.5       | ~1.2      | 9        | **Optimal performance** |
| 3      | ~1.4       | ~1.2      | 19       | Slight improvement |
| 4      | ~1.2       | ~3.5      | 34       | Overfitting begins |
| 5      | ~0.8       | ~7.8      | 55       | Severe overfitting |

### Model Diagnostics
- **Residual Analysis**: ✅ Normally distributed, centered at zero
- **Heteroscedasticity**: ⚠️ Some variance increase at higher sales values
- **Q-Q Plot**: ✅ Strong adherence to normal distribution line
- **Feature Interactions**: TV×Radio interaction captured effectively

## Next Steps & Extensions

### Immediate Improvements
- ✅ **Completed**: Systematic polynomial degree evaluation (1-9)
- ✅ **Completed**: Model persistence and loading for production
- [ ] Try regularized polynomial regression (Ridge/Lasso) for high degrees
- [ ] Implement proper cross-validation for degree selection (vs single split)
- [ ] Add feature scaling pipeline for numerical stability
- [ ] Compare with other non-linear methods (splines, kernel methods)

### Advanced Topics to Explore
- [ ] Cross-validation with `validation_curve` for robust degree selection
- [ ] Ridge/Lasso regularization to handle overfitting at high degrees
- [ ] Feature scaling with `StandardScaler` after polynomial transformation
- [ ] Pipeline approach combining all preprocessing steps
- [ ] Orthogonal polynomials (Chebyshev, Legendre) for numerical stability
- [ ] Piecewise polynomial regression (splines)
- [ ] Polynomial regression with interaction-only features
- [ ] Bayesian polynomial regression with uncertainty quantification

## Resources & References

### Documentation
- [Scikit-learn PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

### Further Reading
- [ ] "Pattern Recognition and Machine Learning" by Christopher Bishop (Chapter 1)
- [ ] "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman (Chapter 2)
- [ ] Online courses and tutorials on polynomial regression

### Related Topics
- Linear Regression
- Ridge/Lasso Regression  
- Spline Regression
- Kernel Methods
- Feature Engineering

---

## Notes & Reflections

### What Worked Well
- **Systematic approach**: Testing degrees 1-9 revealed clear overfitting pattern
- **Baseline comparison**: Linear regression provided solid performance benchmark (RMSE 1.65)
- **Feature engineering**: PolynomialFeatures effectively captured non-linear relationships
- **Model persistence**: Successfully saved and loaded both transformer and model
- **Visualization**: Bias-variance tradeoff clearly visible in degree comparison plots
- **Real predictions**: Successfully predicted sales for new advertising campaigns

### Challenges Faced
- **Overfitting detection**: Learned to recognize when test error increases while training error decreases
- **Feature scaling**: Discovered polynomial features create vastly different magnitudes (TV=230 vs TV³=12M)
- **Transform vs fit_transform**: Critical distinction for production deployment
- **Model persistence**: Need to save both PolynomialFeatures and LinearRegression objects
- **Residual analysis**: Interpreting Q-Q plots and heteroscedasticity patterns

### Key Takeaways
- **Start simple**: Always establish linear regression baseline before adding complexity
- **Systematic evaluation**: Test multiple degrees to find bias-variance sweet spot
- **Production awareness**: Feature scaling and proper pipelines essential for real deployment
- **Overfitting is real**: High-degree polynomials can dramatically hurt generalization
- **Feature interactions matter**: TV×Radio interactions significantly improved model performance
- **Model diagnostics**: Residual analysis validates model assumptions and reveals insights

---

*Last Updated: July 13, 2025*
