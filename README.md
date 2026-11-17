# Medical Insurance Dataset EDA

This project performs exploratory data analysis on the medical insurance dataset in [medical_insurance_eda.ipynb](medical_insurance_eda.ipynb).

The dataset is stored in [Dataset/insurance.csv](Dataset/insurance.csv) and contains 1,338 rows and 7 variables:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`
- `charges`

The goal is to understand the data, check quality issues, summarize distributions, examine relationships, and prepare the dataset for future modeling.

## Project structure

```text
medical-insurance-dataset-eda/
├── medical_insurance_eda.ipynb
├── README.md
├── pyproject.toml
├── uv.lock
├── Dataset/
│   └── insurance.csv
└── src/
    └── exploratory_data_analysis_project_1/
        └── __init__.py
```

## Setup

This project uses Python and common data-science libraries:

- pandas
- NumPy
- matplotlib
- seaborn
- scipy

Dependencies are listed in [pyproject.toml](pyproject.toml).

## Run

```bash
uv sync
jupyter notebook medical_insurance_eda.ipynb
```

Or with a standard virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scipy
jupyter notebook medical_insurance_eda.ipynb
```

## Notes

The notebook checks:

- missing values
- duplicate rows
- column types
- distributions and summary statistics
- relationships between features and `charges`
- outliers and skewness

This is a preprocessing and EDA project, not a final prediction model.

- average age: about 39.2 years
- average BMI: about 30.7
- average children: about 1.1
- average charges: about 13,270.42

This gives a first snapshot of the population and the target distribution.

### 7. Univariate Analysis

The notebook focuses on understanding each variable individually.

#### Distribution of Numeric Features

Histograms and KDE plots are used to inspect whether the distribution is normal, skewed, or highly spread out.

Observations:

- age is fairly balanced around middle age
- BMI has a moderate distribution with some spread
- children is right-skewed, with many people having 0–2 children
- charges is heavily right-skewed, with a long tail of high costs

This is a crucial finding because highly skewed target variables often require transformation before certain machine-learning models are trained.

#### Distribution of Categorical Features

The notebook inspects category counts for:

- `sex`
- `smoker`
- `region`

It finds:

- sex is almost balanced between male and female
- smoker is imbalanced, with far fewer smokers than non-smokers
- region is fairly evenly distributed across the four U.S. regions

This imbalance matters because a model trained on this dataset may find that smoking is a highly predictive feature, which is exactly what the later analysis suggests.

### 8. Bivariate and Multivariate Analysis

The notebook then moves from inspecting individual columns to understanding relationships between columns.

#### Correlation Analysis

It checks how numeric variables relate to each other using correlation. The notebook computes pairwise correlations between:

- age and charges
- bmi and charges
- children and charges

The key result is:

- age and charges have a modest positive correlation
- bmi and charges also show a slight positive correlation
- children does not show a strong relationship with charges

The notebook also calculates Pearson correlation coefficients and p-values to judge whether the relationships are statistically meaningful.

This helps separate meaningful relationships from random noise.

#### Scatterplots

The notebook creates scatterplots such as:

- `age` vs `charges`
- `bmi` vs `charges`

These plots help reveal trends and clusters.

The key observation is:

- charges rise with age
- charges rise with BMI
- but the strongest pattern is the cluster of very high charges among smokers

#### Pairplot

The notebook builds a pairplot including numerical variables and smoker status as the hue.

This is especially useful because it lets the analyst see multiple relationships at once.

The pairplot reveals:

- clusters of higher charges among smokers
- clear separation between smoker and non-smoker groups
- strong interaction between smoking and cost

#### Grouped Boxplots

The notebook compares the distribution of numeric variables by category using boxplots.

Examples:

- charges by smoker status
- charges by sex
- charges by region

Important observation:

- smokers have substantially higher insurance charges than non-smokers
- region does not appear to strongly change charges
- there may be a mild difference by sex, but smoking status is much more dominant

This provides early evidence that smoking is likely the main driver of cost differences.

### 9. Crosstab Analysis

The notebook also looks at relationships between categorical variables using crosstabs.

Examples:

- sex vs smoker
- smoker vs region

This helps answer whether some groups are more represented among smokers.

This matters because if smoking is concentrated in a particular segment, then a model may confuse demographic structure with smoking effect unless carefully checked.

### 10. Outlier Detection

The notebook places a strong emphasis on detecting unusual values, especially in the `charges` column.

#### Boxplots

Boxplots show the spread of:

- age
- bmi
- charges

The output shows that `charges` has several extreme high values, while `bmi` also has some outliers.

#### IQR Method

The notebook calculates the interquartile range (IQR) for numeric columns and identifies values outside the lower/upper bound.

This approach helps quantify how many values are considered outliers.

#### Z-Score Method

The notebook also computes z-scores to detect extreme values in standardized form.

The notebook notes that using z-score methods on a highly skewed distribution can be less reliable, because the data are not normally distributed.

#### Practical Interpretation

The conclusion is:

- charges are skewed and highly variable
- several extreme values exist in the target column
- these outliers may not necessarily be errors, but they are important because they affect model behavior

The notebook explicitly states that outlier removal is not generally recommended as a modeling step unless there is evidence of bad data or a specific modeling reason.

### 11. Summary of Key Insights

The notebook ends with a summary of its major findings:

#### Data Quality

- no missing values found
- no constant or quasi-constant columns
- no ID-like columns
- duplicate row identified and documented
- data types are appropriate

#### Univariate Insights

- age and BMI are fairly regular
- children is mildly skewed
- charges are heavily right-skewed with many extreme values
- smoker distribution is imbalanced
- region distribution is balanced

#### Bivariate / Multivariate Insights

- charges rise slightly with age
- charges increase with BMI
- smokers have dramatically higher insurance charges
- region has only a small effect
- smoker status is the most influential pattern discovered

#### Outliers

- charges contain many extreme values
- BMI also has some outliers

These insights are exactly the type of findings that a good EDA should produce before modeling begins.

## Key Business / Research Interpretation

From a practical viewpoint, the analysis suggests:

- smoking is the strongest factor related to higher medical insurance cost
- age and BMI contribute gradually, but not as strongly as smoking status
- regional differences are smaller than expected
- the target variable is highly skewed, so it may need transformation for some modeling approaches

This means the notebook is not just describing raw data; it is uncovering the most important relationships that will influence how a later predictive model should be built.

## Preprocessing Steps Recommended Before Modeling

The notebook explicitly recommends the following preprocessing steps for a future modeling stage:

1. Encode categorical variables
   - convert `sex`, `smoker`, and `region` into numeric values
   - use label encoding or one-hot encoding depending on the model

2. Scale numerical features
   - useful for models such as SVMs, KNN, and neural networks

3. Handle skewness in the target
   - apply a log transformation to `charges` to reduce right-skewness

4. Handle outliers if necessary
   - optional depending on the model and the reason for the outliers
   - especially relevant for linear models

5. Split data before transformations
   - train/test split should be done before applying scaling or transformations to avoid data leakage

6. Consider feature engineering
   - age groups, BMI categories, or interaction features may improve the modeling workflow

These rules are standard and show how the EDA directly informs the next stage of an ML pipeline.

## Final Takeaway

This project is a complete introduction to structured exploratory data analysis in a real-world dataset.

It demonstrates how to:

- inspect the raw dataset
- clean and validate it
- summarize distributions
- check for data issues
- analyze relationships between variables
- detect skewness and outliers
- prepare a foundation for future modeling

In simple terms: the notebook answers the question, “What does the insurance data look like, and what should we do before building a predictive model?”

The answer is that the dataset is clean, smoking is the strongest driver of high insurance charges, and the target variable is skewed enough that preprocessing will be important before modeling.

## How to Run the Project

To explore the notebook locally:

```bash
# create a virtual environment
python -m venv .venv

# activate environment
# Windows
.venv\Scripts\activate

# install dependencies
pip install numpy pandas matplotlib seaborn scipy

# open the notebook
jupyter notebook medical_insurance_eda.ipynb
```

Or, if you use uv:

```bash
uv sync
jupyter notebook medical_insurance_eda.ipynb
```

## Summary

This project is an educational EDA project designed to teach full-data understanding before machine learning.

It uses the medical insurance dataset to:

- explore structure and distributions
- identify quality issues
- uncover business-relevant patterns
- build a strong foundation for later predictive modeling

It is a good example of how EDA should be done: carefully, methodically, and with clear interpretation of the results.
