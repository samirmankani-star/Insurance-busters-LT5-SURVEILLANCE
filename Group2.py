import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


df = pd.read_excel('surveillance_formatted.csv')


# MAPPINGS & DATA TRANSFORMATION 
aware_mapping = {
    "Extremely unaware": 1, "Very unaware": 2, "Somewhat unaware": 3,
    "Neither aware or unware": 4, "Somewhat aware": 5, "Very aware": 6, "Extremely aware": 7
}

implement_mapping = {
    "Not at all": 1, "Very little": 2, "Somewhat": 3, 
    "Moderately": 4, "Considerably": 5, "Almost always": 6, "Always": 7
}

insurance_aware_mapping = {
    "Not aware at all": 1, "Slightly aware": 2, "Moderately aware": 3,
    "Somewhat aware": 4, "Very aware": 5, "Extremely aware": 6
}

comfortable_mapping = {
    "Extremely uncomfortable": 1, "Very uncomfortable": 2, "Somewhat uncomfortable": 3,
    "Neither comfortable nor uncomfortable": 4, "Somewhat comfortable": 5, 
    "Very comfortable": 6, "Extremely comfortable": 7
}

accept_mapping = {
    "Do not accept at all": 1, "Very slightly accept": 2, "Slightly accept": 3,
    "Moderately accept": 4, "Quite accept": 5, "Mostly accept": 6, "Fully accept": 7
}

education_mapping = {
    "No formal education": 0, "High school diploma": 1, 
    "Bachelor's Degree": 2, "Master's degree or higher": 3
}

# Apply Mappings
df["Q1"] = df['To what extent are you aware of big data profiling?'].map(aware_mapping)
df["Q2"] = df['To what extent do you implement measures to safeguard your personal data in online environments?'].map(implement_mapping)
df["Q3"] = df['To what extent are you aware that insurance companies use personal data for profiling purposes?'].map(insurance_aware_mapping)
df["Q4"] = df['To what extent are you aware that insurance companies use profiling in their decision-making processes?'].map(insurance_aware_mapping)
df["Q5"] = df['To what extent do you feel comfortable with insurance companies using your personal data for profiling purposes?'].map(comfortable_mapping)
df["Q6"] = df['To what extent do you accept insurance companies using your personal data for profiling purposes?'].map(accept_mapping)
df["Education_numeric"] = df["What is your education?"].map(education_mapping)

# Additional Score columns for Country Analysis
aware_decision_col = "To what extent are you aware that insurance companies use profiling in their decision-making processes?"
aware_bdp_col = "To what extent are you aware of big data profiling?"
accept_col = "To what extent do you accept insurance companies using your personal data for profiling purposes?"

df['Awareness_Score'] = df[aware_decision_col].map(insurance_aware_mapping) 
df['Awareness_Score_NIS'] = df[aware_bdp_col].map(aware_mapping)
df['Acceptance_Score'] = df[accept_col].map(accept_mapping)

# DESCRIPTIVE STATISTICS 
print("Head of Data:")
print(df.head())

question_cols = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
stats_table = df[question_cols].agg(["mean", "median", "std"]).T
print("\nQuestion Statistics:")
print(stats_table)

stats_by_country = df.groupby(country_col)[['Awareness_Score', 'Acceptance_Score']].agg(['count', 'mean', 'median', 'std'])
print("\nDescriptive Statistics by Country:")
print(stats_by_country.round(2))

# Correlation calculations
correlations = {
    "Awareness (Q3)": df[["Q3", "Education_numeric"]].corr().iloc[0,1],
    "Acceptance (Q6)": df[["Q6", "Education_numeric"]].corr().iloc[0,1],
    "Big Data Awareness (Q1)": df[["Q1", "Education_numeric"]].corr().iloc[0,1]
}
print("\nCorrelations with Education:", correlations)

# VISUALIZATIONS 
sns.set_theme(style="whitegrid")

# Age Distribution
plt.figure(figsize=(8, 6))
age_counts = df['What age group do you belong in?'].value_counts().sort_index()
plt.bar(age_counts.index.astype(str), age_counts.values)
plt.title('Age Group Distribution')
plt.xticks(rotation=45)
plt.show()

# Correlation Bar Chart
plt.figure(figsize=(8,5))
plt.bar(correlations.keys(), correlations.values(), color="skyblue")
plt.ylim(-0.1, 0.5)
plt.ylabel("Correlation with Education")
plt.title("Correlation between Education and Responses")
plt.axhline(0, color='gray', linestyle='--')
plt.show()

# Age Group Boxplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
age_order = ['18-24','25-34','35-44','45-54','55-64','65+']
age_col = 'What age group do you belong in?'

sns.boxplot(data=df, x=age_col, y='Awareness_Score', hue=age_col, palette="Set3", ax=axes[0], order=age_order, legend=False)
axes[0].set_title('Awareness by Age Group')
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(data=df, x=age_col, y='Acceptance_Score', hue=age_col, palette="Set3", ax=axes[1], order=age_order, legend=False)
axes[1].set_title('Acceptance by Age Group')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Acceptance vs Awareness Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Awareness_Score', y='Acceptance_Score', hue='Awareness_Score', palette="viridis", legend=False)
plt.title("Acceptance Levels by Awareness Score for Insurance")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Awareness_Score_NIS', y='Acceptance_Score', hue='Awareness_Score_NIS', palette="viridis", legend=False)
plt.title("Acceptance Levels by Awareness Score (NIS)")
plt.xticks(ticks=range(7), labels=[1,2,3,4,5,6,7])
plt.show()

