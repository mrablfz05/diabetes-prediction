import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import skew, norm, kurtosis

diabetes = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")
diabetes.head()

# Data Cleaning and Exploring Data
diabetes.shape
diabetes.columns.tolist()

# EDA for Diabetes_012 col
diabetes_012 = diabetes["Diabetes_012"]
diabetes_012.describe()
diabetes_012.unique()
diabetes_012.isnull().sum()
diabetes_012.head(10)
diabetes_012.value_counts()
diabetes_012.dtype

diabetes_012_counts = diabetes_012.value_counts().reindex([0, 1, 2], fill_value=0)

diabetes_012_counts.plot(kind='bar')
plt.xlabel("Diabetes Class")
plt.ylabel("Count")
plt.title("Distribution of Diabetes_012 Classes")
plt.xticks(ticks=[0, 1, 2], labels=["No Diabetes (0)", "Prediabetes (1)", "Diabetes (2)"], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

skew(diabetes_012)

diabetes_012_skew = skew(diabetes_012)

plt.figure(figsize=(8, 5))
sns.histplot(diabetes_012, kde=False, bins=3, discrete=True, color="skyblue", edgecolor="black")

plt.title(f"Skewness Plot of Diabetes_012 (Skewness = {diabetes_012_skew:.2f})")
plt.xlabel("Diabetes Class")
plt.ylabel("Frequency")
plt.xticks([0, 1, 2], ["No Diabetes (0)", "Prediabetes (1)", "Diabetes (2)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(diabetes_012, kde=True, stat="density", bins=3, color="skyblue", edgecolor="black", label="Data KDE")

# Overlay normal distribution with same mean & std
mu, std = diabetes_012.mean(), diabetes_012.std()
xmin, xmax = diabetes_012.min(), diabetes_012.max()
x = np.linspace(xmin - 1, xmax + 1, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r--', label="Normal Distribution")

plt.title(f"Distribution of Diabetes_012 (Skew = {diabetes_012.skew():.2f})")
plt.xlabel("Diabetes Class")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


diabetes_012_counts = diabetes_012.value_counts().sort_index()
diabetes_012_percentages = diabetes_012.value_counts(normalize=True).sort_index() * 100
diabetes_012_df = pd.DataFrame({
    "Count": diabetes_012_counts,
    "Percentage": diabetes_012_percentages.round(2)
})
print(diabetes_012_df)

if "BMI" in diabetes.columns:
    print(diabetes.groupby("Diabetes_012")["BMI"].mean())


# EDA for HighBP col
highBp = diabetes["HighBP"]
highBp.value_counts()
highBp.describe()

highBp_counts = highBp.value_counts().sort_index()
highBp_counts.plot(kind='bar', color=["#5DADE2", "#E74C3C"])
plt.title("Distribution of High Blood Pressure (HighBP)")
plt.xlabel("High Blood Pressure (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.xticks([0, 1], ["No (0)", "Yes (1)"], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


percent_df = pd.crosstab(diabetes_012, diabetes["HighBP"], normalize='index') * 100
percent_df = percent_df.reset_index().melt(id_vars="Diabetes_012", var_name="HighBP", value_name="Percentage")

plt.figure(figsize=(8, 5))
sns.barplot(data=percent_df, x="Diabetes_012", y="Percentage", hue="HighBP", palette=["#5DADE2", "#E74C3C"])

plt.title("Percentage of High Blood Pressure by Diabetes Class")
plt.xlabel("Diabetes Class")
plt.ylabel("Percentage")
plt.xticks(ticks=[0, 1, 2], labels=["No Diabetes (0)", "Prediabetes (1)", "Diabetes (2)"])
plt.legend(title="HighBP", labels=["No (0)", "Yes (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# EDA for HighChol col
HighChol = diabetes["HighChol"]
diabetes["HighChol"].describe()

HighChol_counts = HighChol.value_counts().sort_index()
HighChol_counts.plot(kind='bar', color=["#5DADE2", "#A8E73C"])
plt.title("Distribution of High Chol (HighBP)")
plt.xlabel("High Chol (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.xticks([0, 1], ["No (0)", "Yes (1)"], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


percent_HighCol_diabetes_df = pd.crosstab(diabetes_012, HighChol, normalize='index') * 100
percent_HighCol_diabetes_df = percent_HighCol_diabetes_df.reset_index().melt(id_vars="Diabetes_012", var_name="HighChol", value_name="Percentage")

plt.figure(figsize=(8, 5))
sns.barplot(data=percent_HighCol_diabetes_df, x="Diabetes_012", y="Percentage", hue="HighChol", palette=["#5DADE2", "#E73CB9"])

plt.title("Percentage of High Chol Pressure by Diabetes Class")
plt.xlabel("Diabetes Class")
plt.ylabel("Percentage")
plt.xticks(ticks=[0, 1, 2], labels=["No Diabetes (0)", "Prediabetes (1)", "Diabetes (2)"])
plt.legend(title="HighChol", labels=["No (0)", "Yes (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

skew(HighChol)

plt.figure(figsize=(8, 5))
sns.histplot(HighChol, kde=False, bins=3, discrete=True, color="skyblue", edgecolor="black")

plt.title(f"Skewness Plot of HighChol (Skewness = {skew(HighChol):.2f})")
plt.xlabel("HighChol Class")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["No HighChol (0)", "Have HighChol (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
stats.probplot(HighChol, dist="norm", plot=plt)
plt.title("QQ Plot of HighChol")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(HighChol, kde=True, stat="density", bins=3, color="skyblue", edgecolor="black", label="Data KDE")

# Overlay normal distribution with same mean & std
mu, std = HighChol.mean(), HighChol.std()
xmin, xmax = HighChol.min(), HighChol.max()
x = np.linspace(xmin - 1, xmax + 1, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r--', label="Normal Distribution")

plt.title(f"Distribution of HighChol (Skew = {HighChol.skew():.2f})")
plt.xlabel("HighChol Class")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# EDA for CholCheck col
CholCheck = diabetes["CholCheck"]
diabetes["CholCheck"].describe()

CholCheck_counts = CholCheck.value_counts().reindex([0, 1], fill_value=0)

CholCheck_counts.plot(kind='bar')
plt.xlabel("CholCheck Class")
plt.ylabel("Count")
plt.title("Distribution of Had CholCheck Classes")
plt.xticks(ticks=[0, 1], labels=["Had not CholCheck (0)", "Had CholCheck (1)"], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

CholCheck_skew = skew(CholCheck)

plt.figure(figsize=(8, 5))
sns.histplot(CholCheck, kde=False, bins=3, discrete=True, color="skyblue", edgecolor="black")

plt.title(f"Skewness Plot of CholCheck (Skewness = {CholCheck_skew:.2f})")
plt.xlabel("CholCheck Class")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["Had not CholCheck (0)", "Had CholCheck (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=CholCheck, palette="pastel", edgecolor="black")

plt.title("CholCheck Distribution (Binary Variable)")
plt.xlabel("CholCheck (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# functional code instead of repeated them **********************************************
CholCheck_counts = CholCheck.value_counts().sort_index()
percentages = CholCheck.value_counts(normalize=True).sort_index() * 100
CholCheck_df = pd.DataFrame({
    "Count": CholCheck_counts,
    "Percentage": percentages.round(2)
})
print(CholCheck_df)

plt.figure(figsize=(6, 6))
stats.probplot(CholCheck, dist="norm", plot=plt)
plt.title("QQ Plot of CholCheck")
plt.grid(True)
plt.tight_layout()
plt.show()

# EDA for BMI col
BMI = diabetes["BMI"]
diabetes["BMI"].dtype

plt.figure(figsize=(8, 5))
sns.histplot(BMI, kde=True, color="skyblue", edgecolor="black", bins=100)
plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

stats.probplot(BMI, dist="norm", plot=plt)
plt.title("Q-Q Plot of BMI")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

skew(BMI)

plt.figure(figsize=(7, 5))
sns.boxplot(x="CholCheck", y="BMI", data=diabetes, palette="Set3")
plt.title("BMI Distribution by CholCheck")
plt.xlabel("CholCheck (0 = No, 1 = Yes)")
plt.ylabel("BMI")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
sns.boxplot(x=BMI, color="lightcoral")
plt.title("Box Plot of BMI")
plt.xlabel("BMI")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=range(len(BMI)), 
    y="BMI", 
    hue="Smoker", 
    palette={0: "skyblue", 1: "salmon"}, 
    data=diabetes,
    alpha=0.7,
    edgecolor="black"
)

plt.title("Scatter Plot of BMI Colored by Smoker Status")
plt.xlabel("Index")
plt.ylabel("BMI")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Smoker (0 = No, 1 = Yes)")
plt.tight_layout()
plt.show()

bmi = diabetes["BMI"]
Q1 = bmi.quantile(0.25)
Q3 = bmi.quantile(0.75)
IQR = Q3 - Q1

lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR

outliers = bmi[(bmi < lower_threshold) | (bmi > upper_threshold)]
n_outliers = outliers.count()

print(f"Number of outliers: {n_outliers}")


# Correlation on dataset
corr = diabetes.corr(numeric_only=True)

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title('Heatmap of Feature Correlations')
plt.show()

# EDA for Smoker col
Smoker = diabetes["Smoker"]
diabetes["Smoker"].describe()

plt.figure(figsize=(6, 4))
sns.countplot(x=Smoker, palette="pastel", edgecolor="black")

plt.title("Smoker Distribution (Binary Variable)")
plt.xlabel("Smoker (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

smoker_ratio = diabetes.groupby("Diabetes_012")["Smoker"].mean().reset_index()

plt.figure(figsize=(7, 5))
sns.barplot(x="Diabetes_012", y="Smoker", data=smoker_ratio, palette="viridis")

plt.title("Proportion of Smokers by Diabetes Class")
plt.xlabel("Diabetes Class (0 = No, 1 = Pre, 2 = Yes)")
plt.ylabel("Smoker Ratio")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# EDA for Stroke col
Stroke = diabetes["Stroke"]
diabetes["Stroke"].value_counts()
plt.figure(figsize=(6, 4))
sns.countplot(x=Stroke, palette="Blues", edgecolor="black")

plt.title("Stroke Distribution (Binary Variable)")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Stroke', y='BMI', data=diabetes)

plt.title('Box Plot of BMI by Stroke Status')
plt.xlabel('Stroke (0 = No, 1 = Yes)')
plt.ylabel('BMI')
plt.show()

stroke = diabetes["Stroke"]
Q1 = stroke.quantile(0.25)
Q3 = stroke.quantile(0.75)
IQR = Q3 - Q1

lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR

outliers = stroke[(stroke < lower_threshold) | (stroke > upper_threshold)]
n_outliers = outliers.count()

print(f"Number of outliers: {n_outliers}")

# EDA for HeartDiseaseorAttack col
HeartDiseaseorAttack = diabetes["HeartDiseaseorAttack"]
diabetes["HeartDiseaseorAttack"].value_counts()

HeartDiseaseorAttack_skew = skew(diabetes["HeartDiseaseorAttack"])

plt.figure(figsize=(8, 5))
sns.histplot(HeartDiseaseorAttack, kde=False, bins=3, discrete=True, color="skyblue", edgecolor="black")

plt.title(f"Skewness Plot of HeartDiseaseorAttack (Skewness = {HeartDiseaseorAttack_skew:.2f})")
plt.xlabel("HeartDiseaseorAttack Class")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["Had not HeartDiseaseorAttack (0)", "Had HeartDiseaseorAttack (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# EDA for PyisActivity col
PhysActivity = diabetes["PhysActivity"]
PhysActivity_skew = skew(PhysActivity)

plt.figure(figsize=(8, 5))
sns.histplot(diabetes["PhysActivity"], kde=False, bins=3, discrete=True, color="skyblue", edgecolor="black")

plt.title(f"Skewness Plot of PhysActivity (Skewness = {PhysActivity_skew:.2f})")
plt.xlabel("PhysActivity Class")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["Had not PhysActivity lastmonth (0)", "Had PhysActivity lastmonth (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# EDA for Fruits col
Fruits = diabetes["Fruits"]

Fruits_skew = skew(Fruits)
plt.figure(figsize=(8, 5))
sns.histplot(Fruits, kde=False, bins=3, discrete=True, color="skyblue", edgecolor="black")

plt.title(f"Skewness Plot of Fruits (Skewness = {Fruits_skew:.2f})")
plt.xlabel("Fruits Class")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["Not Consume fruit regularly. (0)", "Consume fruit regularly. (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# EDA for Veggies col
Veggies = diabetes["Veggies"]
Veggies_skew = skew(Veggies)
plt.figure(figsize=(8, 5))
sns.histplot(Veggies, kde=False, bins=3, discrete=True, color="skyblue", edgecolor="black")

plt.title(f"Skewness Plot of Veggies (Skewness = {Veggies_skew:.2f})")
plt.xlabel("Veggies Class")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["Not Consume Veggies regularly. (0)", "Consume Veggies regularly. (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

veggies_counts = Veggies.value_counts().sort_index()
veggies_percentages = Veggies.value_counts(normalize=True).sort_index() * 100
veggies_df = pd.DataFrame({
    "Count": veggies_counts,
    "Percentage": veggies_percentages.round(2)
})
print(veggies_df)

# EDA for HvyAlcoholConsump col
HvyAlcoholConsump = diabetes["HvyAlcoholConsump"]
prop_df = pd.crosstab(diabetes['Sex'], HvyAlcoholConsump, normalize='index')
prop_df.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='coolwarm')
plt.title('Proportion of Heavy Alcohol Consumption by Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Proportion')
plt.legend(title='HvyAlcoholConsump (0 = No, 1 = Yes)')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

diabetes["Sex"].value_counts()
women = diabetes[diabetes['Sex'] == 0]
men = diabetes[diabetes['Sex'] == 1]

women_counts = women['HvyAlcoholConsump'].value_counts(normalize=True).sort_index()
men_counts = men['HvyAlcoholConsump'].value_counts(normalize=True).sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Pie Chart for Women
axes[0].pie(
    women_counts, 
    labels=['No (0)', 'Yes (1)'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightblue', 'salmon']
)
axes[0].set_title('Women (Sex = 0)\nHeavy Alcohol Consumption')

# Plot 2: Bar Chart for Men
axes[1].bar(
    ['No (0)', 'Yes (1)'], 
    men_counts.values,
    color=['lightgreen', 'tomato']
)
axes[1].set_title('Men (Sex = 1)\nHeavy Alcohol Consumption')
axes[1].set_ylabel('Proportion')
axes[1].set_ylim(0, 1)
for i, v in enumerate(men_counts.values):
    axes[1].text(i, v + 0.02, f"{v:.1%}", ha='center')

plt.tight_layout()
plt.show()

heavy_alcohol = diabetes[diabetes['HvyAlcoholConsump'] == 1]

# Count how many men and women
sex_counts = heavy_alcohol['Sex'].value_counts().sort_index()  # 0 = female, 1 = male

# Plot
plt.figure(figsize=(6, 5))
sns.barplot(x=sex_counts.index, y=sex_counts.values, palette=['lightblue', 'tomato'])

plt.xticks([0, 1], ['Female (0)', 'Male (1)'])
plt.xlabel('Sex')
plt.ylabel('Number of Heavy Alcohol Consumers')
plt.title('Heavy Alcohol Consumption by Sex (Count)')

# Add count labels
for i, v in enumerate(sex_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# EDA for AnyHealthcare col
AnyHealthcare = diabetes["AnyHealthcare"]
AnyHealthcare_skew = skew(AnyHealthcare)

plt.figure(figsize=(8, 5))
sns.histplot(AnyHealthcare, kde=False, bins=3, discrete=True, color="skyblue", edgecolor="black")

plt.title(f"Skewness Plot of AnyHealthcare (Skewness = {AnyHealthcare_skew:.2f})")
plt.xlabel("AnyHealthcare Class")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["No AnyHealthcare. (0)", "Have AnyHealthcare. (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# EDA for NoDocBcCost col
NoDocbcCost = diabetes["NoDocbcCost"]
NoDocbcCost_skew = skew(NoDocbcCost)
plt.figure(figsize=(8, 5))
sns.histplot(NoDocbcCost, kde=False, bins=3, discrete=True, color="skyblue", edgecolor="black")

plt.title(f"Skewness Plot of NoDocbcCost (Skewness = {NoDocbcCost_skew:.2f})")
plt.xlabel("AnyHealthcare Class")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["Yes NoDocbcCost. (1)", "No NoDocbcCost. (0)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# EDA for GenHlth col
GenHlth = diabetes["GenHlth"]

genhlth_counts = GenHlth.value_counts().sort_index()
genhlth_percentages = GenHlth.value_counts(normalize=True).sort_index() * 100
genhlth_df = pd.DataFrame({
    "Count": genhlth_counts,
    "Percentage": genhlth_percentages.round(2)
})
print(genhlth_df)

labels = ['Excellent (1)', 'Very Good (2)', 'Good (3)', 'Fair (4)', 'Poor (5)']

cmap = plt.cm.PuBuGn_r
colors = cmap(np.linspace(0.2, 0.8, len(genhlth_counts)))

plt.figure(figsize=(7, 7))
plt.pie(
    genhlth_counts,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors
)
plt.title('General Health Distribution (Pie Chart)')
plt.axis('equal')
plt.show()

# EDA for MenHlth col
MentHlth = diabetes["MentHlth"]
invalid_values = diabetes[(MentHlth < 0) | (MentHlth > 30)]
print("Number of out of range: ", len(invalid_values))

plt.figure(figsize=(8, 6))
sns.scatterplot(x='MentHlth', y='PhysHlth', data=diabetes, alpha=0.3)

plt.title('Scatter Plot: Mental vs Physical Health (past 30 days)')
plt.xlabel('MentHlth (Days mentally unhealthy)')
plt.ylabel('PhysHlth (Days physically unhealthy)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(MentHlth.index, MentHlth, alpha=0.3, color='purple')
plt.title('Scatter Plot of Mental Health Days')
plt.xlabel('Index')
plt.ylabel('MentHlth (Days)')
plt.grid(True)
plt.tight_layout()
plt.show()

MentHlth_skew = skew(MentHlth)

plt.figure(figsize=(8, 6))
sns.histplot(MentHlth, bins=31, kde=True, color='mediumseagreen')

plt.title(f'Distribution of MentHlth (Skewness = {MentHlth_skew:.2f})')
plt.xlabel('MentHlth (Days)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

menthlth_counts = MentHlth.value_counts().sort_index()
menthlth_percentages = MentHlth.value_counts(normalize=True).sort_index() * 100
menthlth_df = pd.DataFrame({
    "Count": menthlth_counts,
    "Percentage": menthlth_percentages.round(2)
})
print(menthlth_df)

MentHlth_skew_val = skew(MentHlth)
MentHlth_kurt_val = kurtosis(MentHlth)

plt.figure(figsize=(8, 6))
sns.histplot(MentHlth, bins=31, kde=True, color='royalblue')

plt.title(f'MentHlth Distribution\nSkewness = {MentHlth_skew_val:.2f} | Kurtosis = {MentHlth_kurt_val:.2f}')
plt.xlabel('MentHlth (Days)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

diabetes['MentHlth_flag'] = MentHlth.apply(lambda x: '0 days' if x == 0 else '1+ days')

counts = diabetes['MentHlth_flag'].value_counts()

plt.figure(figsize=(6, 5))
sns.barplot(x=counts.index, y=counts.values, palette=['lightcoral', 'mediumseagreen'])

plt.title('Number of People by MentHlth Zero vs Non-Zero Days')
plt.xlabel('MentHlth Category')
plt.ylabel('Count')

for i, v in enumerate(counts.values):
    plt.text(i, v + 2000, f"{v:,}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

value_counts = diabetes['MentHlth'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.stem(value_counts.index, value_counts.values, linefmt='gray', markerfmt='o', basefmt=' ')
plt.title('Lollipop Plot of MentHlth (Unhealthy Mental Days)')
plt.xlabel('Number of Unhealthy Mental Days')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

sns.violinplot(x='Stroke', y='MentHlth', data=diabetes, palette='Set2')
plt.title('MentHlth Distribution by Stroke Status')
plt.xlabel('Stroke (0 = No, 1 = Yes)')
plt.ylabel('Unhealthy Mental Days')
plt.show()

diabetes['AgeBin'] = pd.cut(diabetes['Age'], bins=6)
diabetes['MentBin'] = pd.cut(MentHlth, bins=[-1, 0, 5, 15, 30])

heatmap_data = pd.crosstab(diabetes['AgeBin'], diabetes['MentBin'], normalize='index')

sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Proportion of MentHlth Levels Across Age Bins')
plt.ylabel('Age Group')
plt.xlabel('MentHlth Days')
plt.show()

sns.swarmplot(x='Sex', y='MentHlth', data=diabetes.sample(1000), palette='coolwarm', size=3)
plt.title('MentHlth by Sex (sample of 1000)')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('MentHlth Days')
plt.show()

sns.boxplot(x='Sex', y='MentHlth', data=diabetes, palette='pastel')
plt.title('Distribution of MentHlth by Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('MentHlth Days')
plt.show()