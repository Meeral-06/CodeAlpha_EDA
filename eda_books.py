import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
df = pd.read_csv("books.csv")

# 1. Basic structure
print("=== Dataset Info ===")
print(df.info())
print("\n=== Summary Statistics ===")
print(df.describe(include="all"))
print("\n=== Missing Values ===")
print(df.isnull().sum())

# 2. Rating distribution
rating_order = ["One", "Two", "Three", "Four", "Five"]
print("\n=== Rating Counts ===")
print(df["rating"].value_counts().reindex(rating_order))

# 3. Convert rating to number
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["rating_num"] = df["rating"].map(rating_map)

# 4. Price statistics
print("\n=== Price Statistics ===")
print("Average price:", round(df["price"].mean(), 2))
print("Median price:", round(df["price"].median(), 2))
print("Max price:", df["price"].max())
print("Min price:", df["price"].min())

# 5. Hypothesis test: Does price differ by rating?
groups = [group["price"].values for _, group in df.groupby("rating")]
f_stat, p_value = stats.f_oneway(*groups)
print(f"\n=== ANOVA Test ===")
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.5f}")
if p_value < 0.05:
    print("Result: There is a significant difference in price across ratings.")
else:
    print("Result: No significant difference in price across ratings.")

# 6. Visualizations
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=20, kde=True, color="teal")
plt.title("Price Distribution")
plt.savefig("price_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="rating", order=rating_order, palette="viridis")
plt.title("Book Count by Rating")
plt.savefig("rating_count.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="rating", y="price", order=rating_order, palette="Set2")
plt.title("Price by Rating")
plt.savefig("price_by_rating.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df[["price", "rating_num"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ EDA complete. Charts saved.")
