import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = (
    "hf://datasets/dhanapalpalanisamy/tourism-dataset/tourism.csv"
)

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded:", df.shape)

# Drop unwanted columns
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Split data
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload back to HF
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="dhanapalpalanisamy/tourism-dataset",
        repo_type="dataset"
    )

print("Train-test data uploaded successfully")
