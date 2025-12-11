import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================
# LOAD FEATURE-ONLY SPLITS
# ============================

X_train = pd.read_csv("train_reduced_pca_gX.csv").values
X_val   = pd.read_csv("validation_reduced_pca_gX.csv").values
X_test  = pd.read_csv("test_reduced_pca_gX.csv").values

print("Shapes before PCA:", X_train.shape, X_val.shape, X_test.shape)


# ============================
# SCALE FEATURES (TRAIN ONLY)
# ============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("Scaling complete.")


# ============================
# APPLY PCA (TRAIN ONLY)
# ============================

pca = PCA(n_components=6)   # or whatever number you chose
pca.fit(X_train_scaled)

X_train_pca = pca.transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("PCA complete.")
print("Shapes after PCA:", X_train_pca.shape, X_val_pca.shape, X_test_pca.shape)
