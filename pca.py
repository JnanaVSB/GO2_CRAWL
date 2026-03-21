import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA

poses = pd.read_csv("/home/jnana/ARLTask/Go2_crawl/dataset/new_dataset_v1.csv")
labels = poses["label"].values
X = poses.iloc[:,1:].values

print(f"dataset shape {X.shape}")
print(f"labels: {labels}")

pca = PCA()
X_pca = pca.fit_transform(X)

print(f"\n varaince per componenet")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f" PC{i+1}: {var:4f} ({var*100:.1f}%)")

cummulative = np.cumsum(pca.explained_variance_ratio_)
print(f"\n cummulative variance:")
for i, cum in enumerate(cummulative):
    print(f" PC1 - {i+1}: {cum:4f} ({cum*100:.1f}%)")

