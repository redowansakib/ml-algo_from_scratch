import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mypca import myPCA

cancer = load_breast_cancer()
df = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])

data = df.to_numpy().tolist()
scaled_data = StandardScaler().fit_transform(df)

n_components = 3

pca = PCA(n_components=n_components)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

my_pca = myPCA(scaled_data.tolist(), n_components=n_components)

'''signs may be opposite for corresponding column due the reversal of eigenvector.
   Nonetheless the values work as expected'''

print(my_pca)
print(x_pca)



