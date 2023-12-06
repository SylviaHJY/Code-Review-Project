import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import spacy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 加载spaCy的英文模型
nlp = spacy.load("en_core_web_sm")

# 读取数据
df = pd.read_csv('Code_Review_Project.csv')
df['combined_text'] = df['subject'] + ' ' + df['description']

# 数据预处理
def preprocess(text):
    # 使用spaCy进行分词和词干提取
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return ' '.join(tokens)

df['processed_text'] = df['combined_text'].apply(preprocess)

# 特征提取：TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['processed_text'])

# K-means聚类
k = 5  # 聚类的数量
model = KMeans(n_clusters=k, random_state=1)
model.fit(X)

# 将聚类结果添加到数据框
df['cluster'] = model.labels_

# 使用PCA进行初步降维到50维（或任何适合你数据集大小的维数）
pca = PCA(n_components=50)
pca_result = pca.fit_transform(X.toarray())

# 在PCA的结果上运行t-SNE
tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, random_state=1)
tsne_result = tsne.fit_transform(pca_result)

# 可视化
plt.figure(figsize=(16,10))
for i in range(k):
    # 选取每个聚类的数据点
    indices = df['cluster'] == i
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=f'Cluster {i}')

plt.legend()
plt.title('t-SNE visualization of PCA-reduced data')
plt.show()

# 保存带有聚类结果的数据框
df.to_csv('clustered_code_reviews_with_pca_tsne.csv', index=False)


# Elbow Method find the best k

# wcss = []
# # 试验不同的k值，比如从1到10
# for i in range(1, 11):
#     kmeans = kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=1)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

# # 绘制肘部法则图
# plt.figure(figsize=(10, 8))
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS') # WCSS是每个点到其聚类中心的距离的平方和
# plt.show()
