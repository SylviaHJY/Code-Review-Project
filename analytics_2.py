import pandas as pd
from collections import Counter
import spacy
import matplotlib.pyplot as plt

# 加载英文模型
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

df = pd.read_csv('clustered_code_reviews_Safe.csv', encoding='utf-8')

# 显示前几行以检查数据
#print(df.head())

# 计算每个聚类的代码审查数量
cluster_counts = df['cluster'].value_counts()

# 计算每个聚类的百分比
cluster_percentages = df['cluster'].value_counts(normalize=True) * 100

# 打印结果
print("聚类计数:\n", cluster_counts)
print("\n聚类百分比:\n", cluster_percentages)

cluster_word_counts = {}

for clust in range(5):  # 5个聚类
    # 选择当前聚类的数据
    cluster_data = df[df['cluster'] == clust]

    # 合并所有文本数据
    all_text = ' '.join(cluster_data['subject'] + ' ' + cluster_data['description'])

    # 分词并移除停用词
    words = [word for word in all_text.split() if word.lower() not in stopwords]

    # 计算词频
    word_counts = Counter(words)

    # 取最常见的前20个词汇
    cluster_word_counts[clust] = word_counts.most_common(10)

# 显示每个聚类的最常见的前10个词汇
print(cluster_word_counts)

# 可视化每个聚类的最常见的前10个词汇
for clust in range(5):
    words, counts = zip(*cluster_word_counts[clust])
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.title(f'Cluster {clust} - Top 10 Common Words')
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.show()