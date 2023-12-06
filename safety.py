import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'clustered_code_reviews_Safe.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Define keywords for different quality attributes
performance_keywords = ['performance','speed', 'efficiency', 'latency', 'throughput']
availability_keywords = ['availability','service', 'plugin', 'uptime', 'redundancy', 'reliability']
modifiability_keywords = ['modifiability','modular', 'scalability', 'flexibility', 'maintainability','modify', 
                          'modified','maintain', 'maintained', 'maintaining', 'maintains', 'maintainable', 'maintainer', 
                          'mainta','new', 'remove','removed', 'removes','delete', 'deleted', 'deletes', 'add', 'adds',
                          'adding', 'added', 'remove', 'update', 'updates', 'create', 'fix', 'change', 'changed', 'changes', 
                          'changing', 'improve', 'improved', 'improves', 'improving', 'improvement', 'improvements']

# Function to count occurrences of keywords in a given text
def count_keyword_occurrences(text, keywords):
    return sum(text.lower().count(keyword) for keyword in keywords)

# Analyzing the text data for each quality attribute
data['performance_count'] = data['combined_text'].apply(lambda x: count_keyword_occurrences(x, performance_keywords))
data['availability_count'] = data['combined_text'].apply(lambda x: count_keyword_occurrences(x, availability_keywords))
data['modifiability_count'] = data['combined_text'].apply(lambda x: count_keyword_occurrences(x, modifiability_keywords))

# Grouping the counts by clusters
clustered_counts = data.groupby('cluster').sum()[['performance_count', 'availability_count', 'modifiability_count']]

# Displaying the result
print(clustered_counts)

# Reshape the data for visualization
melted_data = pd.melt(clustered_counts.reset_index(), id_vars=['cluster'], 
                      value_vars=['performance_count', 'availability_count', 'modifiability_count'])

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='variable', y='value', hue='cluster', data=melted_data)
plt.title('Cluster Comparison for Different Quality Attributes')
plt.xlabel('Quality Attributes')
plt.ylabel('Counts')
plt.show()
