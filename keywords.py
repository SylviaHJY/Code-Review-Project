import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
file_path = 'clustered_code_reviews_Safe.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Combine 'subject' and 'description' for analysis
combined_text = data['subject'] + " " + data['description']

# Use CountVectorizer to get the most common words
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(combined_text)
word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
most_common_words = word_counts.sum().sort_values(ascending=False).head(60)

# Display the most common words
print(most_common_words)
