from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def preprocess_data(threat1, threat2):
    """
    Converts two threat descriptions into TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    documents = [threat1, threat2]
    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()

    # Convert TF-IDF matrix to a readable format
    tfidf_dataframe = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    print("TF-IDF Matrix:")
    print(tfidf_dataframe)
    
    return tfidf_matrix

def calculate_similarity(threat1, threat2):
    """
    Calculates similarity score between two threats using TF-IDF and cosine similarity.
    """
    # Preprocess data
    tfidf_matrix = preprocess_data(threat1, threat2)
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Convert to percentage
    return similarity_score * 100

def read_from_file(filepath):
    """
    Reads the content of a text file and returns it as a string.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        return content
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None

def main():
    print("Threat Similarity Tool - Read From Files")
    print("=========================================")
    
    # Input file paths
    file1 = input("Enter the path to the first threat description file: ").strip()
    file2 = input("Enter the path to the second threat description file: ").strip()
    
    # Read threat descriptions from the files
    threat1 = read_from_file(file1)
    threat2 = read_from_file(file2)
    
    if threat1 is None or threat2 is None:
        print("Error: Could not read one or both files. Exiting.")
        return
    
    # Calculate similarity
    similarity_percentage = calculate_similarity(threat1, threat2)
    print(f"\nSimilarity Score: {similarity_percentage:.2f}%")

if __name__ == "__main__":
    main()
