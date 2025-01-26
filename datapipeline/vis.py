import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingVisualizer:
    def __init__(self, csv_path: str = "C:/Users/jacks/Documents/Life/Projects/Current/Chorono/datapipeline/data/output/projected_vectors_with_metadata.csv"):
        """Initialize the visualizer with path to CSV file containing embeddings"""
        self.csv_path = Path(csv_path)
        self.load_data()
        
    def load_data(self):
        """Load embeddings and metadata from the CSV file"""
        logger.info(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        
        # Parse vector strings to numpy arrays
        logger.info("Parsing vectors from string format...")
        self.embeddings = np.stack([
            np.fromstring(vec_str.strip()[1:-1], sep=",") 
            for vec_str in self.df["vector"].values
        ])
        
        logger.info(f"DataFrame columns: {self.df.columns.tolist()}")
        logger.info(f"Sample of first row: {self.df.iloc[0].to_dict()}")
        logger.info(f"Loaded {len(self.embeddings)} embeddings with shape {self.embeddings.shape}")
        
    def reduce_dimensions(self, n_neighbors=15, min_dist=0.1, n_components=3):
        """Reduce dimensionality of embeddings using UMAP"""
        logger.info("Reducing dimensions with UMAP...")
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                      n_components=n_components, random_state=42)
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        logger.info("Dimension reduction complete")
        
    def create_3d_scatter(self, color_by='language', sample_size=5000):
        """Create an interactive 3D scatter plot"""
        if not hasattr(self, 'reduced_embeddings'):
            self.reduce_dimensions()
            
        # Sample data if needed
        if sample_size and sample_size < len(self.reduced_embeddings):
            indices = np.random.choice(len(self.reduced_embeddings), sample_size, replace=False)
            reduced_sample = self.reduced_embeddings[indices]
            df_sample = self.df.iloc[indices]
        else:
            reduced_sample = self.reduced_embeddings
            df_sample = self.df
            
        # Create figure
        fig = px.scatter_3d(
            x=reduced_sample[:, 0],
            y=reduced_sample[:, 1],
            z=reduced_sample[:, 2],
            color=df_sample[color_by],
            hover_data={
                'word': df_sample['word'],
                'english_translation': df_sample['english_translation'],
                'language': df_sample['language'],
                'time_period': df_sample['time_period'],
                'phonetic_representation': df_sample['phonetic_representation']
            },
            title=f'Word Embeddings Visualization (colored by {color_by})'
        )
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            width=1200,
            height=800
        )
        
        return fig
    
    def create_time_animation(self, sample_size=5000):
        """Create an animated scatter plot showing evolution over time"""
        if not hasattr(self, 'reduced_embeddings'):
            self.reduce_dimensions(n_components=2)  # Use 2D for timeline
            
        # Sample data if needed
        if sample_size and sample_size < len(self.reduced_embeddings):
            indices = np.random.choice(len(self.reduced_embeddings), sample_size, replace=False)
            reduced_sample = self.reduced_embeddings[indices]
            df_sample = self.df.iloc[indices]
        else:
            reduced_sample = self.reduced_embeddings
            df_sample = self.df
            
        # Create figure with frames for each time period
        fig = px.scatter(
            x=reduced_sample[:, 0],
            y=reduced_sample[:, 1],
            animation_frame=df_sample['time_period'],
            color=df_sample['language'],
            hover_data={
                'word': df_sample['word'],
                'english_translation': df_sample['english_translation']
            },
            title='Word Evolution Over Time'
        )
        
        # Update layout
        fig.update_layout(
            width=1000,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def find_similar_words(self, word: str, n: int = 5):
        """Find n most similar words to the given word"""
        # Find the word in our dataset
        word_idx = self.df[self.df['word'] == word].index
        if len(word_idx) == 0:
            logger.warning(f"Word '{word}' not found in dataset")
            return None
            
        word_idx = word_idx[0]
        word_embedding = self.embeddings[word_idx]
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, word_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(word_embedding)
        )
        
        # Get top n similar words (excluding the word itself)
        most_similar_idx = np.argsort(similarities)[-n-1:-1][::-1]
        
        # Create results DataFrame
        results = []
        for idx in most_similar_idx:
            results.append({
                'word': self.df.iloc[idx]['word'],
                'english_translation': self.df.iloc[idx]['english_translation'],
                'language': self.df.iloc[idx]['language'],
                'time_period': self.df.iloc[idx]['time_period'],
                'similarity': similarities[idx]
            })
            
        return pd.DataFrame(results)

def main():
    """Main function to demonstrate visualizations"""
    visualizer = EmbeddingVisualizer()
    
    # Create and save 3D scatter plot
    fig_3d = visualizer.create_3d_scatter()
    fig_3d.write_html("datapipeline/data/output/embeddings_3d.html")
    logger.info("Saved 3D visualization to embeddings_3d.html")
    
    # Create and save time animation
    fig_time = visualizer.create_time_animation()
    fig_time.write_html("datapipeline/data/output/embeddings_timeline.html")
    logger.info("Saved timeline visualization to embeddings_timeline.html")
    
    # Demonstrate similar words
    sample_words = ['bʰer-', 'h₂ster-', 'dʰeh₁-']
    for word in sample_words:
        similar = visualizer.find_similar_words(word)
        if similar is not None:
            print(f"\nWords similar to '{word}':")
            print(similar.to_string(index=False))

if __name__ == "__main__":
    main() 