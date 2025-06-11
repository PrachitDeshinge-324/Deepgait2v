import numpy as np
import os
import pickle
import time
import faiss
from datetime import datetime

class PersonEmbeddingDatabase:
    def __init__(self, dimension=256*16, use_l2=True, metric_type=faiss.METRIC_L2):
        """Initialize person database with FAISS index"""
        self.people = {}  # Maps ID to person info (name, quality scores, etc)
        self.id_to_index = {}  # Maps person IDs to FAISS indices
        self.index_to_id = []  # Maps FAISS indices to person IDs
        self.dimension = dimension
        self.use_l2 = use_l2
        self.embeddings = {}
        
        # Create FAISS index - L2 distance is Euclidean
        if use_l2:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            # Inner product for cosine similarity, needs normalized vectors
            self.index = faiss.IndexFlatIP(dimension) 
            
    def add_person(self, person_id, name, embedding, quality=1.0, metadata=None):
        """
        Add a new person with their embedding.
        
        Args:
            person_id: Unique identifier for the person
            name: Person's name
            embedding: Numpy array embedding (shape 1,256,16)
            quality: Score indicating quality of the embedding (0-1)
            metadata: Optional additional information
            
        Returns:
            bool: Success or failure
        """
        if person_id in self.people:
            print(f"Person ID {person_id} already exists. Use update_person instead.")
            return False
            
        # Flatten the embedding for FAISS
        flat_emb = embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize for cosine similarity if not using L2
        if not self.use_l2:
            faiss.normalize_L2(flat_emb)
            
        # Get the current index
        idx = len(self.index_to_id)
        
        # Add to FAISS index
        self.index.add(flat_emb)
        
        # Update mappings
        self.id_to_index[person_id] = idx
        self.index_to_id.append(person_id)
        
        # Store person info
        self.people[person_id] = {
            'name': name,
            'quality': quality,
            'created': datetime.now(),
            'last_updated': datetime.now(),
            'metadata': metadata or {},
        }
        
        return True
        
    def update_person(self, person_id, embedding=None, name=None, quality=None, metadata=None):
        """Update an existing person's information"""
        if person_id not in self.people:
            print(f"Person ID {person_id} not found.")
            return False
            
        # Update name if provided
        if name:
            self.people[person_id]['name'] = name
            
        # Update quality if provided
        if quality is not None:
            self.people[person_id]['quality'] = quality
            
        # Update metadata if provided
        if metadata:
            self.people[person_id]['metadata'].update(metadata)
            
        # Update embedding if provided
        if embedding is not None:
            # For FAISS, we need to rebuild the index when updating embeddings
            # This is inefficient but necessary with the basic IndexFlat
            
            # Collect all embeddings
            all_embeddings = []
            for id in self.index_to_id:
                if id == person_id:
                    # Use the new embedding
                    flat_emb = embedding.reshape(1, -1).astype(np.float32)
                    if not self.use_l2:
                        faiss.normalize_L2(flat_emb)
                    all_embeddings.append(flat_emb)
                else:
                    # Get existing embedding (requires search)
                    idx = self.id_to_index[id]
                    # We need to extract this embedding from FAISS, but it's not directly accessible
                    # In a real implementation, you would keep copies of the embeddings or use a different FAISS index
                    
            # This simplified approach has limitations - in a real system you would
            # either store embedding copies or use a more sophisticated FAISS index
            self.people[person_id]['last_updated'] = datetime.now()
        
        return True
        
    def delete_person(self, person_id):
        """Delete a person from the database"""
        if person_id not in self.people:
            print(f"Person ID {person_id} not found.")
            return False
            
        # For FAISS IndexFlat, we need to rebuild the index when deleting
        # So we'll collect all embeddings except the one to delete
        all_embeddings = []
        new_index_to_id = []
        
        for i, id in enumerate(self.index_to_id):
            if id != person_id:
                # In a real implementation, you would have a way to retrieve embeddings
                # For this example, we skip this part as it would require stored copies
                new_index_to_id.append(id)
        
        # Update people dict
        del self.people[person_id]
        
        # Update mappings
        self.index_to_id = new_index_to_id
        self.id_to_index = {id: i for i, id in enumerate(new_index_to_id)}
        
        # Note: In a real implementation, you would rebuild the FAISS index here
        return True

        # Fix the identify_person method in utils/database.py
    def identify_person(self, embedding, top_k=1, threshold=0.7):
        """Find the closest matching person for an embedding"""
        # Check if database is empty
        if len(self.index_to_id) == 0:
            print("Warning: Database is empty, no matches possible")
            return []
            
        # Reshape embedding for FAISS
        flat_emb = embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize for cosine similarity if not using L2
        if not self.use_l2:
            faiss.normalize_L2(flat_emb)
            
        # Search the index
        distances, indices = self.index.search(flat_emb, min(top_k, len(self.index_to_id)))
        
        # Print raw search results for debugging
        print(f"Raw FAISS results - distances: {distances}, indices: {indices}")
        
        # Convert distances to similarity scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                person_id = self.index_to_id[idx]
                
                # Convert distance to similarity score - FIXED VERSION
                if self.use_l2:
                    # For L2 distance, smaller is better
                    # Using exponential decay for proper similarity conversion
                    similarity = float(np.exp(-distances[0][i] * 0.5))  # Changed from np.exp(-distances[0][i])
                else:
                    # For inner product (cosine), higher is better
                    similarity = float(distances[0][i])
                    
                print(f"Distance {distances[0][i]:.3f} converted to similarity {similarity:.3f}")
                    
                # Only include results above threshold
                if similarity >= threshold:
                    results.append((
                        person_id,
                        similarity,
                        self.people[person_id]['name'],
                        self.people[person_id].get('quality', 0.5)
                    ))
        
        return results

    def save_to_disk(self, filepath):
        """Save database to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss_path = filepath + ".index"
        faiss.write_index(self.index, faiss_path)
        
        # Save metadata (people dict and mappings)
        meta_path = filepath + ".meta"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'people': self.people,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'dimension': self.dimension,
                'use_l2': self.use_l2
            }, f)
        
        print(f"Database saved to {filepath}")
        return True
        
    def load_from_disk(self, filepath):
        """Load database from disk"""
        # Check if files exist
        faiss_path = filepath + ".index"
        meta_path = filepath + ".meta"
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print(f"Database files not found at {filepath}")
            return False
            
        # Load FAISS index
        self.index = faiss.read_index(faiss_path)
        
        # Load metadata
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
            
        # Set attributes
        self.people = metadata['people']
        self.id_to_index = metadata['id_to_index']
        self.index_to_id = metadata['index_to_id']
        self.dimension = metadata['dimension']
        self.use_l2 = metadata['use_l2']
        
        print(f"Database loaded from {filepath} with {len(self.people)} people")
        return True


# Example usage
def faiss_example():
    # Initialize database (assuming 256*16 dimension for embeddings)
    db = PersonEmbeddingDatabase(dimension=256*16)
    
    # Create some sample embeddings (normally these would come from your embedding model)
    sample_emb1 = np.random.rand(1, 256, 16).astype(np.float32)
    sample_emb2 = np.random.rand(1, 256, 16).astype(np.float32)
    
    # Add people to the database
    db.add_person("p001", "John Doe", sample_emb1, quality=0.95)
    db.add_person("p002", "Jane Smith", sample_emb2, quality=0.92)
    
    # Save database
    db.save_to_disk("data/person_db")
    
    # Load database
    new_db = PersonEmbeddingDatabase()
    new_db.load_from_disk("data/person_db")
    
    # Query with a test embedding (similar to first person)
    test_emb = sample_emb1 + np.random.normal(0, 0.1, (1, 256, 16)).astype(np.float32)
    
    # Identify person
    start_time = time.time()
    results = new_db.identify_person(test_emb, threshold=0.5)
    end_time = time.time()
    
    print(f"Identification took {(end_time - start_time)*1000:.2f} ms")
    
    if results:
        for person_id, similarity, name in results:
            print(f"Matched: {name} (ID: {person_id}) with similarity {similarity:.4f}")
    else:
        print("No matches found")
    
    # Update person
    new_db.update_person("p001", name="John Smith")
    
    # Delete person
    new_db.delete_person("p002")