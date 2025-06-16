import numpy as np
import os
import pickle
import time
import faiss
from datetime import datetime

class PersonEmbeddingDatabase:
    def __init__(self, dimension=256*16, use_cosine=True, metric_type=faiss.METRIC_L2):
        """Initialize person database with FAISS index"""
        self.people = {}  # Maps ID to person info (name, quality scores, etc)
        self.id_to_index = {}  # Maps person IDs to FAISS indices
        self.index_to_id = []  # Maps FAISS indices to person ID
        self.dimension = dimension
        self.use_cosine = use_cosine  # Changed from use_l2 to use_cosine for clarity
        self.embeddings = {}
        
        # Performance enhancements
        self.embedding_cache = {}  # Cache for frequently accessed embeddings
        self.search_cache = {}     # Cache for recent search results
        self.cache_max_size = 1000
        self.index_type = 'FlatIP' if use_cosine else 'FlatL2'  # Track index type
        
        # Create FAISS index - use cosine similarity for better gait recognition
        if use_cosine:
            # Inner product for cosine similarity, needs normalized vectors
            self.index = faiss.IndexFlatIP(dimension)
        else:
            # L2 distance is Euclidean
            self.index = faiss.IndexFlatL2(dimension)
            
        # Use GPU if available for larger databases (check after index creation)
        self._enable_gpu_if_available()
                
    def _enable_gpu_if_available(self):
        """Enable GPU acceleration if available and beneficial"""
        try:
            if len(self.index_to_id) > 1000 and faiss.get_num_gpus() > 0:
                gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
                print("Using GPU acceleration for FAISS")
        except Exception as e:
            print(f"GPU acceleration failed, using CPU: {e}")
    
    def _hash_embedding(self, embedding):
        """Create a hash of the embedding for caching"""
        return hash(embedding.tobytes())
    
    def _get_embedding_from_cache(self, person_id):
        """Get embedding from cache if available"""
        return self.embedding_cache.get(person_id)
    
    def _cache_embedding(self, person_id, embedding):
        """Cache embedding for faster access"""
        if len(self.embedding_cache) >= self.cache_max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[person_id] = embedding.copy()
    
    def _get_search_cache_key(self, embedding, top_k, threshold):
        """Generate cache key for search results"""
        # Use hash of embedding for cache key
        emb_hash = self._hash_embedding(embedding)
        return f"{emb_hash}_{top_k}_{threshold}"
    
    def _cache_search_results(self, cache_key, results):
        """Cache search results"""
        if len(self.search_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = results.copy()
    
    def _get_cached_search_results(self, cache_key):
        """Get cached search results if available"""
        return self.search_cache.get(cache_key)
    
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
        
        # Normalize for cosine similarity
        if self.use_cosine:
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
        
        # Cache the embedding
        self._cache_embedding(person_id, embedding)
        
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
                    if self.use_cosine:
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

    def identify_person(self, embedding, top_k=1, threshold=None):
        """
        Enhanced person identification with improved similarity calculation and adaptive thresholds
        
        Args:
            embedding: Input embedding for identification
            top_k: Number of top matches to return
            threshold: Similarity threshold (if None, uses adaptive threshold)
            
        Returns:
            List of tuples: (person_id, similarity, name, quality)
        """
        # Check if database is empty
        if len(self.index_to_id) == 0:
            print("Warning: Database is empty, no matches possible")
            return []
            
        # Reshape embedding for FAISS
        flat_emb = embedding.reshape(1, -1).astype(np.float32)
        
        # Check cache first
        cache_key = self._get_search_cache_key(flat_emb, top_k, threshold)
        cached_results = self._get_cached_search_results(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Normalize for cosine similarity
        if self.use_cosine:
            faiss.normalize_L2(flat_emb)
            
        # Search the index with more candidates for better adaptive threshold calculation
        search_k = min(max(10, top_k * 3), len(self.index_to_id))
        distances, indices = self.index.search(flat_emb, search_k)
        
        # Calculate adaptive threshold if not provided
        if threshold is None:
            threshold = self._calculate_adaptive_threshold(distances[0])
        
        # Print raw search results for debugging
        print(f"Raw FAISS results - distances: {distances[0][:3]}, indices: {indices[0][:3]}")
        print(f"Using threshold: {threshold:.3f}")
        
        # Convert distances to similarity scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                person_id = self.index_to_id[idx]
                
                # Enhanced similarity calculation
                if self.use_cosine:
                    # For FAISS IndexFlatIP, distances are dot products (cosine similarity)
                    # For normalized vectors, this is the cosine similarity directly
                    raw_distance = float(distances[0][i])
                    
                    # Cosine similarity ranges from -1 to +1
                    # We need to map this to a 0-1 range for threshold comparison
                    # Using (cosine + 1) / 2 to map [-1,1] to [0,1]
                    if raw_distance >= -1.0 and raw_distance <= 1.0:
                        similarity = float((raw_distance + 1.0) / 2.0)
                    else:
                        # If outside expected range, clip it
                        clipped = max(-1.0, min(1.0, raw_distance))
                        similarity = float((clipped + 1.0) / 2.0)
                else:
                    # L2 distance: convert to similarity with adaptive scaling
                    distance = distances[0][i]
                    normalized_dist = distance / np.sqrt(self.dimension)
                    similarity = float(np.exp(-0.5 * normalized_dist**2))
                
                # Only include results above threshold
                if similarity >= threshold:
                    person_quality = self.people[person_id].get('quality', 0.5)
                    
                    # Enhanced quality-based adjustment
                    quality_boost = 0.1 * np.log(1 + person_quality)
                    adjusted_similarity = min(1.0, similarity + quality_boost)
                    
                    results.append((
                        person_id,
                        adjusted_similarity,
                        self.people[person_id]['name'],
                        person_quality
                    ))
        
        # Sort by adjusted similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        final_results = results[:top_k]
        
        # Cache the results for future queries
        self._cache_search_results(cache_key, final_results)
        
        return final_results

    def identify_person_nucleus(self, embedding, top_p=0.9, min_candidates=1, max_candidates=10, threshold=None, 
                               close_sim_threshold=0.05, amplification_factor=20.0, quality_weight=0.5, enhanced_ranking=True):
        """
        Enhanced person identification using nucleus (top-p) sampling with improved handling of close similarities
        
        Args:
            embedding: Input embedding for identification
            top_p: Cumulative probability mass to include (e.g., 0.9 = top 90% probability mass)
            min_candidates: Minimum number of candidates to return
            max_candidates: Maximum number of candidates to return
            threshold: Similarity threshold (if None, uses adaptive threshold)
            close_sim_threshold: Range below which similarities are considered "close"
            amplification_factor: Factor to amplify small differences in close similarities
            quality_weight: Weight for quality-based tie breaking
            enhanced_ranking: Whether to use multi-factor ranking for close similarities
            
        Returns:
            List of tuples: (person_id, similarity, name, quality)
        """
        # Check if database is empty
        if len(self.index_to_id) == 0:
            print("Warning: Database is empty, no matches possible")
            return []
            
        # Reshape embedding for FAISS
        flat_emb = embedding.reshape(1, -1).astype(np.float32)
        
        # Generate cache key for nucleus sampling
        cache_key = f"nucleus_{self._hash_embedding(flat_emb)}_{top_p}_{threshold}"
        cached_results = self._get_cached_search_results(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Normalize for cosine similarity
        if self.use_cosine:
            faiss.normalize_L2(flat_emb)
            
        # Search all candidates for nucleus sampling (search more candidates for better distribution)
        search_k = min(len(self.index_to_id), 50)  # Search more candidates for better distribution
        distances, indices = self.index.search(flat_emb, search_k)
        
        # Calculate adaptive threshold if not provided
        if threshold is None:
            threshold = self._calculate_adaptive_threshold(distances[0])
        
        print(f"Nucleus sampling - Raw FAISS results: {distances[0][:5]}")
        print(f"Using threshold: {threshold:.3f}, top_p: {top_p}")
        
        # Convert distances to similarity scores and normalize
        similarities = []
        valid_candidates = []
        
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                person_id = self.index_to_id[idx]
                
                # Enhanced similarity calculation for nucleus sampling
                if self.use_cosine:
                    raw_distance = float(distances[0][i])
                    # Keep raw cosine similarity for better discrimination
                    # Don't map to [0,1] range here - preserve original values
                    raw_similarity = raw_distance
                    
                    # For threshold comparison, still use [0,1] mapping
                    threshold_similarity = (raw_distance + 1.0) / 2.0 if raw_distance >= -1.0 and raw_distance <= 1.0 else 0.5
                else:
                    # L2 distance: convert to similarity
                    distance = distances[0][i]
                    normalized_dist = distance / np.sqrt(self.dimension)
                    raw_similarity = float(np.exp(-0.5 * normalized_dist**2))
                    threshold_similarity = raw_similarity
                
                # Only include results above threshold (using threshold-comparable similarity)
                if threshold_similarity >= threshold:
                    person_quality = self.people[person_id].get('quality', 0.5)
                    
                    # Store raw similarity for better discrimination
                    similarities.append(raw_similarity)
                    valid_candidates.append((
                        person_id,
                        raw_similarity,  # Use raw similarity, not adjusted
                        self.people[person_id]['name'],
                        person_quality
                    ))
        
        if not valid_candidates:
            return []
        
        # Sort by similarity (highest first)
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        similarities = [x[1] for x in valid_candidates]
        
        # Enhanced probability distribution for close similarity scores
        similarities_array = np.array(similarities)
        
        # Detect if similarities are too close (variance is low)
        sim_variance = np.var(similarities_array)
        sim_range = np.max(similarities_array) - np.min(similarities_array)
        
        print(f"Similarity variance: {sim_variance:.6f}, range: {sim_range:.6f}")
        print(f"Close similarity detection: variance < 0.0001? {sim_variance < 0.0001}, range < {close_sim_threshold}? {sim_range < close_sim_threshold}")
        
        # Enhanced normalization strategies for close similarities
        if sim_variance < 0.001 or sim_range < close_sim_threshold:  # Relaxed threshold for close similarities
            print("Close similarities detected - using enhanced discrimination")
            
            # Strategy 1: Use exponential amplification of differences
            amplified_similarities = np.exp(amplification_factor * (similarities_array - np.min(similarities_array)))
            
            # Strategy 2: Add quality-based tie-breaking
            quality_scores = np.array([candidate[3] for candidate in valid_candidates])  # Extract quality scores
            quality_weights = 1.0 + quality_weight * quality_scores  # Boost based on quality
            
            # Combine amplified similarities with quality weights
            combined_scores = amplified_similarities * quality_weights
            
            # Strategy 3: Add small random noise for tie-breaking (reproducible)
            np.random.seed(hash(str(similarities_array.tobytes())) % 2**32)  # Reproducible randomness
            noise = np.random.normal(0, 0.01, len(combined_scores))
            final_scores = combined_scores + noise
            
            # Normalize to probabilities
            probabilities = final_scores / np.sum(final_scores)
            
        else:  # Normal case - similarities are well separated
            print("Well-separated similarities - using standard softmax")
            
            # Adaptive temperature based on similarity spread
            if sim_range > 0.1:
                temperature = 1.0  # Standard temperature for well-separated values
            else:
                temperature = 2.0  # Higher temperature for moderately close values
            
            # Apply temperature scaling
            scaled_similarities = similarities_array / temperature
            
            # Compute softmax probabilities
            exp_similarities = np.exp(scaled_similarities - np.max(scaled_similarities))
            probabilities = exp_similarities / np.sum(exp_similarities)
        
        # Enhanced nucleus sampling with adaptive cutoff
        cumulative_probs = np.cumsum(probabilities)
        
        # Dynamic top_p adjustment based on distribution flatness
        effective_top_p = top_p
        if sim_variance < 0.0001:  # Very flat distribution
            effective_top_p = min(0.95, top_p + 0.1)  # Be more inclusive
            print(f"Adjusted top_p from {top_p} to {effective_top_p} for flat distribution")
        
        # Find the cutoff index where cumulative probability exceeds effective_top_p
        nucleus_cutoff = np.searchsorted(cumulative_probs, effective_top_p) + 1
        nucleus_cutoff = max(min_candidates, min(nucleus_cutoff, max_candidates, len(valid_candidates)))
        
        # Select nucleus candidates
        nucleus_candidates = valid_candidates[:nucleus_cutoff]
        
        # Convert raw similarities back to [0,1] range for display consistency
        display_candidates = []
        for person_id, raw_sim, name, quality in nucleus_candidates:
            if self.use_cosine:
                # Map cosine similarity [-1,1] to [0,1] for display
                display_sim = (raw_sim + 1.0) / 2.0 if raw_sim >= -1.0 else 0.5
                
                # Apply quality boost for display
                person_quality = self.people[person_id].get('quality', 0.5)
                quality_boost = 0.1 * np.log(1 + person_quality)
                display_sim = min(1.0, display_sim + quality_boost)
            else:
                display_sim = raw_sim
                
            display_candidates.append((person_id, display_sim, name, quality))
        
        print(f"Nucleus sampling selected {len(display_candidates)} candidates (effective_top_p={effective_top_p:.3f})")
        print(f"Raw similarity scores: {[f'{c[1]:.6f}' for c in nucleus_candidates[:5]]}")
        print(f"Display similarity scores: {[f'{c[1]:.6f}' for c in display_candidates[:5]]}")
        print(f"Probability distribution: {[f'{p:.4f}' for p in probabilities[:nucleus_cutoff]]}")
        
        # Additional ranking based on multiple factors for close similarities
        if enhanced_ranking and (sim_variance < 0.001 or sim_range < close_sim_threshold) and len(display_candidates) > 1:
            print("Applying secondary ranking for close similarities")
            
            # Secondary ranking factors
            enhanced_candidates = []
            for i, (person_id, display_sim, name, quality) in enumerate(display_candidates):
                # Factor 1: Original similarity (40% weight)
                sim_score = display_sim * 0.4
                
                # Factor 2: Quality score (30% weight)  
                quality_score = quality * 0.3
                
                # Factor 3: Probability from distribution (20% weight)
                prob_score = probabilities[i] * 0.2
                
                # Factor 4: Recency/frequency bonus (10% weight)
                person_data = self.people[person_id]
                embedding_count = len(person_data.get('embeddings', []))
                recency_score = min(1.0, embedding_count / 10.0) * 0.1
                
                # Combine all factors
                final_score = sim_score + quality_score + prob_score + recency_score
                
                enhanced_candidates.append((
                    person_id, display_sim, name, quality, final_score
                ))
            
            # Re-sort by enhanced score
            enhanced_candidates.sort(key=lambda x: x[4], reverse=True)
            
            # Convert back to original format
            display_candidates = [(pid, sim, name, qual) for pid, sim, name, qual, _ in enhanced_candidates]
            
            print(f"Enhanced ranking applied - final order: {[c[2] for c in display_candidates[:3]]}")
        
        # Cache the results
        self._cache_search_results(cache_key, display_candidates)
        
        return display_candidates

    def identify_person_adaptive(self, embedding, method='nucleus', **kwargs):
        """
        Adaptive person identification that can use either top-k or nucleus sampling
        
        Args:
            embedding: Input embedding for identification
            method: 'top_k' or 'nucleus'
            **kwargs: Arguments for the chosen method
            
        Returns:
            List of tuples: (person_id, similarity, name, quality)
        """
        if method == 'nucleus':
            # Default nucleus parameters if not provided
            top_p = kwargs.get('top_p', 0.9)
            min_candidates = kwargs.get('min_candidates', 1)
            max_candidates = kwargs.get('max_candidates', 10)
            threshold = kwargs.get('threshold', None)
            close_sim_threshold = kwargs.get('close_sim_threshold', 0.05)
            amplification_factor = kwargs.get('amplification_factor', 20.0)
            quality_weight = kwargs.get('quality_weight', 0.5)
            enhanced_ranking = kwargs.get('enhanced_ranking', True)
            
            return self.identify_person_nucleus(
                embedding, top_p=top_p, 
                min_candidates=min_candidates, 
                max_candidates=max_candidates, 
                threshold=threshold,
                close_sim_threshold=close_sim_threshold,
                amplification_factor=amplification_factor,
                quality_weight=quality_weight,
                enhanced_ranking=enhanced_ranking
            )
        else:  # top_k
            top_k = kwargs.get('top_k', 1)
            threshold = kwargs.get('threshold', None)
            
            return self.identify_person(embedding, top_k=top_k, threshold=threshold)
    
    
    def _calculate_adaptive_threshold(self, distances):
        """
        Enhanced adaptive threshold calculation based on database statistics and distribution
        
        Args:
            distances: Array of distances from FAISS search
            
        Returns:
            float: Adaptive threshold
        """
        if len(distances) == 0:
            return 0.5
            
        valid_distances = distances[distances != float('inf')]
        
        if len(valid_distances) == 0:
            return 0.5
            
        if self.use_cosine:
            # For cosine similarity with IndexFlatIP, distances are dot products
            # We need to convert them to [0,1] range for threshold calculation
            valid_distances = distances[distances >= -1.0]  # Valid cosine range
            
            if len(valid_distances) == 0:
                return 0.1  # Conservative threshold when no valid distances
                
            # Convert cosine similarities to [0,1] range
            converted_similarities = (valid_distances + 1.0) / 2.0
            
            mean_sim = np.mean(converted_similarities)
            std_sim = np.std(converted_similarities)
            
            # For cosine similarity, we want a threshold that's lower than the mean
            # to catch similar embeddings
            db_size_factor = min(1.0, len(self.index_to_id) / 100.0)
            
            # More permissive threshold calculation for real-world scenarios  
            base_threshold = max(0.02, mean_sim - 3 * std_sim)  # Use 3*std for very permissive matching
            
            # Cap the threshold to very permissive bounds for CCTV scenarios
            return max(0.01, min(0.6, base_threshold))
        else:
            # For L2 distance, convert to similarity first
            scale_factor = 1.0 / np.sqrt(self.dimension)
            similarities = np.exp(-valid_distances * scale_factor)
            
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            
            # Adaptive threshold
            db_size_factor = min(1.0, len(self.index_to_id) / 100.0)
            base_threshold = max(0.3, mean_sim - std_sim * (1 + db_size_factor))
            
            return min(0.8, base_threshold)
    
    def _calculate_confidence(self, distances, current_idx):
        """
        Calculate confidence based on distance gap to next best match
        
        Args:
            distances: Array of distances
            current_idx: Index of current match
            
        Returns:
            float: Confidence score (0-1)
        """
        if len(distances) <= 1 or current_idx >= len(distances) - 1:
            return 0.5  # Default confidence
        
        current_dist = distances[current_idx]
        next_dist = distances[current_idx + 1]
        
        # Calculate relative gap
        if self.use_cosine:
            # For cosine similarity, higher is better
            gap = next_dist - current_dist if current_dist > next_dist else 0
            confidence = min(1.0, gap * 10)  # Scale gap to confidence
        else:
            # For L2 distance, lower is better
            gap = next_dist - current_dist
            confidence = min(1.0, gap / max(current_dist, 0.1))
        
        return max(0.1, confidence)  # Minimum confidence of 0.1
    
    def update_distance_statistics(self):
        """Update distance statistics for better similarity calculation"""
        if len(self.index_to_id) < 2:
            return
            
        # Sample some embeddings to calculate statistics
        sample_size = min(50, len(self.index_to_id))
        sample_indices = np.random.choice(len(self.index_to_id), sample_size, replace=False)
        
        all_distances = []
        for idx in sample_indices:
            # Get embedding (this is a simplified approach - would need actual embedding storage)
            # For now, we'll estimate based on recent search results
            pass
        
        # This would be implemented with actual embedding storage
        # For now, set default values
        self._distance_stats = (0.5, 0.2)  # mean, std

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
                'use_cosine': self.use_cosine  # Changed from use_l2
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
        # Handle backward compatibility
        self.use_cosine = metadata.get('use_cosine', not metadata.get('use_l2', True))
        
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

    def identify_person_ensemble(self, embeddings_list, top_k=1, threshold=None, ensemble_method='weighted_average'):
        """
        Enhanced ensemble identification using multiple embeddings with improved methods
        
        Args:
            embeddings_list: List of embedding arrays
            top_k: Number of top matches to return
            threshold: Similarity threshold
            ensemble_method: 'weighted_average', 'majority_vote', 'max_confidence', or 'adaptive_ensemble'
            
        Returns:
            List of tuples: (person_id, similarity, name, quality, confidence)
        """
        if not embeddings_list:
            return []
            
        # Handle single embedding case
        if len(embeddings_list) == 1:
            return self.identify_person(embeddings_list[0], top_k, threshold)
        
        if ensemble_method == 'weighted_average':
            return self._ensemble_weighted_average(embeddings_list, top_k, threshold)
        elif ensemble_method == 'majority_vote':
            return self._ensemble_majority_vote(embeddings_list, top_k, threshold)
        elif ensemble_method == 'max_confidence':
            return self._ensemble_max_confidence(embeddings_list, top_k, threshold)
        elif ensemble_method == 'adaptive_ensemble':
            return self._ensemble_adaptive(embeddings_list, top_k, threshold)
        else:
            # Fallback to single embedding
            return self.identify_person(embeddings_list[0], top_k, threshold)
    
    def _ensemble_weighted_average(self, embeddings_list, top_k, threshold):
        """Enhanced weighted average ensemble with quality-based weighting"""
        # Calculate embedding quality weights
        weights = []
        for emb in embeddings_list:
            # Estimate embedding quality based on norm and distribution
            norm = np.linalg.norm(emb)
            quality = min(1.0, norm / np.sqrt(self.dimension))  # Normalize by expected norm
            weights.append(quality)
        
        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(embeddings_list)) / len(embeddings_list)
        
        # Create weighted average embedding
        avg_embedding = np.zeros_like(embeddings_list[0])
        for i, emb in enumerate(embeddings_list):
            avg_embedding += weights[i] * emb
            
        return self.identify_person(avg_embedding, top_k, threshold)
    
    def _ensemble_majority_vote(self, embeddings_list, top_k, threshold):
        """Enhanced majority vote with confidence weighting"""
        all_results = []
        embedding_confidences = []
        
        # Get results from each embedding with confidence tracking
        for i, emb in enumerate(embeddings_list):
            results = self.identify_person(emb, top_k=min(10, len(self.index_to_id)), threshold=threshold)
            
            # Calculate embedding confidence based on top match quality
            emb_confidence = results[0][4] if results and len(results[0]) > 4 else 0.5
            embedding_confidences.append(emb_confidence)
            
            # Add embedding index to results for weighting
            for result in results:
                all_results.append(result + (i, emb_confidence))
        
        # Aggregate votes with confidence weighting
        vote_data = {}  # person_id -> {'votes': [], 'similarities': [], 'confidences': []}
        
        for result in all_results:
            person_id = result[0]
            similarity = result[1]
            confidence = result[5] if len(result) > 5 else 0.5
            
            if person_id not in vote_data:
                vote_data[person_id] = {'votes': [], 'similarities': [], 'confidences': []}
            
            vote_data[person_id]['votes'].append(1)
            vote_data[person_id]['similarities'].append(similarity)
            vote_data[person_id]['confidences'].append(confidence)
        
        # Calculate final scores
        final_results = []
        for person_id, data in vote_data.items():
            vote_count = len(data['votes'])
            avg_similarity = np.mean(data['similarities'])
            avg_confidence = np.mean(data['confidences'])
            
            # Weighted score considering votes, similarity, and confidence
            vote_strength = vote_count / len(embeddings_list)
            final_score = (avg_similarity * 0.6 + 
                          vote_strength * 0.3 + 
                          avg_confidence * 0.1)
            
            person_info = self.people[person_id]
            final_results.append((
                person_id, 
                final_score, 
                person_info['name'], 
                person_info.get('quality', 0.5),
                avg_confidence
            ))
        
        # Sort by final score
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:top_k]
    
    def _ensemble_max_confidence(self, embeddings_list, top_k, threshold):
        """Enhanced max confidence ensemble with fallback"""
        best_results = []
        best_confidence = 0
        all_results = []
        
        for emb in embeddings_list:
            results = self.identify_person(emb, top_k, threshold)
            all_results.extend(results)
            
            if results:
                current_confidence = results[0][4] if len(results[0]) > 4 else results[0][1]
                if current_confidence > best_confidence:
                    best_confidence = current_confidence
                    best_results = results
        
        # If no single embedding gives high confidence, fall back to majority vote
        if best_confidence < 0.6:
            return self._ensemble_majority_vote(embeddings_list, top_k, threshold)
        
        return best_results
    
    def _ensemble_adaptive(self, embeddings_list, top_k, threshold):
        """Adaptive ensemble that chooses best method based on embedding characteristics"""
        # Analyze embedding consistency
        if len(embeddings_list) < 3:
            return self._ensemble_weighted_average(embeddings_list, top_k, threshold)
        
        # Calculate pairwise similarities between embeddings
        similarities = []
        for i in range(len(embeddings_list)):
            for j in range(i+1, len(embeddings_list)):
                if self.use_cosine:
                    # Cosine similarity between embeddings
                    emb1 = embeddings_list[i].reshape(-1)
                    emb2 = embeddings_list[j].reshape(-1)
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(sim)
                else:
                    # L2 distance converted to similarity
                    dist = np.linalg.norm(embeddings_list[i] - embeddings_list[j])
                    sim = np.exp(-dist / np.sqrt(self.dimension))
                    similarities.append(sim)
        
        avg_consistency = np.mean(similarities)
        
        # Choose method based on consistency
        if avg_consistency > 0.8:
            # High consistency: use weighted average
            return self._ensemble_weighted_average(embeddings_list, top_k, threshold)
        elif avg_consistency > 0.5:
            # Medium consistency: use majority vote
            return self._ensemble_majority_vote(embeddings_list, top_k, threshold)
        else:
            # Low consistency: use max confidence
            return self._ensemble_max_confidence(embeddings_list, top_k, threshold)
    
    def _cache_embedding_query(self, embedding_hash, results):
        """Cache query results for performance"""
        if not self.enable_caching:
            return
            
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[embedding_hash] = results
    
    def _get_cached_query(self, embedding_hash):
        """Get cached query results"""
        if not self.enable_caching:
            return None
        return self.query_cache.get(embedding_hash)
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        print("Query cache cleared")