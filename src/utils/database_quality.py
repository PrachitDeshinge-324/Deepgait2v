import numpy as np
import time
import os
import pickle
import logging
from datetime import datetime, timedelta

class EmbeddingQualityManager:
    """
    Manages embedding quality and consistency in the database
    
    Handles:
    - Consistent quality weighting across ranking & display
    - Database cleanup and maintenance
    - Drift detection and correction
    """
    
    def __init__(self, database_path, quality_threshold=0.4, reindex_interval_days=7):
        self.database_path = database_path
        self.quality_threshold = quality_threshold
        self.reindex_interval = timedelta(days=reindex_interval_days)
        self.last_reindex = None
        self.logger = self._setup_logger()
        
        # Load metadata if exists
        self.metadata_path = os.path.join(os.path.dirname(database_path), 'db_metadata.pkl')
        self._load_metadata()
    
    def _setup_logger(self):
        """Setup logging for database operations"""
        logger = logging.getLogger('embedding_quality')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler('embedding_quality.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_metadata(self):
        """Load database metadata"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                self.last_reindex = self.metadata.get('last_reindex')
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")
                self.metadata = {'last_reindex': None, 'quality_stats': {}}
        else:
            self.metadata = {'last_reindex': None, 'quality_stats': {}}
    
    def _save_metadata(self):
        """Save database metadata"""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def apply_quality_weighting(self, similarities, person_qualities, mode='ranking'):
        """
        Apply consistent quality weighting to similarity scores
        
        Args:
            similarities: Array of similarity scores
            person_qualities: Array of quality scores for each person
            mode: 'ranking' or 'display' (consistent handling for both)
            
        Returns:
            Array of weighted similarity scores
        """
        if len(similarities) == 0 or len(person_qualities) == 0:
            return similarities
        
        # Normalize quality scores to [0.5, 1.0] range
        # This ensures we don't overly penalize but still reflect quality differences
        normalized_qualities = 0.5 + (0.5 * np.array(person_qualities))
        
        # Apply multiplicative weighting - same for both ranking and display
        # This ensures consistency across the system
        weighted_similarities = similarities * normalized_qualities
        
        return weighted_similarities
    
    def check_maintenance_needed(self):
        """
        Check if database maintenance is needed based on time interval or drift metrics
        
        Returns:
            bool: True if maintenance is needed
        """
        now = datetime.now()
        
        # If first time or interval has passed since last reindexing
        if self.last_reindex is None or (now - self.last_reindex) > self.reindex_interval:
            self.logger.info("Database maintenance needed - scheduled interval reached")
            return True
        
        # Could add additional drift detection here
        return False
    
    def perform_maintenance(self, database_obj):
        """
        Perform regular database maintenance
        
        Args:
            database_obj: The database object to maintain
        """
        self.logger.info("Starting database maintenance")
        start_time = time.time()
        
        # 1. Re-evaluate all qualities with current algorithm version
        updated_count = self._recompute_qualities(database_obj)
        
        # 2. Remove low-quality embeddings 
        removed_count = self._cleanup_low_quality(database_obj)
        
        # 3. Analyze quality distribution
        self._analyze_quality_distribution(database_obj)
        
        # 4. Update maintenance timestamp
        self.last_reindex = datetime.now()
        self.metadata['last_reindex'] = self.last_reindex
        self._save_metadata()
        
        elapsed = time.time() - start_time
        self.logger.info(f"Maintenance completed in {elapsed:.1f}s - Updated: {updated_count}, Removed: {removed_count}")
        
        return updated_count, removed_count
    
    def _recompute_qualities(self, database_obj):
        """
        Recompute quality scores for all embeddings using latest algorithm
        
        Args:
            database_obj: Database object containing embeddings
            
        Returns:
            int: Number of updated embeddings
        """
        updated_count = 0
        
        for person_id in database_obj.people:
            person = database_obj.people[person_id]
            if 'embeddings' in person:
                for i, embedding_data in enumerate(person['embeddings']):
                    # Extract embedding's silhouette for quality re-evaluation
                    if 'silhouette' in embedding_data:
                        silhouette = embedding_data['silhouette']
                        # Re-evaluate quality with current algorithm
                        new_quality = database_obj.quality_assessor.evaluate(silhouette)
                        
                        # Update if different
                        if 'quality' not in embedding_data or abs(embedding_data['quality'] - new_quality) > 0.05:
                            embedding_data['quality'] = new_quality
                            updated_count += 1
        
        return updated_count
    
    def _cleanup_low_quality(self, database_obj):
        """
        Remove low-quality embeddings from database
        
        Args:
            database_obj: Database object containing embeddings
            
        Returns:
            int: Number of removed embeddings
        """
        removed_count = 0
        
        for person_id in list(database_obj.people.keys()):
            person = database_obj.people[person_id]
            if 'embeddings' in person and len(person['embeddings']) > 0:
                # Keep track of embeddings to remove
                embeddings_to_keep = []
                
                for embedding_data in person['embeddings']:
                    if embedding_data.get('quality', 0) >= self.quality_threshold:
                        embeddings_to_keep.append(embedding_data)
                    else:
                        removed_count += 1
                
                # Update with filtered list
                if len(embeddings_to_keep) > 0:
                    person['embeddings'] = embeddings_to_keep
                    # Update person's overall quality
                    person['quality'] = np.mean([e.get('quality', 0) for e in embeddings_to_keep])
                else:
                    # Remove person if no good embeddings left
                    del database_obj.people[person_id]
                    removed_count += 1
        
        return removed_count
    
    def _analyze_quality_distribution(self, database_obj):
        """
        Analyze quality distribution across database for monitoring
        
        Args:
            database_obj: Database object containing embeddings
        """
        qualities = []
        
        for person_id in database_obj.people:
            person = database_obj.people[person_id]
            if 'embeddings' in person:
                for embedding_data in person['embeddings']:
                    if 'quality' in embedding_data:
                        qualities.append(embedding_data['quality'])
        
        if not qualities:
            return
            
        # Calculate statistics
        quality_stats = {
            'count': len(qualities),
            'mean': np.mean(qualities),
            'median': np.median(qualities),
            'min': np.min(qualities),
            'max': np.max(qualities),
            'std': np.std(qualities),
            'percentiles': {
                '10': np.percentile(qualities, 10),
                '25': np.percentile(qualities, 25),
                '75': np.percentile(qualities, 75),
                '90': np.percentile(qualities, 90)
            }
        }
        
        # Store in metadata
        self.metadata['quality_stats'] = quality_stats
        self._save_metadata()
        
        # Log summary
        self.logger.info(f"Quality stats: mean={quality_stats['mean']:.2f}, "
                         f"median={quality_stats['median']:.2f}, "
                         f"count={quality_stats['count']}")