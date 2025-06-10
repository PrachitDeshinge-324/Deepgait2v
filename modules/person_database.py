"""
Person Database Manager for Gait Recognition
Manages distinct embeddings database for each individual with quality control
"""

import os
import json
import pickle
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import hashlib

class PersonEmbeddingDatabase:
    """Database manager for person gait embeddings with quality control"""
    
    def __init__(self, db_path: str = "person_embeddings.db", config: Dict = None):
        """
        Initialize the person embedding database
        
        Args:
            db_path (str): Path to the database file
            config (dict): Configuration parameters
        """
        self.db_path = db_path
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # In-memory cache for fast access
        self.embedding_cache = {}
        self.quality_cache = {}
        
    def _get_default_config(self) -> Dict:
        """Default configuration for database management"""
        return {
            'max_embeddings_per_person': 10,    # Maximum embeddings to store per person
            'min_quality_threshold': 0.5,       # Minimum quality score to store
            'similarity_threshold': 0.85,       # Threshold for considering embeddings similar
            'clustering_eps': 0.15,             # DBSCAN epsilon for clustering
            'clustering_min_samples': 2,        # DBSCAN minimum samples
            'embedding_dimension': None,        # Will be set automatically
            'auto_cleanup_threshold': 100,      # Auto cleanup when database exceeds this size
            'duplicate_detection': True,        # Enable duplicate detection
            'quality_decay_factor': 0.95,      # Quality decay for old embeddings
            'max_age_days': 30,                # Maximum age for embeddings (days)
        }
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create persons table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_embeddings INTEGER DEFAULT 0,
                    average_quality REAL DEFAULT 0.0,
                    metadata TEXT
                )
            ''')
            
            # Create embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    person_id TEXT,
                    embedding_data BLOB,
                    quality_score REAL,
                    quality_metrics TEXT,
                    sequence_length INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_info TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    cluster_id INTEGER DEFAULT -1,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id)
                )
            ''')
            
            # Create recognition history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_history (
                    recognition_id TEXT PRIMARY KEY,
                    person_id TEXT,
                    confidence_score REAL,
                    embedding_id TEXT,
                    recognized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id),
                    FOREIGN KEY (embedding_id) REFERENCES embeddings (embedding_id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_person_id ON embeddings (person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_quality ON embeddings (quality_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_active ON embeddings (is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recognition_person_id ON recognition_history (person_id)')
            
            conn.commit()
            conn.close()
            self.logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def add_person_embedding(self, person_id: str, embedding: np.ndarray, 
                           quality_result: Dict, sequence_length: int = 0,
                           source_info: str = "", person_name: str = None) -> bool:
        """
        Add a new embedding for a person with quality control
        
        Args:
            person_id (str): Unique identifier for the person
            embedding (np.ndarray): Gait embedding vector
            quality_result (Dict): Quality assessment result
            sequence_length (int): Length of the source sequence
            source_info (str): Information about the source (e.g., video file, camera)
            person_name (str): Optional human-readable name for the person
            
        Returns:
            bool: True if embedding was added successfully
        """
        try:
            quality_score = quality_result.get('overall_score', 0.0)
            
            # Check quality threshold
            if quality_score < self.config['min_quality_threshold']:
                self.logger.info(f"Embedding rejected for person {person_id} due to low quality: {quality_score}")
                return False
            
            # Check for duplicates if enabled
            if self.config['duplicate_detection']:
                if self._is_duplicate_embedding(person_id, embedding):
                    self.logger.info(f"Duplicate embedding detected for person {person_id}")
                    return False
            
            # Set embedding dimension if not set
            if self.config.get('embedding_dimension') is None:
                self.config['embedding_dimension'] = embedding.shape[-1]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create or update person record
            self._upsert_person(cursor, person_id, person_name)
            
            # Generate embedding ID
            embedding_id = self._generate_embedding_id(person_id, embedding, quality_score)
            
            # Serialize embedding and quality data
            embedding_blob = pickle.dumps(embedding)
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, (np.integer, np.int_)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float_)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            quality_json = json.dumps(convert_numpy_types(quality_result))
            source_json = json.dumps({'source': source_info, 'added_at': datetime.now().isoformat()})
            
            # Insert embedding
            cursor.execute('''
                INSERT INTO embeddings 
                (embedding_id, person_id, embedding_data, quality_score, quality_metrics, 
                 sequence_length, source_info)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (embedding_id, person_id, embedding_blob, quality_score, quality_json, 
                  sequence_length, source_json))
            
            # Update person statistics
            self._update_person_stats(cursor, person_id)
            
            # Manage embedding count per person
            self._manage_person_embeddings(cursor, person_id)
            
            conn.commit()
            conn.close()
            
            # Update cache
            self._update_cache(person_id)
            
            self.logger.info(f"Added embedding for person {person_id} with quality {quality_score:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding embedding for person {person_id}: {str(e)}")
            return False
    
    def get_person_embeddings(self, person_id: str, active_only: bool = True) -> List[Dict]:
        """
        Get all embeddings for a specific person
        
        Args:
            person_id (str): Person identifier
            active_only (bool): Only return active embeddings
            
        Returns:
            List[Dict]: List of embedding records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT embedding_id, embedding_data, quality_score, quality_metrics,
                       sequence_length, created_at, source_info, cluster_id
                FROM embeddings 
                WHERE person_id = ?
            '''
            params = [person_id]
            
            if active_only:
                query += ' AND is_active = 1'
            
            query += ' ORDER BY quality_score DESC, created_at DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            embeddings = []
            for row in rows:
                embedding_data = pickle.loads(row[1])
                quality_metrics = json.loads(row[3]) if row[3] else {}
                source_info = json.loads(row[6]) if row[6] else {}
                
                embeddings.append({
                    'embedding_id': row[0],
                    'embedding': embedding_data,
                    'quality_score': row[2],
                    'quality_metrics': quality_metrics,
                    'sequence_length': row[4],
                    'created_at': row[5],
                    'source_info': source_info,
                    'cluster_id': row[7]
                })
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error getting embeddings for person {person_id}: {str(e)}")
            return []
    
    def find_best_match(self, query_embedding: np.ndarray, 
                       quality_score: float = 0.0) -> Optional[Dict]:
        """
        Find the best matching person for a query embedding
        
        Args:
            query_embedding (np.ndarray): Query embedding to match
            quality_score (float): Quality score of the query
            
        Returns:
            Optional[Dict]: Best match result or None
        """
        try:
            # Get all active embeddings
            all_embeddings = self._get_all_active_embeddings()
            
            if not all_embeddings:
                return None
            
            best_match = None
            best_similarity = -1.0
            
            for person_id, embeddings in all_embeddings.items():
                for emb_data in embeddings:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        emb_data['embedding'].reshape(1, -1)
                    )[0][0]
                    
                    # Weight by quality scores
                    weighted_similarity = similarity * (
                        0.7 + 0.3 * (emb_data['quality_score'] + quality_score) / 2.0
                    )
                    
                    if weighted_similarity > best_similarity:
                        best_similarity = weighted_similarity
                        best_match = {
                            'person_id': person_id,
                            'similarity': similarity,
                            'weighted_similarity': weighted_similarity,
                            'matched_embedding': emb_data,
                            'confidence': self._calculate_confidence(similarity, emb_data['quality_score'])
                        }
            
            # Record recognition attempt
            if best_match and best_similarity > self.config['similarity_threshold']:
                self._record_recognition(best_match, query_embedding, quality_score)
                return best_match
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding best match: {str(e)}")
            return None
    
    def get_person_statistics(self, person_id: str = None) -> Dict:
        """
        Get statistics for a specific person or all persons
        
        Args:
            person_id (str): Specific person ID, or None for all persons
            
        Returns:
            Dict: Statistics information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if person_id:
                # Statistics for specific person
                cursor.execute('''
                    SELECT p.person_id, p.name, p.total_embeddings, p.average_quality,
                           p.created_at, p.updated_at,
                           COUNT(e.embedding_id) as active_embeddings,
                           MAX(e.quality_score) as max_quality,
                           AVG(e.sequence_length) as avg_sequence_length
                    FROM persons p
                    LEFT JOIN embeddings e ON p.person_id = e.person_id AND e.is_active = 1
                    WHERE p.person_id = ?
                    GROUP BY p.person_id
                ''', (person_id,))
                
                row = cursor.fetchone()
                if row:
                    stats = {
                        'person_id': row[0],
                        'name': row[1],
                        'total_embeddings': row[2],
                        'average_quality': row[3],
                        'created_at': row[4],
                        'updated_at': row[5],
                        'active_embeddings': row[6] or 0,
                        'max_quality': row[7] or 0.0,
                        'avg_sequence_length': row[8] or 0.0
                    }
                else:
                    stats = {'error': 'Person not found'}
            else:
                # Overall statistics
                cursor.execute('SELECT COUNT(*) FROM persons')
                total_persons = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM embeddings WHERE is_active = 1')
                total_active_embeddings = cursor.fetchone()[0]
                
                cursor.execute('SELECT AVG(quality_score) FROM embeddings WHERE is_active = 1')
                avg_quality = cursor.fetchone()[0] or 0.0
                
                cursor.execute('SELECT COUNT(*) FROM recognition_history')
                total_recognitions = cursor.fetchone()[0]
                
                stats = {
                    'total_persons': total_persons,
                    'total_active_embeddings': total_active_embeddings,
                    'average_quality': avg_quality,
                    'total_recognitions': total_recognitions,
                    'avg_embeddings_per_person': total_active_embeddings / max(total_persons, 1)
                }
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {'error': str(e)}
    
    def cluster_person_embeddings(self, person_id: str) -> Dict:
        """
        Cluster embeddings for a person to identify distinct gait patterns
        
        Args:
            person_id (str): Person identifier
            
        Returns:
            Dict: Clustering results
        """
        try:
            embeddings = self.get_person_embeddings(person_id)
            
            if len(embeddings) < 2:
                return {'clusters': 0, 'embeddings': len(embeddings)}
            
            # Extract embedding vectors
            embedding_vectors = np.array([emb['embedding'] for emb in embeddings])
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(
                eps=self.config['clustering_eps'],
                min_samples=self.config['clustering_min_samples'],
                metric='cosine'
            ).fit(embedding_vectors)
            
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Update database with cluster IDs
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for i, embedding in enumerate(embeddings):
                cursor.execute('''
                    UPDATE embeddings 
                    SET cluster_id = ? 
                    WHERE embedding_id = ?
                ''', (int(labels[i]), embedding['embedding_id']))
            
            conn.commit()
            conn.close()
            
            result = {
                'person_id': person_id,
                'clusters': n_clusters,
                'noise_points': n_noise,
                'total_embeddings': len(embeddings),
                'cluster_labels': labels.tolist()
            }
            
            self.logger.info(f"Clustered embeddings for {person_id}: {n_clusters} clusters, {n_noise} noise points")
            return result
            
        except Exception as e:
            self.logger.error(f"Error clustering embeddings for {person_id}: {str(e)}")
            return {'error': str(e)}
    
    def cleanup_database(self, force: bool = False) -> Dict:
        """
        Clean up old or low-quality embeddings
        
        Args:
            force (bool): Force cleanup regardless of thresholds
            
        Returns:
            Dict: Cleanup statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current statistics
            cursor.execute('SELECT COUNT(*) FROM embeddings WHERE is_active = 1')
            active_embeddings = cursor.fetchone()[0]
            
            cleanup_needed = force or active_embeddings > self.config['auto_cleanup_threshold']
            
            if not cleanup_needed:
                conn.close()
                return {'cleanup_needed': False, 'active_embeddings': active_embeddings}
            
            # Mark old embeddings as inactive
            cursor.execute('''
                UPDATE embeddings 
                SET is_active = 0 
                WHERE created_at < date('now', '-' || ? || ' days')
            ''', (self.config['max_age_days'],))
            
            aged_out = cursor.rowcount
            
            # For each person, keep only the best embeddings
            cursor.execute('SELECT DISTINCT person_id FROM embeddings WHERE is_active = 1')
            person_ids = [row[0] for row in cursor.fetchall()]
            
            deactivated_count = 0
            for person_id in person_ids:
                # Get embeddings for person, ordered by quality
                cursor.execute('''
                    SELECT embedding_id, quality_score
                    FROM embeddings 
                    WHERE person_id = ? AND is_active = 1
                    ORDER BY quality_score DESC, created_at DESC
                ''', (person_id,))
                
                embeddings = cursor.fetchall()
                
                # Keep only the top N embeddings
                if len(embeddings) > self.config['max_embeddings_per_person']:
                    to_deactivate = embeddings[self.config['max_embeddings_per_person']:]
                    
                    for embedding_id, _ in to_deactivate:
                        cursor.execute('''
                            UPDATE embeddings 
                            SET is_active = 0 
                            WHERE embedding_id = ?
                        ''', (embedding_id,))
                        deactivated_count += 1
            
            # Update person statistics
            for person_id in person_ids:
                self._update_person_stats(cursor, person_id)
            
            conn.commit()
            
            # Get final statistics
            cursor.execute('SELECT COUNT(*) FROM embeddings WHERE is_active = 1')
            final_active = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM embeddings WHERE is_active = 0')
            inactive_count = cursor.fetchone()[0]
            
            conn.close()
            
            result = {
                'cleanup_performed': True,
                'initial_active_embeddings': active_embeddings,
                'final_active_embeddings': final_active,
                'aged_out_embeddings': aged_out,
                'quality_filtered_embeddings': deactivated_count,
                'total_inactive_embeddings': inactive_count
            }
            
            self.logger.info(f"Database cleanup completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {str(e)}")
            return {'error': str(e)}
    
    def export_person_data(self, person_id: str, include_inactive: bool = False) -> Dict:
        """
        Export all data for a specific person
        
        Args:
            person_id (str): Person identifier
            include_inactive (bool): Include inactive embeddings
            
        Returns:
            Dict: Complete person data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get person info
            cursor.execute('SELECT * FROM persons WHERE person_id = ?', (person_id,))
            person_row = cursor.fetchone()
            
            if not person_row:
                return {'error': 'Person not found'}
            
            # Get embeddings
            embeddings = self.get_person_embeddings(person_id, active_only=not include_inactive)
            
            # Get recognition history
            cursor.execute('''
                SELECT recognition_id, confidence_score, recognized_at, metadata
                FROM recognition_history 
                WHERE person_id = ? 
                ORDER BY recognized_at DESC
            ''', (person_id,))
            
            recognition_history = []
            for row in cursor.fetchall():
                recognition_history.append({
                    'recognition_id': row[0],
                    'confidence_score': row[1],
                    'recognized_at': row[2],
                    'metadata': json.loads(row[3]) if row[3] else {}
                })
            
            conn.close()
            
            return {
                'person_id': person_id,
                'name': person_row[1],
                'created_at': person_row[2],
                'updated_at': person_row[3],
                'total_embeddings': person_row[4],
                'average_quality': person_row[5],
                'metadata': json.loads(person_row[6]) if person_row[6] else {},
                'embeddings': embeddings,
                'recognition_history': recognition_history
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting data for person {person_id}: {str(e)}")
            return {'error': str(e)}
    
    # Helper methods
    def _upsert_person(self, cursor, person_id: str, person_name: str = None):
        """Create or update person record"""
        cursor.execute('SELECT person_id FROM persons WHERE person_id = ?', (person_id,))
        if cursor.fetchone():
            # Update existing
            if person_name:
                cursor.execute('''
                    UPDATE persons 
                    SET name = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE person_id = ?
                ''', (person_name, person_id))
        else:
            # Create new
            cursor.execute('''
                INSERT INTO persons (person_id, name) 
                VALUES (?, ?)
            ''', (person_id, person_name or f"Person_{person_id}"))
    
    def _update_person_stats(self, cursor, person_id: str):
        """Update person statistics"""
        cursor.execute('''
            SELECT COUNT(*), AVG(quality_score) 
            FROM embeddings 
            WHERE person_id = ? AND is_active = 1
        ''', (person_id,))
        
        count, avg_quality = cursor.fetchone()
        
        cursor.execute('''
            UPDATE persons 
            SET total_embeddings = ?, average_quality = ?, updated_at = CURRENT_TIMESTAMP
            WHERE person_id = ?
        ''', (count or 0, avg_quality or 0.0, person_id))
    
    def _manage_person_embeddings(self, cursor, person_id: str):
        """Ensure person doesn't exceed max embeddings"""
        cursor.execute('''
            SELECT embedding_id, quality_score 
            FROM embeddings 
            WHERE person_id = ? AND is_active = 1 
            ORDER BY quality_score DESC, created_at DESC
        ''', (person_id,))
        
        embeddings = cursor.fetchall()
        
        if len(embeddings) > self.config['max_embeddings_per_person']:
            # Deactivate lowest quality embeddings
            to_deactivate = embeddings[self.config['max_embeddings_per_person']:]
            
            for embedding_id, _ in to_deactivate:
                cursor.execute('''
                    UPDATE embeddings 
                    SET is_active = 0 
                    WHERE embedding_id = ?
                ''', (embedding_id,))
    
    def _is_duplicate_embedding(self, person_id: str, embedding: np.ndarray) -> bool:
        """Check if embedding is too similar to existing ones"""
        existing = self.get_person_embeddings(person_id)
        
        for existing_emb in existing:
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                existing_emb['embedding'].reshape(1, -1)
            )[0][0]
            
            if similarity > 0.95:  # Very high similarity threshold for duplicates
                return True
        
        return False
    
    def _generate_embedding_id(self, person_id: str, embedding: np.ndarray, quality: float) -> str:
        """Generate unique embedding ID"""
        timestamp = datetime.now().isoformat()
        data = f"{person_id}_{timestamp}_{quality}_{embedding.sum()}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _get_all_active_embeddings(self) -> Dict:
        """Get all active embeddings grouped by person"""
        if not self.embedding_cache:
            self._update_cache()
        return self.embedding_cache
    
    def _update_cache(self, person_id: str = None):
        """Update embedding cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if person_id:
                # Update specific person
                embeddings = self.get_person_embeddings(person_id)
                if embeddings:
                    self.embedding_cache[person_id] = embeddings
                elif person_id in self.embedding_cache:
                    del self.embedding_cache[person_id]
            else:
                # Update all
                cursor.execute('SELECT DISTINCT person_id FROM embeddings WHERE is_active = 1')
                person_ids = [row[0] for row in cursor.fetchall()]
                
                self.embedding_cache = {}
                for pid in person_ids:
                    embeddings = self.get_person_embeddings(pid)
                    if embeddings:
                        self.embedding_cache[pid] = embeddings
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating cache: {str(e)}")
    
    def _calculate_confidence(self, similarity: float, quality: float) -> float:
        """Calculate confidence score based on similarity and quality"""
        # Weighted combination of similarity and quality
        confidence = (similarity * 0.7 + quality * 0.3)
        
        # Apply sigmoid-like transformation for better distribution
        confidence = 1.0 / (1.0 + np.exp(-10 * (confidence - 0.7)))
        
        return float(confidence)
    
    def _record_recognition(self, match_result: Dict, query_embedding: np.ndarray, quality: float):
        """Record recognition attempt in history"""
        try:
            recognition_id = hashlib.md5(f"{match_result['person_id']}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            metadata = {
                'query_quality': float(quality),
                'similarity': float(match_result['similarity']),
                'weighted_similarity': float(match_result['weighted_similarity'])
            }
            
            cursor.execute('''
                INSERT INTO recognition_history 
                (recognition_id, person_id, confidence_score, embedding_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (recognition_id, match_result['person_id'], float(match_result['confidence']),
                  match_result['matched_embedding']['embedding_id'], json.dumps(convert_numpy_types(metadata))))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error recording recognition: {str(e)}")
