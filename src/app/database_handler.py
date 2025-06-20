"""
Database handler for the gait recognition system
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
import shutil
from src.utils.device import vprint

class PersonDatabase:
    """Person database with embedding storage and identification"""
    
    def __init__(self):
        """Initialize empty database"""
        self.persons = {}  # {person_id: {name, embedding, face_embedding, quality, metadata}}
        self.last_update = datetime.now().isoformat()
        
    def add_person(self, person_id, name, embedding, quality=1.0, metadata=None):
        """Add person to database"""
        self.persons[person_id] = {
            'id': person_id,
            'name': name,
            'embedding': embedding,
            'face_embedding': None,
            'quality': quality,
            'metadata': metadata or {},
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat()
        }
        self.last_update = datetime.now().isoformat()
        return person_id
        
    def add_person_multimodal(self, person_id, name, gait_embedding, face_embedding=None, 
                             quality=1.0, metadata=None):
        """Add person with multi-modal embeddings"""
        self.persons[person_id] = {
            'id': person_id,
            'name': name,
            'embedding': gait_embedding,
            'face_embedding': face_embedding,
            'quality': quality,
            'metadata': metadata or {},
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat(),
            'has_face': face_embedding is not None
        }
        self.last_update = datetime.now().isoformat()
        return person_id
    
    def update_person(self, person_id, embedding=None, name=None, quality=None, metadata=None):
        """Update existing person"""
        if person_id not in self.persons:
            vprint(f"Warning: Cannot update non-existent person {person_id}")
            return False
            
        if embedding is not None:
            self.persons[person_id]['embedding'] = embedding
            
        if name is not None:
            self.persons[person_id]['name'] = name
            
        if quality is not None and quality > self.persons[person_id]['quality']:
            self.persons[person_id]['quality'] = quality
            
        if metadata is not None:
            self.persons[person_id]['metadata'].update(metadata)
            
        self.persons[person_id]['updated'] = datetime.now().isoformat()
        self.last_update = datetime.now().isoformat()
        return True
        
    def update_person_multimodal(self, person_id, gait_embedding=None, face_embedding=None, 
                               name=None, quality=None, metadata=None):
        """Update existing person with multi-modal data"""
        if person_id not in self.persons:
            vprint(f"Warning: Cannot update non-existent person {person_id}")
            return False
            
        if gait_embedding is not None:
            self.persons[person_id]['embedding'] = gait_embedding
            
        if face_embedding is not None:
            self.persons[person_id]['face_embedding'] = face_embedding
            self.persons[person_id]['has_face'] = True
            
        if name is not None:
            self.persons[person_id]['name'] = name
            
        if quality is not None and quality > self.persons[person_id]['quality']:
            self.persons[person_id]['quality'] = quality
            
        if metadata is not None:
            self.persons[person_id]['metadata'].update(metadata)
            
        self.persons[person_id]['updated'] = datetime.now().isoformat()
        self.last_update = datetime.now().isoformat()
        return True
    
    def identify_person(self, embedding, top_k=1, threshold=0.7):
        """
        Identify person from embedding
        
        Returns:
            List of (person_id, similarity, name, quality) tuples
        """
        if not self.persons:
            return []
            
        similarities = []
        
        # Calculate cosine similarity with all persons
        for person_id, person_data in self.persons.items():
            if 'embedding' in person_data and person_data['embedding'] is not None:
                sim = self._cosine_similarity(embedding, person_data['embedding'])
                similarities.append((person_id, sim, person_data['name'], person_data.get('quality', 1.0)))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and return top_k
        return [match for match in similarities[:top_k] if match[1] >= threshold]
    
    def identify_person_adaptive(self, embedding, method='nucleus', top_p=0.9, 
                               min_candidates=1, max_candidates=5, threshold=0.7,
                               close_sim_threshold=0.05, amplification_factor=1.2,
                               quality_weight=0.3, enhanced_ranking=True):
        """
        Identify person with adaptive sampling (nucleus sampling or other methods)
        
        Args:
            embedding: Query embedding
            method: Sampling method ('nucleus', 'adaptive_threshold', 'quality_weighted')
            top_p: Nucleus sampling parameter (0.0-1.0)
            min_candidates: Minimum number of candidates to return
            max_candidates: Maximum number of candidates to return
            threshold: Minimum similarity threshold
            close_sim_threshold: Threshold for "close" matches
            amplification_factor: Factor to amplify probabilities for close matches
            quality_weight: Weight of quality in ranking (0.0-1.0)
            enhanced_ranking: Whether to use enhanced ranking with quality
            
        Returns:
            List of (person_id, similarity, name, quality) tuples
        """
        if not self.persons:
            return []
            
        similarities = []
        
        # Calculate cosine similarity with all persons
        for person_id, person_data in self.persons.items():
            if 'embedding' in person_data and person_data['embedding'] is not None:
                sim = self._cosine_similarity(embedding, person_data['embedding'])
                similarities.append((person_id, sim, person_data['name'], person_data.get('quality', 1.0)))
        
        # Apply quality-weighted ranking if enabled
        if enhanced_ranking:
            # Combine similarity and quality
            similarities = [(pid, sim * (1 - quality_weight) + qual * quality_weight, name, qual) 
                          for pid, sim, name, qual in similarities]
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply requested sampling method
        if method == 'nucleus':
            # Filter by threshold first
            filtered_sims = [match for match in similarities if match[1] >= threshold]
            
            if not filtered_sims:
                return []
            
            # Apply nucleus sampling
            max_sim = filtered_sims[0][1]
            
            # Convert to probabilities
            probs = np.array([sim for _, sim, _, _ in filtered_sims])
            probs = probs / np.sum(probs)
            
            # Calculate cumulative probabilities
            cum_probs = np.cumsum(probs)
            
            # Find nucleus (where cumulative prob exceeds top_p)
            nucleus_idx = np.argmax(cum_probs >= top_p)
            
            # Ensure we have at least min_candidates
            nucleus_idx = max(nucleus_idx + 1, min_candidates)
            
            # Limit to max_candidates
            nucleus_idx = min(nucleus_idx, max_candidates, len(filtered_sims))
            
            return filtered_sims[:nucleus_idx]
            
        elif method == 'adaptive_threshold':
            # Start with baseline threshold
            filtered_sims = [match for match in similarities if match[1] >= threshold]
            
            if not filtered_sims or len(filtered_sims) < min_candidates:
                # If too few matches, try using a lower threshold
                adaptive_thresh = threshold * 0.9
                filtered_sims = [match for match in similarities if match[1] >= adaptive_thresh]
                
            # Check for close matches and adjust confidence
            if filtered_sims:
                top_sim = filtered_sims[0][1]
                
                # Boost confidence of very close matches
                boosted_matches = []
                for person_id, sim, name, quality in filtered_sims:
                    if top_sim - sim <= close_sim_threshold:
                        # This is a close match, boost confidence
                        boosted_sim = min(1.0, sim * amplification_factor)
                        boosted_matches.append((person_id, boosted_sim, name, quality))
                    else:
                        boosted_matches.append((person_id, sim, name, quality))
                
                # Sort again after boosting
                boosted_matches.sort(key=lambda x: x[1], reverse=True)
                filtered_sims = boosted_matches
            
            # Limit to max_candidates
            return filtered_sims[:max_candidates]
            
        else:
            # Default method - just filter by threshold and return top results
            filtered_sims = [match for match in similarities if match[1] >= threshold]
            return filtered_sims[:max_candidates]
    
    def get_top_persons(self, max_count=10, sort_by='updated'):
        """Get top persons sorted by specified field"""
        if not self.persons:
            return []
        
        persons = list(self.persons.values())
        
        # Sort based on requested field
        if sort_by == 'quality':
            persons.sort(key=lambda p: p.get('quality', 0.0), reverse=True)
        elif sort_by == 'updated':
            persons.sort(key=lambda p: p.get('updated', ''), reverse=True)
        elif sort_by == 'created':
            persons.sort(key=lambda p: p.get('created', ''), reverse=True)
            
        return persons[:max_count]
    
    def get_stats(self):
        """Get database statistics"""
        num_persons = len(self.persons)
        num_with_face = sum(1 for p in self.persons.values() if p.get('face_embedding') is not None)
        avg_quality = np.mean([p.get('quality', 1.0) for p in self.persons.values()]) if self.persons else 0.0
        
        return {
            'count': num_persons,
            'multimodal_count': num_with_face,
            'avg_quality': avg_quality,
            'last_update': self.last_update
        }
    
    def clear_database(self):
        """Clear all persons from database"""
        self.persons = {}
        self.last_update = datetime.now().isoformat()
    
    def _cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
            
        # Flatten embeddings if they are multi-dimensional
        if len(emb1.shape) > 1:
            emb1 = emb1.flatten()
        if len(emb2.shape) > 1:
            emb2 = emb2.flatten()
            
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def __len__(self):
        return len(self.persons)


class DatabaseHandler:
    """Handles database operations and persistence"""
    
    def __init__(self, config):
        """Initialize database handler"""
        self.config = config
        self.person_db = PersonDatabase()
    
    def load_database(self, db_path):
        """
        Load database from disk
        
        Returns:
            Next available person ID
        """
        db_file = os.path.join(db_path, "person_db.pkl")
        meta_file = os.path.join(db_path, "db_metadata.json")
        
        # Try to load database
        try:
            if os.path.exists(db_file):
                with open(db_file, 'rb') as f:
                    self.person_db = pickle.load(f)
                vprint(f"Loaded database with {len(self.person_db)} persons")
            else:
                vprint("No existing database found, creating new one")
                self.person_db = PersonDatabase()
        except Exception as e:
            vprint(f"Error loading database: {e}")
            # Backup corrupted database if it exists
            if os.path.exists(db_file):
                backup_file = f"{db_file}.bak.{int(datetime.now().timestamp())}"
                shutil.copy2(db_file, backup_file)
                vprint(f"Created backup of corrupted database: {backup_file}")
                
            # Create new database
            self.person_db = PersonDatabase()
        
        # Load metadata if available
        next_id = 1
        try:
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    next_id = metadata.get('next_person_id', 1)
        except Exception as e:
            vprint(f"Error loading metadata: {e}")
        
        return next_id
    
    def save_database(self, db_path):
        """Save database to disk"""
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        db_file = os.path.join(db_path, "person_db.pkl")
        meta_file = os.path.join(db_path, "db_metadata.json")
        
        # Save backup first if existing file present
        if os.path.exists(db_file):
            backup_file = os.path.join(db_path, "person_db_backup.pkl")
            shutil.copy2(db_file, backup_file)
        
        # Save database
        try:
            with open(db_file, 'wb') as f:
                pickle.dump(self.person_db, f)
                
            # Save metadata
            metadata = {
                'last_save': datetime.now().isoformat(),
                'person_count': len(self.person_db),
                'next_person_id': self._find_next_available_id()
            }
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            vprint(f"Database saved with {len(self.person_db)} persons")
            return True
        except Exception as e:
            vprint(f"Error saving database: {e}")
            return False
    
    def clear_database(self):
        """Clear the database"""
        self.person_db.clear_database()
        vprint("Database cleared")
    
    def _find_next_available_id(self):
        """Find next available numeric person ID"""
        next_id = 1
        
        # Check existing IDs
        for person_id in self.person_db.persons:
            if person_id.startswith('P'):
                try:
                    id_num = int(person_id[1:])
                    next_id = max(next_id, id_num + 1)
                except ValueError:
                    pass
        
        return next_id
    
    def identify_person_face(self, face_embedding, top_k=5, threshold=0.6):
        """
        Identify person using face embedding only
        
        Args:
            face_embedding: Face embedding vector (512,)
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (person_id, similarity, name, quality) tuples
        """
        if not self.person_db.persons:
            return []
            
        similarities = []
        
        # Calculate cosine similarity with all persons that have face embeddings
        for person_id, person_data in self.person_db.persons.items():
            if person_data.get('face_embedding') is not None:
                face_emb = person_data['face_embedding']
                sim = self.person_db._cosine_similarity(face_embedding, face_emb)
                similarities.append((person_id, sim, person_data['name'], person_data.get('quality', 1.0)))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and return top_k
        return [match for match in similarities[:top_k] if match[1] >= threshold]
    
    def identify_person_nucleus(self, gait_embedding, top_p=0.85, min_candidates=1, max_candidates=5,
                               close_sim_threshold=0.08, amplification_factor=35.0, quality_weight=0.8,
                               enhanced_ranking=True):
        """
        Identify person using nucleus sampling (for gait embeddings)
        This method provides the nucleus sampling interface expected by MultiModalIdentifier
        """
        return self.person_db.identify_person_adaptive(
            gait_embedding, 
            method='nucleus',
            top_p=top_p,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            threshold=0.7,  # Default threshold
            close_sim_threshold=close_sim_threshold,
            amplification_factor=amplification_factor,
            quality_weight=quality_weight,
            enhanced_ranking=enhanced_ranking
        )
    
    def identify_person_multimodal(self, gait_embedding=None, face_embedding=None, 
                                  gait_weight=0.3, face_weight=0.7, 
                                  face_threshold=0.6, gait_threshold=0.7,
                                  require_both=False, top_k=5):
        """
        Identify person using both gait and face embeddings with fusion
        
        Args:
            gait_embedding: Gait embedding (optional)
            face_embedding: Face embedding (optional)
            gait_weight: Weight for gait similarity (0.0-1.0)
            face_weight: Weight for face similarity (0.0-1.0)
            face_threshold: Minimum face similarity threshold
            gait_threshold: Minimum gait similarity threshold
            require_both: If True, require both modalities to match
            top_k: Number of top matches to return
            
        Returns:
            List of (person_id, fused_similarity, name, quality) tuples
        """
        if not self.person_db.persons:
            return []
        
        # Get individual modality matches
        face_matches = []
        gait_matches = []
        
        if face_embedding is not None:
            face_matches = self.identify_person_face(face_embedding, top_k=10, threshold=face_threshold)
        
        if gait_embedding is not None:
            gait_matches = self.person_db.identify_person_adaptive(gait_embedding, threshold=gait_threshold)
        
        # Handle different scenarios
        if require_both and not (face_matches and gait_matches):
            return []  # Require both modalities
        elif face_matches and gait_matches:
            return self._fuse_multimodal_results(face_matches, gait_matches, face_weight, gait_weight, top_k)
        elif face_matches:
            return face_matches[:top_k]
        elif gait_matches:
            return gait_matches[:top_k]
        else:
            return []
    
    def _fuse_multimodal_results(self, face_matches, gait_matches, face_weight, gait_weight, top_k):
        """
        Fuse face and gait recognition results
        
        Args:
            face_matches: List of face recognition matches
            gait_matches: List of gait recognition matches
            face_weight: Weight for face similarity
            gait_weight: Weight for gait similarity
            top_k: Number of top matches to return
            
        Returns:
            Fused list of matches
        """
        # Collect all person IDs
        all_person_ids = set()
        for matches in [face_matches, gait_matches]:
            for match in matches:
                all_person_ids.add(match[0])
        
        # Calculate fused scores
        fused_results = []
        for person_id in all_person_ids:
            face_score = 0.0
            gait_score = 0.0
            person_name = None
            person_quality = 0.0
            
            # Get face score if available
            for match in face_matches:
                if match[0] == person_id:
                    face_score = match[1]
                    person_name = match[2]
                    person_quality = match[3]
                    break
            
            # Get gait score if available
            for match in gait_matches:
                if match[0] == person_id:
                    gait_score = match[1]
                    if not person_name:
                        person_name = match[2]
                        person_quality = match[3]
                    break
            
            # Calculate weighted score
            fused_score = (face_weight * face_score) + (gait_weight * gait_score)
            
            if person_name:
                fused_results.append((person_id, fused_score, person_name, person_quality))
        
        # Sort by fused score and return top candidates
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:top_k]