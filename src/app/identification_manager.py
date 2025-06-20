"""
Person identification management for gait recognition system
"""

from ..utils.device import vprint
from datetime import datetime
from ..identification.multimodal_identifier import MultiModalIdentifier

class IdentificationManager:
    """Manages person identification"""
    
    def __init__(self, config, database_handler, next_person_id):
        """Initialize with configuration and database handler"""
        self.config = config
        self.database_handler = database_handler  # DatabaseHandler instance
        self.person_db = database_handler.person_db  # PersonDatabase instance for direct access
        self.next_person_id = next_person_id
        
        # Initialize multimodal identifier with cross-camera adaptation
        # Note: MultiModalIdentifier expects a database handler, not just the PersonDatabase
        self.multimodal_identifier = MultiModalIdentifier(self.database_handler)
        
        # Configure cross-camera adaptation
        if hasattr(config, 'ENABLE_CROSS_CAMERA_ADAPTATION') and config.ENABLE_CROSS_CAMERA_ADAPTATION:
            self.multimodal_identifier.enable_cross_camera = True
        
        # Set camera ID if available
        if hasattr(config, 'CURRENT_CAMERA_ID'):
            self.multimodal_identifier.set_camera_id(config.CURRENT_CAMERA_ID)
        else:
            self.multimodal_identifier.set_camera_id("default_camera")
            
        # Collect embeddings for adaptation from existing database
        self._initialize_cross_camera_adaptation()
        
    def _initialize_cross_camera_adaptation(self):
        """Initialize cross-camera adaptation with existing database"""
        if not self.multimodal_identifier.enable_cross_camera:
            return
            
        # Collect embeddings from existing database for adaptation
        for person_id, person_data in self.person_db.persons.items():
            gait_embedding = person_data.get('embedding')
            face_embedding = person_data.get('face_embedding')
            
            if gait_embedding is not None or face_embedding is not None:
                self.multimodal_identifier.collect_embeddings_for_adaptation(
                    gait_embedding, face_embedding
                )
        
        # Fit the adaptation if we have enough samples
        self.multimodal_identifier.fit_cross_camera_adaptation()
        vprint("✅ Cross-camera domain adaptation initialized")
        
    def process_identification(self, track_id, embedding, face_embedding, quality, active_track_ids, track_identities):
        """
        Process person identification for a track
        
        Args:
            track_id: Track ID to identify
            embedding: Gait embedding
            face_embedding: Optional face embedding
            quality: Quality score of the sequence
            active_track_ids: Set of active track IDs
            track_identities: Dictionary of track identities
        """
        vprint(f"Generated embedding for track {track_id} - Shape: {embedding.shape}")
        
        # First check if this track already has an identity
        if track_id in track_identities:
            self._update_existing_identity(track_id, embedding, face_embedding, quality, track_identities)
        else:
            # Identify or create new identity
            self._create_new_identity(track_id, embedding, face_embedding, quality, active_track_ids, track_identities)
    
    def _update_existing_identity(self, track_id, embedding, face_embedding, quality, track_identities):
        """Update existing identity with improved embedding if quality is better"""
        # Track already identified, just update embedding if quality improved
        current_person_id = track_identities[track_id]['person_id']
        current_person_quality = track_identities[track_id]['quality']
        
        vprint(f"Track {track_id} already identified as {track_identities[track_id]['name']}")
        if quality > current_person_quality:
            vprint(f"Updating existing identity {current_person_id} with higher quality embedding")
            # Update with face embedding if available
            if face_embedding is not None:
                self.person_db.update_person_multimodal(
                    current_person_id, gait_embedding=embedding, 
                    face_embedding=face_embedding, quality=quality
                )
                vprint(f"Updated person {current_person_id} with both gait and face embeddings")
            else:
                self.person_db.update_person_multimodal(
                    current_person_id, gait_embedding=embedding, quality=quality
                )
                vprint(f"Updated person {current_person_id} with gait embedding only")
            track_identities[track_id]['quality'] = quality
            
            # Collect updated embeddings for cross-camera adaptation
            if self.multimodal_identifier.enable_cross_camera:
                self.multimodal_identifier.collect_embeddings_for_adaptation(
                    embedding, face_embedding
                )
    
    def _create_new_identity(self, track_id, embedding, face_embedding, quality, active_track_ids, track_identities):
        """Identify track or create new identity"""
        vprint(f"Track {track_id} needs identity assignment")
        
        # Get identification matches
        matches = self._get_identification_matches(track_id, embedding, face_embedding, quality)
        
        # Debug matches
        vprint(f"Track {track_id}: Found {len(matches)} potential matches")
        for i, match in enumerate(matches):
            person_id, similarity, name, stored_quality = match
            vprint(f"  Match {i+1}: {name} (ID: {person_id}), similarity: {similarity:.3f}, quality: {stored_quality:.3f}")
        
        # Get list of person IDs already assigned to OTHER active tracks
        assigned_person_ids = self._get_assigned_person_ids(track_id, active_track_ids, track_identities)
        vprint(f"Person IDs already assigned to other tracks: {assigned_person_ids}")
        
        # Find matches that aren't already assigned
        available_matches = [match for match in matches if match[0] not in assigned_person_ids]
        vprint(f"Available matches (not used by other tracks): {len(available_matches)}")
        
        # Assign identity or create new person
        if len(available_matches) > 0:
            self._assign_existing_identity(track_id, available_matches[0], embedding, face_embedding, quality, track_identities)
        else:
            self._create_new_person(track_id, embedding, face_embedding, quality, track_identities)
    
    def _get_identification_matches(self, track_id, embedding, face_embedding, quality):
        """Get identification matches using the multimodal identifier with cross-camera adaptation"""
        
        # Use the multimodal identifier which includes cross-camera adaptation
        matches = self.multimodal_identifier.identify_with_proximity_awareness(
            track_id=track_id,
            gait_embedding=embedding,
            face_embedding=face_embedding, 
            quality=quality,
            is_close_to_camera=False  # Could be made dynamic based on detection size
        )
        
        vprint(f"Track {track_id}: Multimodal identifier returned {len(matches)} matches")
        
        return matches
    
    def _get_assigned_person_ids(self, current_track_id, active_track_ids, track_identities):
        """Get person IDs already assigned to other active tracks"""
        assigned_person_ids = set()
        for other_id in active_track_ids:
            if other_id != current_track_id and other_id in track_identities:
                assigned_person_ids.add(track_identities[other_id]['person_id'])
        return assigned_person_ids
    
    def _assign_existing_identity(self, track_id, match, embedding, face_embedding, quality, track_identities):
        """Assign an existing identity to a track"""
        person_id, similarity, name, stored_quality = match
        
        vprint(f">>> ASSIGNING Track {track_id} to existing database person {name} with similarity {similarity:.3f}")
        
        # Assign this identity to the track
        track_identities[track_id] = {
            'person_id': person_id,
            'name': name,
            'confidence': similarity,
            'quality': quality,
            'is_new': False  # Not new, from database
        }
        
        vprint(f"SUCCESS: Track {track_id} identified as existing person {name}")
        
        # Update database if quality improved
        if quality > stored_quality:
            if face_embedding is not None:
                self.person_db.update_person_multimodal(
                    person_id, gait_embedding=embedding, 
                    face_embedding=face_embedding, quality=quality
                )
            else:
                self.person_db.update_person(person_id, embedding=embedding, quality=quality)
                
            # Collect updated embeddings for cross-camera adaptation
            if self.multimodal_identifier.enable_cross_camera:
                self.multimodal_identifier.collect_embeddings_for_adaptation(
                    embedding, face_embedding
                )
    
    def _create_new_person(self, track_id, embedding, face_embedding, quality, track_identities):
        """Create a new person identity"""
        # Use sequential numbering for consistent IDs
        new_person_id = f"P{self.next_person_id:04d}"
        
        # Generate a simple numbered name
        new_person_name = f"Person-{self.next_person_id:04d}"
        
        # Increment the counter for next time
        self.next_person_id += 1
        
        vprint(f"Creating new person {new_person_name} for track {track_id}")
        
        # Add to database with multimodal capabilities
        if face_embedding is not None:
            self.person_db.add_person_multimodal(
                person_id=new_person_id,
                name=new_person_name,
                gait_embedding=embedding,
                face_embedding=face_embedding,
                quality=quality,
                metadata={'first_seen': datetime.now().isoformat()}
            )
            vprint(f"Added new person {new_person_name} with both gait and face embeddings")
        else:
            self.person_db.add_person(
                person_id=new_person_id,
                name=new_person_name,
                embedding=embedding,
                quality=quality,
                metadata={'first_seen': datetime.now().isoformat()}
            )
            vprint(f"Added new person {new_person_name} with gait embedding only")
        
        # Collect new embeddings for cross-camera adaptation
        if self.multimodal_identifier.enable_cross_camera:
            self.multimodal_identifier.collect_embeddings_for_adaptation(
                embedding, face_embedding
            )
        
        # Assign to track
        track_identities[track_id] = {
            'person_id': new_person_id,
            'name': new_person_name,
            'confidence': 1.0,
            'quality': quality,
            'is_new': True  # New person
        }