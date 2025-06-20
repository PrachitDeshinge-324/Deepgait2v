import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from typing import List, Dict, Tuple, Optional
import config
from src.utils.device import vprint
class FaceEmbeddingExtractor:
    """Extracts face embeddings from video tracks using InsightFace"""
    
    def __init__(self, 
                 device: str = 'cuda:0',
                 det_name: str = 'buffalo_l', 
                 rec_name: str = 'buffalo_l',
                 det_size: Tuple[int, int] = (640, 640)):
        """
        Initialize face detection and recognition models
        
        Args:
            device: Device to run inference on ('cuda:0', 'cpu')
            det_name: Face detection model name
            rec_name: Face recognition model name
            det_size: Detection size (width, height)
        """
        self.device = device
        
        # Initialize InsightFace
        self.face_analyzer = FaceAnalysis(name=det_name)
        self.face_analyzer.prepare(ctx_id=0 if 'cuda' in device else -1, det_size=det_size)
        
        # Note: We'll use the face_analyzer for both detection and embedding extraction
        # The face_analyzer.get() method returns face objects with embeddings included
        
        # Cache for face quality tracking
        self.face_cache = {}  # track_id -> list of (frame_num, face_crop, face_obj, quality)
        self.best_faces = {}  # track_id -> (face_obj, quality, face_crop)
        self.face_embeddings_cache = {}  # track_id -> cached embedding
        self.last_face_boxes = {}  # track_id -> face box info for visualization
        self.face_window_size = getattr(config, 'FACE_CACHE_SIZE', 30)  # Number of faces to cache before selecting best
        self.min_quality_threshold = getattr(config, 'FACE_QUALITY_THRESHOLD', 0.5)
        self.frame_num = 0
        
    def extract_faces(self, frame: np.ndarray, tracks: List) -> Dict:
        """
        Extract faces from tracked persons and assess quality
        
        Args:
            frame: Current video frame
            tracks: List of tracked persons with track_ids
            
        Returns:
            Dict mapping track_id to best face embedding
        """
        # Increment frame counter
        self.frame_num += 1
        
        # Process faces for each track
        for track in tracks:
            track_id = track.track_id
            
            # Debug: Print track info
            frame_count = getattr(track, 'frame_count', 0)
            vprint(f"DEBUG Face Extractor - Track {track_id}: frame_count={frame_count}")
            
            # Skip if track hasn't been visible long enough (reduced for debugging)
            if not hasattr(track, 'frame_count') or track.frame_count < 3:
                vprint(f"DEBUG Face Extractor - Track {track_id} skipped: not visible long enough")
                continue
                
            # Extract bounding box
            tlwh = track.tlwh
            x, y, w, h = map(int, tlwh)
            
            vprint(f"DEBUG Face Extractor - Track {track_id}: extracting from region {x},{y} size {w}x{h}")
            
            # Create a slightly larger crop for face detection (focus on upper body)
            face_crop_x = max(0, x - int(w * 0.1))
            face_crop_y = max(0, y - int(h * 0.1))
            face_crop_w = min(frame.shape[1] - face_crop_x, int(w * 1.2))
            face_crop_h = min(frame.shape[0] - face_crop_y, int(h * 0.6))  # Focus on upper body area
            
            face_crop = frame[face_crop_y:face_crop_y+face_crop_h, face_crop_x:face_crop_x+face_crop_w]
            
            vprint(f"DEBUG Face Extractor - Track {track_id}: face crop region {face_crop_x},{face_crop_y} size {face_crop_w}x{face_crop_h}, actual shape: {face_crop.shape}")
            
            # Skip if crop is empty
            if face_crop.size == 0:
                continue
                
            # Detect faces
            faces = self.face_analyzer.get(face_crop)
            
            vprint(f"DEBUG Face Extractor - Track {track_id}: detected {len(faces)} faces")
            
            if len(faces) > 0:
                # Find best face in this frame based on detection score
                best_face = max(faces, key=lambda x: x.det_score)
                vprint(f"DEBUG Face Extractor - Track {track_id}: best face det_score={best_face.det_score}")
                
                # Calculate quality score
                quality_score = self._evaluate_face_quality(best_face, face_crop)
                
                vprint(f"DEBUG Face Extractor - Track {track_id}: face quality={quality_score:.3f}")
                
                # Store face in cache
                if track_id not in self.face_cache:
                    self.face_cache[track_id] = []
                
                # Add to cache
                self.face_cache[track_id].append((self.frame_num, face_crop, best_face, quality_score))
                
                # Update best face if better
                if track_id not in self.best_faces or quality_score > self.best_faces[track_id][1]:
                    self.best_faces[track_id] = (best_face, quality_score, face_crop)
                
                # Clean up old faces from cache
                if len(self.face_cache[track_id]) > self.face_window_size:
                    self._select_best_face(track_id)
        
        # Return track_id -> embeddings for tracks that have a face
        result = {}
        face_boxes = {}  # Store face bounding boxes for visualization
        
        for track_id in self.best_faces:
            embedding = self.get_face_embedding(track_id)
            if embedding is not None:
                result[track_id] = embedding
                
                # Get face bounding box for visualization
                face_obj, quality_score, face_crop = self.best_faces[track_id]
                
                # Get the original track to calculate face position in frame
                track = None
                for t in tracks:
                    if t.track_id == track_id:
                        track = t
                        break
                
                if track is not None:
                    # Calculate face crop region
                    tlwh = track.tlwh
                    x, y, w, h = map(int, tlwh)
                    face_crop_x = max(0, x - int(w * 0.1))
                    face_crop_y = max(0, y - int(h * 0.1))
                    
                    # Get face bbox relative to crop and convert to frame coordinates
                    face_bbox = face_obj.bbox  # [x1, y1, x2, y2] relative to crop
                    frame_face_x1 = int(face_crop_x + face_bbox[0])
                    frame_face_y1 = int(face_crop_y + face_bbox[1])
                    frame_face_x2 = int(face_crop_x + face_bbox[2])
                    frame_face_y2 = int(face_crop_y + face_bbox[3])
                    
                    face_boxes[track_id] = {
                        'bbox': [frame_face_x1, frame_face_y1, frame_face_x2, frame_face_y2],
                        'quality': quality_score,
                        'det_score': face_obj.det_score
                    }
                
                vprint(f"DEBUG Face Extractor - Track {track_id}: face embedding extracted, shape={embedding.shape}")
            else:
                print(f"DEBUG Face Extractor - Track {track_id}: face embedding extraction failed")
        
        # Store face boxes for visualization
        self.last_face_boxes = face_boxes
        
        return result
    
    def _evaluate_face_quality(self, face, face_crop: np.ndarray) -> float:
        """
        Evaluate face quality based on multiple factors
        
        Args:
            face: InsightFace face object
            face_crop: Original image crop containing the face
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        # Face detection confidence
        det_score = face.det_score  # 0-1 range
        
        # Face size relative to frame
        bbox = face.bbox.astype(int)
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        crop_area = face_crop.shape[0] * face_crop.shape[1]
        relative_size = min(face_area / crop_area, 0.5) * 2  # Normalize to 0-1
        
        # Frontal-ness (based on landmarks)
        landmarks = face.kps
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate horizontal eye line angle
        eye_angle = np.abs(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        frontal_score = 1.0 - min(eye_angle / 0.5, 1.0)  # Normalize to 0-1
        
        # Blur detection using Laplacian variance
        face_roi = face_crop[max(0, int(bbox[1])):min(face_crop.shape[0], int(bbox[3])),
                            max(0, int(bbox[0])):min(face_crop.shape[1], int(bbox[2]))]
        if face_roi.size > 0:
            if len(face_roi.shape) == 3:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_roi
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            # Normalize blur score (higher variance = less blur)
            blur_score = min(1.0, blur_score / 1000.0)  # Adjust threshold as needed
        else:
            blur_score = 0.0
        
        # Final quality score - weighted average with additional checks
        base_quality = 0.3 * det_score + 0.25 * relative_size + 0.25 * frontal_score + 0.2 * blur_score
        
        # Additional quality checks
        # Check if face is too close to edge of crop
        edge_penalty = 0.0
        if bbox[0] < 10 or bbox[1] < 10 or bbox[2] > face_crop.shape[1] - 10 or bbox[3] > face_crop.shape[0] - 10:
            edge_penalty = 0.1
        
        # Check for reasonable face size (not too small or too large)
        face_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        if face_size < 50:  # Too small
            edge_penalty += 0.2
        elif face_size > min(face_crop.shape[:2]) * 0.8:  # Too large (likely false detection)
            edge_penalty += 0.15
        
        quality_score = max(0.0, base_quality - edge_penalty)
        
        return quality_score
    
    def _select_best_face(self, track_id: str) -> None:
        """
        Select best face from cache and clean up
        
        Args:
            track_id: ID of the track
        """
        if track_id not in self.face_cache or len(self.face_cache[track_id]) == 0:
            return
            
        # Get the face with highest quality
        best_entry = max(self.face_cache[track_id], key=lambda x: x[3])
        self.best_faces[track_id] = (best_entry[2], best_entry[3], best_entry[1])
        
        # Keep only recent faces in cache
        recent_faces = sorted(self.face_cache[track_id], key=lambda x: x[0], reverse=True)
        self.face_cache[track_id] = recent_faces[:self.face_window_size//2]
    
    def get_face_embedding(self, track_id: str) -> Optional[np.ndarray]:
        """
        Get face embedding for a track with caching
        
        Args:
            track_id: ID of the track
            
        Returns:
            Face embedding as numpy array or None if no face
        """
        # Check if we have a cached embedding
        if track_id in self.face_embeddings_cache:
            return self.face_embeddings_cache[track_id]
            
        if track_id not in self.best_faces:
            return None
            
        best_face, quality, face_crop = self.best_faces[track_id]
        
        # Only extract embedding if quality is above threshold
        if quality < self.min_quality_threshold:
            print(f"DEBUG Face Extractor - Track {track_id}: quality {quality:.3f} below threshold {self.min_quality_threshold}")
            return None
        
        try:
            # Use the face embedding directly from the detected face
            # The face is already detected and has the embedding computed by InsightFace
            face_embedding = best_face.embedding
            
            vprint(f"DEBUG Face Extractor - Track {track_id}: using direct face embedding, shape={face_embedding.shape}")
            
            # Normalize the embedding for better similarity computation
            face_embedding = face_embedding / np.linalg.norm(face_embedding)
            
            # Cache the embedding
            self.face_embeddings_cache[track_id] = face_embedding
            
            vprint(f"DEBUG Face Extractor - Track {track_id}: embedding extracted successfully, normalized shape={face_embedding.shape}")
            return face_embedding
                
        except Exception as e:
            print(f"Error extracting face embedding for track {track_id}: {e}")
            return None
    
    def get_face_statistics(self) -> Dict:
        """Get statistics about face detection and quality"""
        stats = {
            'total_tracks_with_faces': len(self.best_faces),
            'tracks_with_cached_embeddings': len(self.face_embeddings_cache),
            'total_faces_processed': sum(len(cache) for cache in self.face_cache.values()),
            'average_face_quality': 0.0,
            'high_quality_faces': 0,
            'processed_frames': self.frame_num
        }
        
        if self.best_faces:
            qualities = [quality for _, quality, _ in self.best_faces.values()]
            stats['average_face_quality'] = np.mean(qualities)
            stats['high_quality_faces'] = sum(1 for q in qualities if q >= self.min_quality_threshold)
            stats['quality_std'] = np.std(qualities)
            stats['min_quality'] = np.min(qualities)
            stats['max_quality'] = np.max(qualities)
        
        return stats
    
    def clear_track_cache(self, track_id: str):
        """Clear cache for a specific track to free memory"""
        if track_id in self.face_cache:
            del self.face_cache[track_id]
        if track_id in self.best_faces:
            del self.best_faces[track_id]
        if track_id in self.face_embeddings_cache:
            del self.face_embeddings_cache[track_id]
    
    def get_face_quality_for_track(self, track_id: str) -> Optional[float]:
        """Get the current best face quality for a track"""
        if track_id in self.best_faces:
            _, quality, _ = self.best_faces[track_id]
            return quality
        return None