"""
Visualization handler for the gait recognition system
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import colorsys
import math
import colorsys

class VisualizationHandler:
    """Handles visualization of tracks, identities and debug information"""
    
    def __init__(self, config, visualizer, face_extractor):
        """Initialize with configuration and visualizer component"""
        self.config = config
        self.visualizer = visualizer
        self.face_extractor = face_extractor
        self.colors = self._generate_colors(100)  # Generate colors for tracks
        self.display_mode = getattr(config, 'DISPLAY_MODE', 'standard')
        self.show_quality_bar = getattr(config, 'SHOW_QUALITY_BAR', True)
        self.quality_bar_height = 3
        self.show_debug_info = getattr(config, 'SHOW_DEBUG_INFO', False)
        self.previous_frame = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position_history = {}  # For motion trails
        self.blur_faces = getattr(config, 'BLUR_FACES', False)
        self.show_motion_trails = getattr(config, 'SHOW_MOTION_TRAILS', False)
        self.max_trail_length = getattr(config, 'MAX_TRAIL_LENGTH', 30)
        self.cctv_overlay = getattr(config, 'SHOW_CCTV_OVERLAY', True)
        self.box_style = getattr(config, 'BOX_STYLE', 'rounded')  # 'rounded', 'sharp', or 'dashed'
        self.analytics_data = {
            'zone_counts': {},
            'dwell_times': {},
            'interaction_events': []
        }
        # Define zones for analytics
        self.zones = getattr(config, 'ZONES', {
            'entry': [(0, 0.7, 0.3, 1.0)],  # x1, y1, x2, y2 in relative coordinates
            'exit': [(0.7, 0.7, 1.0, 1.0)],
            'center': [(0.3, 0.3, 0.7, 0.7)]
        })
        
    def create_visualized_frame(self, frame, tracks, track_identities, 
                                face_embeddings, active_track_ids, person_db,
                                gait_recognizer, frame_count, fps):
        """
        Create visualization with all annotations and debugging info
        
        Args:
            frame: Original video frame
            tracks: List of tracks from tracker
            track_identities: Dictionary mapping track_id to identity info
            face_embeddings: Dictionary mapping track_id to face embeddings
            active_track_ids: Set of active track IDs
            person_db: Person database object
            gait_recognizer: Gait recognizer object
            frame_count: Current frame count
            fps: Current processing FPS
            
        Returns:
            Annotated frame
        """
        # Copy frame for visualization
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        # Add CCTV overlay elements (timestamp, grid lines, etc.)
        if self.cctv_overlay:
            vis_frame = self._add_cctv_overlay(vis_frame, frame_count)
            
        # Blur faces if privacy mode enabled
        if self.blur_faces:
            vis_frame = self._blur_faces_in_frame(vis_frame, tracks)
            
        # Add motion trails for better movement visualization
        if self.show_motion_trails:
            self._add_motion_trails(vis_frame, tracks, active_track_ids)
            
        # Update analytics data
        self._update_analytics_data(vis_frame, tracks, active_track_ids, frame_count)
        
        # Draw tracks
        self._draw_tracks(vis_frame, tracks, track_identities, active_track_ids)
        
        # Add frame information overlay
        self._add_frame_info(vis_frame, frame_count, fps, len(tracks), len(person_db))
        
        # Draw zone overlays if in analytics mode
        if self.display_mode == 'analytics':
            self._draw_zone_overlays(vis_frame)
        
        # Add visualization mode-specific elements
        if self.display_mode == 'debug':
            vis_frame = self._add_debug_visualization(vis_frame, tracks, person_db, gait_recognizer)
        elif self.display_mode == 'gallery':
            vis_frame = self._add_gallery_visualization(vis_frame, person_db)
        elif self.display_mode == 'analytics':
            vis_frame = self._add_analytics_visualization(vis_frame, track_identities, person_db)
        elif self.display_mode == 'minimal':
            # Just show the basics - already done above
            pass
        
        # Show change detection if enabled
        if getattr(self.config, 'SHOW_CHANGE_DETECTION', False):
            vis_frame = self._visualize_changes(vis_frame)
            
        return vis_frame
        
    def _draw_tracks(self, frame, tracks, track_identities, active_track_ids):
        """Draw tracks with improved visual design and information layout"""
        # Sort tracks by y-position to handle overlapping boxes (drawing back-to-front)
        sorted_tracks = sorted(tracks, key=lambda t: t.tlwh[1] + t.tlwh[3])
        
        for track in sorted_tracks:
            if track.track_id not in active_track_ids:
                continue
                
            # Get track data
            track_id = track.track_id
            tlwh = track.tlwh
            x1, y1, w, h = [int(v) for v in tlwh]
            x2, y2 = x1 + w, y1 + h
            
            # Get color for consistency - unique color per track
            color = self.colors[track_id % len(self.colors)]
            color = [int(c) for c in color]
            
            # Create semi-transparent overlay for information panel
            overlay = frame.copy()
            
            # Draw bounding box with selected style
            if self.box_style == 'rounded':
                self._draw_rounded_box(frame, (x1, y1), (x2, y2), color, 2)
            elif self.box_style == 'dashed':
                self._draw_dashed_box(frame, (x1, y1), (x2, y2), color, 2)
            else:  # 'sharp' or default
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add information panels
            info_panel_height = 70 if track_id in track_identities else 30
            
            # Top info panel (semi-transparent background)
            cv2.rectangle(overlay, (x1, y1-info_panel_height), (x2, y1), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Track ID with distinctive marker
            track_label = f"ID:{track_id}"
            cv2.putText(frame, track_label, (x1+5, y1-10), self.font, 0.5, color, 2)
            
            # Add identity if available
            if track_id in track_identities:
                identity = track_identities[track_id]
                name = identity['name']
                confidence = identity['confidence']
                quality = identity['quality']
                is_new = identity.get('is_new', False)
                
                # Name with larger, more prominent font
                cv2.putText(frame, name, (x1+5, y1-30), self.font, 0.65, (255, 255, 255), 2)
                
                # Show "New Person" indicator if this is a newly created identity
                if is_new:
                    new_indicator = "NEW"
                    text_size = cv2.getTextSize(new_indicator, self.font, 0.4, 1)[0]
                    cv2.rectangle(frame, (x2-text_size[0]-10, y1-50), (x2, y1-35), (0, 0, 200), -1)
                    cv2.putText(frame, new_indicator, (x2-text_size[0]-5, y1-40), 
                                self.font, 0.4, (255, 255, 255), 1)
                
                # Confidence and quality with clearer formatting
                conf_text = f"Conf: {confidence:.2f}"
                cv2.putText(frame, conf_text, (x1+5, y1-50), self.font, 0.45, (220, 220, 220), 1)
                
                # Draw quality bar with improved visual representation
                if self.show_quality_bar:
                    # Background bar (gray)
                    cv2.rectangle(frame, (x1, y2+5), (x2, y2+12), (100, 100, 100), -1)
                    
                    # Colored quality bar
                    quality_color = (0, 255, 0) if quality > 0.7 else (0, 255, 255) if quality > 0.5 else (0, 0, 255)
                    bar_width = int(w * quality)
                    cv2.rectangle(frame, (x1, y2+5), (x1 + bar_width, y2+12), quality_color, -1)
                    
                    # Quality label
                    cv2.putText(frame, f"Q:{quality:.2f}", (x1+w-40, y2+11), 
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)
    
    def _draw_rounded_box(self, img, pt1, pt2, color, thickness):
        """Draw a rectangle with rounded corners"""
        # Unpack points
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Radius of the rounded corners
        r = 10
        r = min(r, int((x2-x1)/4), int((y2-y1)/4))  # Make sure radius isn't too large
        
        # Draw the main lines
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)  # Top line
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)  # Bottom line
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)  # Left line
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)  # Right line
        
        # Draw the rounded corners
        cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thickness)
    
    def _draw_dashed_box(self, img, pt1, pt2, color, thickness, gap=5):
        """Draw a dashed rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw top line
        for x in range(x1, x2, gap*2):
            x_end = min(x + gap, x2)
            cv2.line(img, (x, y1), (x_end, y1), color, thickness)
            
        # Draw bottom line
        for x in range(x1, x2, gap*2):
            x_end = min(x + gap, x2)
            cv2.line(img, (x, y2), (x_end, y2), color, thickness)
            
        # Draw left line
        for y in range(y1, y2, gap*2):
            y_end = min(y + gap, y2)
            cv2.line(img, (x1, y), (x1, y_end), color, thickness)
            
        # Draw right line
        for y in range(y1, y2, gap*2):
            y_end = min(y + gap, y2)
            cv2.line(img, (x2, y), (x2, y_end), color, thickness)
    
    def _add_frame_info(self, frame, frame_count, fps, num_tracks, num_persons):
        """Add enhanced frame information overlay with better visual organization"""
        height, width = frame.shape[:2]
        
        # Create more visual top bar with gradient effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 40), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Add system title with larger font
        title = "Gait Recognition with Quality Control"
        cv2.putText(frame, title, (width//2 - 180, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Add timestamp with better formatting
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        # Add metrics panel on the right with clearer organization
        metrics_text = f"Frame: {frame_count} | FPS: {fps:.1f} | Tracks: {num_tracks} | DB: {num_persons}"
        metrics_size = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, metrics_text, (width - metrics_size[0] - 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        # Add visual indicator for performance status
        indicator_color = (0, 255, 0) if fps > 10 else (0, 255, 255) if fps > 5 else (0, 0, 255)
        cv2.circle(frame, (width - 20, 55), 5, indicator_color, -1)
        
        # Add display mode indicator
        mode_text = f"Mode: {self.display_mode.capitalize()}"
        cv2.putText(frame, mode_text, (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    def _add_debug_visualization(self, frame, tracks, person_db, gait_recognizer):
        """Add debug information visualization"""
        height, width = frame.shape[:2]
        
        # Create debug panel at bottom
        panel_height = 150
        debug_panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        # Add model info
        model_info = gait_recognizer.get_model_info()
        cv2.putText(debug_panel, f"Model: {model_info['name']} - {model_info['embedding_size']}d", 
                   (10, 20), self.font, 0.5, (255, 255, 255), 1)
        
        # Add database stats
        db_stats = person_db.get_stats()
        cv2.putText(debug_panel, f"DB: {db_stats['count']} persons, {db_stats.get('multimodal_count', 0)} with face", 
                   (10, 40), self.font, 0.5, (255, 255, 255), 1)
        
        # Add active tracks info
        track_ids = [t.track_id for t in tracks]
        cv2.putText(debug_panel, f"Active tracks: {track_ids}", 
                   (10, 60), self.font, 0.5, (255, 255, 255), 1)
        
        # Add system status
        cv2.putText(debug_panel, f"Status: {'Normal' if gait_recognizer.is_ready() else 'Loading model...'}", 
                   (10, 80), self.font, 0.5, (255, 255, 255), 1)
        
        # Add keyboard shortcuts
        cv2.putText(debug_panel, "Keys: (q)uit, (d)isplay mode, (s)ave DB, (h)elp", 
                   (10, 100), self.font, 0.5, (255, 255, 255), 1)
                   
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            mem_usage = f"Memory: {memory_info.rss / (1024 * 1024):.1f} MB"
            cv2.putText(debug_panel, mem_usage, (10, 120), self.font, 0.5, (255, 255, 255), 1)
        except ImportError:
            pass
        
        # Append debug panel to frame
        frame_with_debug = np.vstack((frame, debug_panel))
        return frame_with_debug
    
    def _add_gallery_visualization(self, frame, person_db):
        """Add person gallery visualization"""
        height, width = frame.shape[:2]
        
        # Create gallery panel at bottom
        panel_height = 120
        gallery_width = width
        gallery_panel = np.ones((panel_height, gallery_width, 3), dtype=np.uint8) * 50  # Dark gray background
        
        # Get top N persons from database
        top_persons = person_db.get_top_persons(max_count=8)
        
        if top_persons:
            # Calculate thumbnail size
            thumb_width = min(100, gallery_width // len(top_persons))
            thumb_height = min(100, panel_height - 20)
            
            # Draw each person
            for i, person in enumerate(top_persons):
                # Draw name and ID
                x_pos = i * (thumb_width + 10) + 10
                y_pos = 10
                
                # Draw colored rectangle for each person
                color = self.colors[i % len(self.colors)]
                cv2.rectangle(gallery_panel, (x_pos, y_pos), 
                             (x_pos + thumb_width, y_pos + thumb_height), color, 2)
                
                # Add name label
                name = person.get('name', f"Person {person.get('id', 'unknown')}")
                cv2.putText(gallery_panel, name, (x_pos, y_pos + thumb_height + 15), 
                           self.font, 0.4, (255, 255, 255), 1)
                
                # Add quality indicator
                quality = person.get('quality', 0.0)
                quality_text = f"Q: {quality:.2f}"
                cv2.putText(gallery_panel, quality_text, (x_pos, y_pos + thumb_height + 30), 
                           self.font, 0.4, (180, 180, 180), 1)
                           
                # Draw multimodal indicator if face data is available
                if person.get('has_face', False):
                    cv2.circle(gallery_panel, (x_pos + thumb_width - 10, y_pos + 10), 5, (0, 200, 0), -1)
        
        # Append gallery panel to frame
        frame_with_gallery = np.vstack((frame, gallery_panel))
        return frame_with_gallery
    
    def _add_analytics_visualization(self, frame, track_identities, person_db):
        """Add analytics visualization with occupancy stats and dwell times"""
        height, width = frame.shape[:2]
        
        # Create analytics panel at bottom
        panel_height = 150
        analytics_panel = np.ones((panel_height, width, 3), dtype=np.uint8) * 40  # Dark background
        
        # Draw zone occupancy stats
        cv2.putText(analytics_panel, "Zone Statistics:", (10, 20), self.font, 0.6, (200, 200, 200), 1)
        
        zone_y = 45
        for zone_name, count in self.analytics_data['zone_counts'].items():
            zone_text = f"{zone_name.capitalize()}: {count} persons"
            cv2.putText(analytics_panel, zone_text, (10, zone_y), self.font, 0.5, (180, 180, 180), 1)
            zone_y += 20
        
        # Draw dwell time histogram if we have data
        if self.analytics_data['dwell_times']:
            # Create mini histogram of dwell times
            max_dwell = max(self.analytics_data['dwell_times'].values(), default=0) + 1
            histogram_width = 300
            histogram_height = 60
            histogram = np.zeros((histogram_height, histogram_width, 3), dtype=np.uint8)
            
            # Title
            cv2.putText(analytics_panel, "Dwell Time Distribution:", (width//2 - 100, 20), 
                       self.font, 0.6, (200, 200, 200), 1)
            
            # Draw bars
            bars = {}
            for track_id, dwell_time in self.analytics_data['dwell_times'].items():
                # Group in 5 second bins
                bin_idx = min(9, int(dwell_time / 5))
                if bin_idx not in bars:
                    bars[bin_idx] = 0
                bars[bin_idx] += 1
            
            max_count = max(bars.values(), default=1)
            bar_width = histogram_width // 10
            
            for bin_idx, count in bars.items():
                bar_height = int((count / max_count) * histogram_height)
                x = bin_idx * bar_width
                y = histogram_height - bar_height
                cv2.rectangle(histogram, (x, y), (x + bar_width - 2, histogram_height), 
                             (0, 100, 200), -1)
                
                # Label
                label = f"{bin_idx*5}-{(bin_idx+1)*5}s"
                cv2.putText(histogram, label, (x, histogram_height - 5), 
                           cv2.FONT_HERSHEY_PLAIN, 0.6, (150, 150, 150), 1)
            
            # Copy histogram to panel
            analytics_panel[30:30+histogram_height, width//2-150:width//2+150] = histogram
        
        # Add recent interaction events
        cv2.putText(analytics_panel, "Recent Events:", (width-300, 20), self.font, 0.6, (200, 200, 200), 1)
        
        event_y = 45
        for i, event in enumerate(self.analytics_data['interaction_events'][-5:]):  # Show last 5 events
            event_text = f"{event['time']}: {event['description']}"
            cv2.putText(analytics_panel, event_text, (width-300, event_y), 
                       self.font, 0.4, (180, 180, 180), 1)
            event_y += 20
        
        # Append analytics panel to frame
        frame_with_analytics = np.vstack((frame, analytics_panel))
        return frame_with_analytics
    
    def _visualize_changes(self, frame):
        """Visualize changes between consecutive frames"""
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return frame
        
        # Calculate absolute difference
        diff = cv2.absdiff(frame, self.previous_frame)
        
        # Convert to grayscale and apply threshold
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Dilate to fill gaps
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw movement contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        # Update previous frame
        self.previous_frame = frame.copy()
        return frame
    
    def _add_motion_trails(self, frame, tracks, active_track_ids):
        """Add motion trails to track visualization"""
        height, width = frame.shape[:2]
        
        # Update and draw motion trails
        for track in tracks:
            track_id = track.track_id
            if track_id not in active_track_ids:
                continue
                
            # Calculate center position
            x1, y1, w, h = [int(v) for v in track.tlwh]
            center = (int(x1 + w/2), int(y1 + h))  # Use bottom-center of bounding box
            
            # Update position history
            if track_id not in self.position_history:
                self.position_history[track_id] = []
            self.position_history[track_id].append(center)
            
            # Limit history length
            if len(self.position_history[track_id]) > self.max_trail_length:
                self.position_history[track_id] = self.position_history[track_id][-self.max_trail_length:]
            
            # Draw motion trail with fading opacity
            color = self.colors[track_id % len(self.colors)]
            points = self.position_history[track_id]
            for i in range(1, len(points)):
                # Calculate opacity based on recency (newer = more opaque)
                opacity = 0.3 + 0.7 * (i / len(points))
                thickness = max(1, int(opacity * 3))
                
                cv2.line(frame, points[i-1], points[i], color, thickness)
        
        # Clean up history for tracks no longer active
        for track_id in list(self.position_history.keys()):
            if track_id not in active_track_ids:
                del self.position_history[track_id]
    
    def _add_cctv_overlay(self, frame, frame_count):
        """Add CCTV-style overlay elements"""
        height, width = frame.shape[:2]
        
        # Add timestamp in CCTV format
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to top-right corner
        cv2.putText(frame, timestamp, (width - 180, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Optionally add grid lines
        if getattr(self.config, 'SHOW_GRID', False):
            # Draw vertical grid lines
            for x in range(0, width, width // 8):
                cv2.line(frame, (x, 0), (x, height), (80, 80, 80), 1)
            
            # Draw horizontal grid lines
            for y in range(0, height, height // 6):
                cv2.line(frame, (0, y), (width, y), (80, 80, 80), 1)
        
        return frame
    
    def _blur_faces_in_frame(self, frame, tracks):
        """Blur faces in the frame for privacy"""
        blurred_frame = frame.copy()
        
        for track in tracks:
            # Get track data
            tlwh = track.tlwh
            x1, y1, w, h = [int(v) for v in tlwh]
            
            # Estimate face region (approximately upper 1/3 of the bounding box)
            face_y1 = y1
            face_y2 = y1 + int(h * 0.33)
            face_x1 = x1
            face_x2 = x1 + w
            
            # Make sure coordinates are within image bounds
            face_y1 = max(0, face_y1)
            face_y2 = min(frame.shape[0], face_y2)
            face_x1 = max(0, face_x1)
            face_x2 = min(frame.shape[1], face_x2)
            
            # Extract face region
            face_region = frame[face_y1:face_y2, face_x1:face_x2]
            
            if face_region.size > 0:  # Make sure region is not empty
                # Apply blur
                blurred_face = cv2.GaussianBlur(face_region, (23, 23), 30)
                blurred_frame[face_y1:face_y2, face_x1:face_x2] = blurred_face
        
        return blurred_frame
    
    def _update_analytics_data(self, frame, tracks, active_track_ids, frame_count):
        """Update analytics data based on tracks and frame"""
        height, width = frame.shape[:2]
        current_time = time.strftime("%H:%M:%S")
        
        # Reset zone counts for current frame
        current_zones = {zone: 0 for zone in self.zones.keys()}
        
        for track in tracks:
            track_id = track.track_id
            if track_id not in active_track_ids:
                continue
                
            # Calculate normalized position (0-1 range)
            x1, y1, w, h = [int(v) for v in track.tlwh]
            center_x = (x1 + w/2) / width
            center_y = (y1 + h/2) / height
            
            # Check which zone this track is in
            for zone_name, zone_areas in self.zones.items():
                for zone in zone_areas:
                    x1_zone, y1_zone, x2_zone, y2_zone = zone
                    if (x1_zone <= center_x <= x2_zone and y1_zone <= center_y <= y2_zone):
                        current_zones[zone_name] += 1
                        
                        # Record when track entered a zone if not already recorded
                        zone_key = f"{track_id}_{zone_name}"
                        if zone_key not in self.analytics_data.get('zone_entries', {}):
                            self.analytics_data.setdefault('zone_entries', {})[zone_key] = frame_count
                            
                            # Log event
                            self.analytics_data['interaction_events'].append({
                                'time': current_time,
                                'track_id': track_id,
                                'description': f"Track {track_id} entered {zone_name} zone"
                            })
            
            # Update dwell times
            if track_id not in self.analytics_data['dwell_times']:
                self.analytics_data['dwell_times'][track_id] = 0
            else:
                # Assuming 30fps, increment by 1/30th of a second
                self.analytics_data['dwell_times'][track_id] += 1/30
        
        # Update zone counts
        self.analytics_data['zone_counts'] = current_zones
        
        # Limit event history
        if len(self.analytics_data['interaction_events']) > 100:
            self.analytics_data['interaction_events'] = self.analytics_data['interaction_events'][-100:]
    
    def _draw_zone_overlays(self, frame):
        """Draw zone overlays on frame"""
        height, width = frame.shape[:2]
        
        for zone_name, zone_areas in self.zones.items():
            # Assign different color per zone
            if zone_name == 'entry':
                color = (0, 200, 0)  # Green
            elif zone_name == 'exit':
                color = (0, 0, 200)  # Red
            else:
                color = (200, 200, 0)  # Yellow
                
            for zone in zone_areas:
                x1_rel, y1_rel, x2_rel, y2_rel = zone
                x1 = int(x1_rel * width)
                y1 = int(y1_rel * height)
                x2 = int(x2_rel * width)
                y2 = int(y2_rel * height)
                
                # Draw semi-transparent zone
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                # Draw zone border
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw zone name
                cv2.putText(frame, zone_name.upper(), (x1+5, y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def _generate_colors(self, n):
        """Generate n visually distinct colors"""
        hsv_colors = [(i / n, 0.8, 0.9) for i in range(n)]
        colors = []
        for h, s, v in hsv_colors:
            # Convert HSV to BGR
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append((int(b * 255), int(g * 255), int(r * 255)))
        return colors
    
    def set_display_mode(self, mode):
        """Set display mode"""
        if mode in ['standard', 'debug', 'gallery', 'analytics', 'minimal']:
            self.display_mode = mode
            return True
        return False
    
    def toggle_motion_trails(self):
        """Toggle motion trails on/off"""
        self.show_motion_trails = not self.show_motion_trails
        return self.show_motion_trails
    
    def toggle_face_blur(self):
        """Toggle face blurring for privacy"""
        self.blur_faces = not self.blur_faces
        return self.blur_faces
    
    def cycle_box_style(self):
        """Cycle through bounding box styles"""
        styles = ['sharp', 'rounded', 'dashed']
        current_idx = styles.index(self.box_style)
        self.box_style = styles[(current_idx + 1) % len(styles)]
        return self.box_style