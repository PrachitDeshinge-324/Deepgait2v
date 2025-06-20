"""
Keyboard input handler for the gait recognition system
"""

import cv2
import os
from src.utils.device import vprint

class KeyboardHandler:
    """Handles keyboard input for interactive commands"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.display_modes = ['standard', 'debug', 'gallery', 'minimal']
        self.current_display_mode_index = 0
        
    def handle_key(self, key, gait_recognizer, database_handler, db_path):
        """
        Handle keyboard input
        
        Args:
            key: Key code from cv2.waitKey
            gait_recognizer: Gait recognizer object for model commands
            database_handler: Database handler for database commands
            db_path: Path to database for save operations
            
        Returns:
            True if application should exit, False otherwise
        """
        # Skip if no key pressed
        if key == -1:
            return False
        
        # 'q' - Quit application
        if key == ord('q'):
            vprint("Quit requested by user")
            return True
        
        # 's' - Save database
        elif key == ord('s'):
            vprint("Saving database...")
            database_handler.save_database(db_path)
            vprint(f"Database saved to {db_path}")
        
        # 'd' - Toggle display mode
        elif key == ord('d'):
            self.current_display_mode_index = (self.current_display_mode_index + 1) % len(self.display_modes)
            new_mode = self.display_modes[self.current_display_mode_index]
            vprint(f"Switched display mode to: {new_mode}")
            self.config.DISPLAY_MODE = new_mode
        
        # 'r' - Toggle real-time mode (slower but more accurate)
        elif key == ord('r'):
            self.config.REAL_TIME_MODE = not getattr(self.config, 'REAL_TIME_MODE', True)
            vprint(f"Real-time mode: {'ON' if self.config.REAL_TIME_MODE else 'OFF'}")
        
        # 'c' - Clear database
        elif key == ord('c'):
            vprint("Clearing person database...")
            database_handler.clear_database()
            vprint("Database cleared")
        
        # 'h' - Show help
        elif key == ord('h'):
            self._show_help()
            
        # Space - Pause/resume
        elif key == 32:  # Space key
            self.config.PAUSED = not getattr(self.config, 'PAUSED', False)
            vprint(f"Playback {'paused' if self.config.PAUSED else 'resumed'}")
            
            # If paused, wait for another keypress
            if self.config.PAUSED:
                while True:
                    key = cv2.waitKey(100)
                    if key == 32:  # Space again to resume
                        self.config.PAUSED = False
                        vprint("Playback resumed")
                        break
                    elif key == ord('q'):  # Quit while paused
                        vprint("Quit requested by user while paused")
                        return True
        
        return False
    
    def _show_help(self):
        """Show help information"""
        help_text = """
        Keyboard Controls:
        ------------------
        q - Quit application
        s - Save database
        d - Toggle display mode (standard/debug/gallery/minimal)
        r - Toggle real-time processing mode
        c - Clear database
        m - Switch gait recognition model
        h - Show this help
        Space - Pause/resume playback
        """
        print(help_text)