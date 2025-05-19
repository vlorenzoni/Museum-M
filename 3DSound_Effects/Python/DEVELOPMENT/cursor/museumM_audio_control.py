import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
import json
import time
from scipy.spatial import Delaunay

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

@dataclass
class AudioEvent:
    """Class to represent an audio event with timing and parameters"""
    start_time: float  # in seconds
    duration: float    # in seconds
    event_type: str   # 'pan', 'crossfade', 'expand', 'static'
    parameters: dict  # event-specific parameters
    
    def __post_init__(self):
        """Validate event parameters"""
        # Get the number of speakers from the layout file
        speakers_df = pd.read_csv(SCRIPT_DIR / "layout_8loudspeaker_LOV_GENELEC.csv", sep=';', header=None)
        speakers_df = speakers_df.dropna(how='all')
        num_speakers = len(speakers_df)
        
        if self.event_type == 'static':
            if 'speakers' in self.parameters and 'position' in self.parameters:
                raise ValueError("Static event cannot have both 'speakers' and 'position' parameters")
            if 'speakers' in self.parameters:
                if 'gains' not in self.parameters:
                    raise ValueError("Static event with speakers requires 'gains' parameter")
                if len(self.parameters['speakers']) != len(self.parameters['gains']):
                    raise ValueError("Number of speakers must match number of gains")
                for speaker in self.parameters['speakers']:
                    if not 1 <= speaker <= num_speakers:
                        raise ValueError(f"Speaker number must be between 1 and {num_speakers}")
            elif 'position' in self.parameters:
                if not 1 <= self.parameters['position'] <= num_speakers:
                    raise ValueError(f"Position must be between 1 and {num_speakers}")
                # Allow either overall_gain or gains parameter
                if 'gains' in self.parameters and 'overall_gain' in self.parameters:
                    raise ValueError("Cannot specify both 'gains' and 'overall_gain'")
                if 'gains' in self.parameters and len(self.parameters['gains']) != 2:
                    raise ValueError("For intermediate positions, gains must be a list of 2 values")
            else:
                raise ValueError("Static event requires either 'speakers' or 'position' parameter")
                
        elif self.event_type == 'pan':
            if 'start_position' not in self.parameters or 'end_position' not in self.parameters:
                raise ValueError("Pan event requires 'start_position' and 'end_position' parameters")
            if not 1 <= self.parameters['start_position'] <= num_speakers:
                raise ValueError(f"Start position must be between 1 and {num_speakers}")
            if not 1 <= self.parameters['end_position'] <= num_speakers:
                raise ValueError(f"End position must be between 1 and {num_speakers}")
                
        elif self.event_type == 'expand':
            if 'start_position' not in self.parameters:
                raise ValueError("Expand event requires 'start_position' parameter")
            if not 1 <= self.parameters['start_position'] <= num_speakers:
                raise ValueError(f"Start position must be between 1 and {num_speakers}")
            if 'expansion_interval' not in self.parameters:
                self.parameters['expansion_interval'] = 5.0  # Default value
                
        elif self.event_type == 'crossfade':
            if 'direction' not in self.parameters:
                raise ValueError("Crossfade event requires 'direction' parameter")
            if self.parameters['direction'] not in ['dry_to_wet', 'wet_to_dry']:
                raise ValueError("Crossfade direction must be 'dry_to_wet' or 'wet_to_dry'")
        else:
            raise ValueError(f"Unknown event type: {self.event_type}")

class MuseumMAudioControl:
    def __init__(self, 
                 dry_file: str = "sollazzo_48k.wav",
                 wet_file: str = "SDM_Sollazzo_Brou_church.wav",
                 speaker_layout: str = "layout_8loudspeaker_LOV_GENELEC.csv",
                 output_dir: str = "audio_output",
                 device_name: str = "Digiface USB",
                 sample_rate: int = 48000,
                 gain: float = None,  # Changed default to None
                 max_amplitude: float = 0.8,  # Maximum allowed amplitude
                 transition_time: float = 0.2):  # Increased default transition time to 0.2 seconds
        """
        Initialize the audio control system with safety parameters
        
        Args:
            dry_file: Path to the dry audio file
            wet_file: Path to the wet audio file
            speaker_layout: Path to the speaker layout CSV file
            output_dir: Directory to save output files
            device_name: Name of the audio device to use
            sample_rate: Sample rate in Hz
            gain: Audio gain (0.0 to 1.0). If None, will be set from config file
            max_amplitude: Maximum allowed amplitude (0.0 to 1.0)
            transition_time: Time for smooth transitions between events in seconds
            
        Speaker Layout:
            - Speaker 1: Front Left
            - Speaker 2: Front Center
            - Speaker 3: Front Right
            - Speaker 4: Side Right
            - Speaker 5: Back Right
            - Speaker 6: Back Center
            - Speaker 7: Back Left
            - Speaker 8: Side Left
        """
        self.dry_file = str(SCRIPT_DIR / dry_file)
        self.wet_file = str(SCRIPT_DIR / wet_file)
        self.speaker_layout = str(SCRIPT_DIR / speaker_layout)
        self.output_dir = Path(output_dir)
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.gain = gain  # Don't set default here, will be set from config
        self.max_amplitude = max_amplitude
        self.transition_time = transition_time
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load and check audio files
        self._load_and_check_audio()
        
        # Load speaker positions
        self.speaker_positions = self._load_speaker_positions()
        self.num_speakers = len(self.speaker_positions)
        
        # Set up VBAP
        self._setup_vbap()
        
        # Set up audio device
        self._setup_audio_device()
        
        # Initialize event list
        self.events: List[AudioEvent] = []
        
    def _load_and_check_audio(self):
        """Load and check audio files for safety"""
        # Load audio files
        self.dry_audio, self.fs_dry = sf.read(self.dry_file)
        self.wet_audio, self.fs_wet = sf.read(self.wet_file)
        
        if self.fs_dry != self.fs_wet:
            raise ValueError("Sample rates of dry and wet files must match")
            
        # Convert stereo to mono if needed
        if len(self.dry_audio.shape) > 1:
            self.dry_audio = np.mean(self.dry_audio, axis=1)
        if len(self.wet_audio.shape) > 1:
            self.wet_audio = np.mean(self.wet_audio, axis=1)
            
        # Check for audio spikes
        self._check_audio_spikes(self.dry_audio, "dry")
        self._check_audio_spikes(self.wet_audio, "wet")
        
        # Normalize audio to prevent clipping
        self.dry_audio = self._normalize_audio(self.dry_audio)
        self.wet_audio = self._normalize_audio(self.wet_audio)
        
    def _check_audio_spikes(self, audio: np.ndarray, name: str):
        """Check for audio spikes in the input file"""
        max_amp = np.max(np.abs(audio))
        if max_amp > self.max_amplitude:
            print(f"Warning: {name} audio file contains spikes (max amplitude: {max_amp:.2f})")
            print("Consider normalizing the input audio file")
            
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping"""
        max_amp = np.max(np.abs(audio))
        if max_amp > self.max_amplitude:
            return audio * (self.max_amplitude / max_amp)
        return audio
        
    def _create_transition_window(self, duration: int, fade_in: bool = True, fade_out: bool = True) -> np.ndarray:
        """Create a smooth transition window with sine-based fade in/out
        
        Args:
            duration: Number of samples
            fade_in: Whether to include fade in
            fade_out: Whether to include fade out
        """
        window = np.ones(duration)
        transition_samples = int(self.transition_time * self.sample_rate)
        
        # Create fade in using sine curve for even smoother transition
        if fade_in and duration > transition_samples:
            # Use sine curve with adjusted phase for smoother start
            fade_in = np.sin(np.linspace(0, np.pi/2, transition_samples))
            window[:transition_samples] = fade_in
            
        # Create fade out using sine curve for even smoother transition
        if fade_out and duration > transition_samples:
            # Use sine curve with adjusted phase for smoother end
            fade_out = np.sin(np.linspace(np.pi/2, np.pi, transition_samples))
            window[-transition_samples:] = fade_out
            
        return window
        
    def _load_speaker_positions(self) -> np.ndarray:
        """Load speaker positions from CSV file"""
        speakers_df = pd.read_csv(self.speaker_layout, sep=';', header=None)
        # Remove empty rows and get only the first 3 columns
        speakers_df = speakers_df.dropna(how='all')
        # Convert to numpy array and ensure numeric values
        positions = speakers_df.iloc[:, :3].values.astype(float)
        return positions
        
    def _setup_audio_device(self):
        """Set up the audio device"""
        devices = sd.query_devices()
        device_found = False
        
        for dev in devices:
            if self.device_name in dev['name']:
                sd.default.device = dev['name']
                sd.default.channels = self.num_speakers
                sd.default.samplerate = self.sample_rate
                device_found = True
                break
                
        if not device_found:
            raise ValueError(f"Audio device {self.device_name} not found")
            
    def add_event(self, event: AudioEvent):
        """Add an audio event to the sequence"""
        self.events.append(event)
        
    def _calculate_position_gains(self, position: float, overall_gain: float = 1.0) -> Tuple[np.ndarray, List[int]]:
        """Calculate gains for an intermediate position between speakers using VBAP
        
        Args:
            position: Position between speakers (e.g., 1.5 is between speaker 1 and 2)
            overall_gain: Overall gain at the center of the room (default: 1.0)
        """
        # Get the two closest speakers (1-based indexing)
        floor_pos = int(np.floor(position))
        ceil_pos = int(np.ceil(position))
        
        if floor_pos == ceil_pos:
            # Exact speaker position
            gains = np.zeros(self.num_speakers)
            gains[floor_pos - 1] = overall_gain  # Convert to 0-based index for array
            return gains, [floor_pos]
        
        # Calculate intermediate position in Cartesian coordinates
        floor_coords = self.speaker_positions[floor_pos - 1]
        ceil_coords = self.speaker_positions[ceil_pos - 1]
        frac = position - floor_pos
        current_pos = floor_coords * (1 - frac) + ceil_coords * frac
        
        # Find containing triangle
        triangle = self._find_vbap_triangle(current_pos)
        if triangle is not None:
            # Calculate VBAP gains
            gains = self._calculate_vbap_gains(current_pos, triangle)
            # Apply overall gain
            gains = gains * overall_gain
        else:
            # If position is outside the speaker array, use nearest speaker
            distances = np.linalg.norm(self.speaker_positions[:, :2] - current_pos[:2], axis=1)
            nearest_speaker = np.argmin(distances)
            gains = np.zeros(self.num_speakers)
            gains[nearest_speaker] = overall_gain
        
        return gains, [floor_pos, ceil_pos]
        
    def _create_static_gains(self, parameters: dict) -> np.ndarray:
        """Create gains for static playback from one or more speakers"""
        gains = np.zeros(self.num_speakers)
        max_safe_gain = 0.8  # Maximum safe gain per speaker
        
        if 'speakers' in parameters:
            # Multiple speakers with custom gains (1-based indexing)
            total_gain = sum(parameters['gains'])  # Sum of all gains
            if total_gain > 0:
                # Calculate normalization factor but ensure it doesn't make any gain exceed max_safe_gain
                norm_factor = min(1.0 / total_gain, max_safe_gain / max(parameters['gains']))
                normalized_gains = [g * norm_factor for g in parameters['gains']]
                for speaker, gain in zip(parameters['speakers'], normalized_gains):
                    gains[speaker - 1] = gain  # Convert to 0-based index for array
        else:
            # Intermediate position with VBAP
            overall_gain = min(parameters.get('overall_gain', 1.0), max_safe_gain)
            gains, _ = self._calculate_position_gains(
                parameters['position'],
                overall_gain
            )
            
        # Ensure total gain is constant
        total_gain = np.sum(np.abs(gains))
        if total_gain > 0:
            gains = gains / total_gain
            
        return gains
        
    def _setup_vbap(self):
        """Set up VBAP triangulation for the speaker layout"""
        # Extract only x and y coordinates for 2D triangulation
        xy_positions = self.speaker_positions[:, :2]
        
        # Create 2D triangulation of speaker positions
        self.tri = Delaunay(xy_positions)
        
        # Store speaker triangles for quick lookup
        self.speaker_triangles = {}
        for simplex in self.tri.simplices:
            for speaker in simplex:
                if speaker not in self.speaker_triangles:
                    self.speaker_triangles[speaker] = []
                self.speaker_triangles[speaker].append(simplex)
    
    def _find_vbap_triangle(self, position: np.ndarray) -> Optional[np.ndarray]:
        """Find the triangle containing the target position"""
        # Use only x and y coordinates for 2D triangulation
        xy_position = position[:2]
        
        # Find the simplex containing the position
        simplex_idx = self.tri.find_simplex(xy_position)
        if simplex_idx == -1:
            return None
        return self.tri.simplices[simplex_idx]
    
    def _calculate_vbap_gains(self, position: np.ndarray, triangle: np.ndarray) -> np.ndarray:
        """Calculate VBAP gains for a position within a triangle"""
        max_safe_gain = 0.8  # Maximum safe gain per speaker
        
        # Get the three speaker positions forming the triangle (using only x and y)
        p1, p2, p3 = self.speaker_positions[triangle, :2]
        
        # Create the matrix of speaker positions (2D)
        L = np.array([p1, p2, p3]).T
        
        # Calculate the inverse of L
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # If matrix is singular, use equal gains
            g = np.ones(3) / max(3, 1/max_safe_gain)  # Ensure each gain doesn't exceed max_safe_gain
        else:
            # Calculate gains using only x and y coordinates
            g = L_inv @ position[:2]
            
            # Ensure non-negative gains and apply safety limit
            g = np.maximum(g, 0)
            if np.sum(g) > 0:
                # Normalize but ensure no gain exceeds max_safe_gain
                scale_factor = min(1.0 / np.sum(g), max_safe_gain / np.max(g))
                g = g * scale_factor
        
        # Create full gain vector for all speakers
        gains = np.zeros(self.num_speakers)
        gains[triangle] = g
        
        return gains
    
    def _create_pan_gains(self, start_position: float, end_position: float, 
                         duration: float) -> np.ndarray:
        """Create gains for VBAP panning between two positions"""
        num_samples = int(duration * self.sample_rate)
        gains = np.zeros((num_samples, self.num_speakers))
        max_safe_gain = 0.8  # Maximum safe gain per speaker
        
        # Convert speaker-based positions to Cartesian coordinates
        start_pos = self.speaker_positions[int(np.floor(start_position)) - 1]
        end_pos = self.speaker_positions[int(np.floor(end_position)) - 1]
        
        # Create transition window for smooth fade in/out
        transition_window = self._create_transition_window(num_samples)
        
        # Create smooth position interpolation using raised cosine
        t = np.linspace(0, 1, num_samples)
        t_smooth = 0.5 * (1 - np.cos(np.pi * t))  # Raised cosine for smoother movement
        
        for i in range(num_samples):
            # Interpolate position using smooth curve
            current_pos = start_pos * (1 - t_smooth[i]) + end_pos * t_smooth[i]
            
            # Find containing triangle
            triangle = self._find_vbap_triangle(current_pos)
            if triangle is not None:
                # Calculate VBAP gains
                gains[i] = self._calculate_vbap_gains(current_pos, triangle)
            else:
                # If position is outside the speaker array, use nearest speaker
                distances = np.linalg.norm(self.speaker_positions[:, :2] - current_pos[:2], axis=1)
                nearest_speaker = np.argmin(distances)
                gains[i, nearest_speaker] = max_safe_gain
            
            # Apply transition window
            gains[i] = gains[i] * transition_window[i]
            
            # Ensure constant total gain
            total_gain = np.sum(np.abs(gains[i]))
            if total_gain > 0:
                gains[i] = gains[i] / total_gain
            
        return gains
        
    def _create_crossfade_gains(self, duration: float, direction: str, 
                              overall_gain: float = 1.0,
                              speakers: List[int] = None,
                              shift_percentage: float = 1.0,
                              shift_time: float = None) -> np.ndarray:
        """Create gains for crossfading between dry and wet sounds with constant total gain
        
        Args:
            duration: Duration of the crossfade in seconds
            direction: 'dry_to_wet' or 'wet_to_dry'
            overall_gain: Overall gain at the center of the room (default: 1.0)
            speakers: List of speaker numbers to crossfade (1-based indexing)
            shift_percentage: Percentage of shift (0.0 to 1.0)
            shift_time: Time in seconds for the shift to occur (default: full duration)
        """
        num_samples = int(duration * self.sample_rate)
        gains = np.zeros((num_samples, self.num_speakers, 2))  # 2 for dry/wet
        
        # Create transition window for smooth fade in/out
        transition_window = self._create_transition_window(num_samples)
        
        # If no shift time specified, use full duration
        if shift_time is None:
            shift_time = duration
            
        # Calculate number of samples for the shift
        shift_samples = int(shift_time * self.sample_rate)
        
        # Create smooth crossfade using raised cosine
        t = np.linspace(0, 1, shift_samples)
        t_smooth = 0.5 * (1 - np.cos(np.pi * t))  # Raised cosine for smoother crossfade
        
        # Apply shift percentage
        t_smooth = t_smooth * shift_percentage
        
        if direction == 'dry_to_wet':
            dry_gain = 1 - t_smooth
            wet_gain = t_smooth
        else:  # wet_to_dry
            dry_gain = t_smooth
            wet_gain = 1 - t_smooth
        
        # Extend the gains to full duration
        if shift_samples < num_samples:
            # If shift is shorter than duration, maintain the final state
            dry_gain = np.pad(dry_gain, (0, num_samples - shift_samples), mode='edge')
            wet_gain = np.pad(wet_gain, (0, num_samples - shift_samples), mode='edge')
        
        # Apply transition window to both dry and wet gains
        dry_gain = dry_gain * transition_window
        wet_gain = wet_gain * transition_window
        
        # If no speakers specified, use all speakers
        if speakers is None:
            speakers = list(range(1, self.num_speakers + 1))
            
        # Apply gains to specified speakers
        for speaker in speakers:
            if 1 <= speaker <= self.num_speakers:
                gains[:, speaker-1, 0] = dry_gain * overall_gain
                gains[:, speaker-1, 1] = wet_gain * overall_gain
        
        # Ensure constant total gain for each time step
        for i in range(num_samples):
            total_gain = np.sum(np.abs(gains[i, :, 0])) + np.sum(np.abs(gains[i, :, 1]))
            if total_gain > 0:
                gains[i] = gains[i] / total_gain
            
        return gains
        
    def _create_expansion_gains(self, start_position: float, duration: float, 
                              expansion_interval: float) -> np.ndarray:
        """Create gains for expanding sound to neighboring speakers"""
        num_samples = int(duration * self.sample_rate)
        gains = np.zeros((num_samples, self.num_speakers))
        
        # Create transition window for smooth fade in/out
        transition_window = self._create_transition_window(num_samples)
        
        # Start with initial position
        position_gains, _ = self._calculate_position_gains(start_position)
        gains[:, :] = position_gains
        
        # Expand every expansion_interval seconds
        samples_per_expansion = int(expansion_interval * self.sample_rate)
        num_expansions = int(duration // expansion_interval)
        
        # Create a smoother fade-in curve using a raised cosine
        def raised_cosine(x):
            return 0.5 * (1 - np.cos(np.pi * x))
        
        for i in range(1, num_expansions + 1):
            start_sample = i * samples_per_expansion
            if start_sample >= num_samples:
                break
                
            # Add speakers to the right and left
            right_pos = start_position + i
            left_pos = start_position - i
            
            end_sample = min(start_sample + samples_per_expansion, num_samples)
            current_samples = end_sample - start_sample
            
            # Create smooth fade in for this expansion using raised cosine
            fade_in = raised_cosine(np.linspace(0, 1, current_samples))
            
            # Add right expansion with reduced gain
            if 0 <= right_pos <= 7:
                right_gains, _ = self._calculate_position_gains(right_pos)
                # Reduce the gain of new speakers to prevent sudden bursts
                right_gains = right_gains * 0.5  # Reduce by 50%
                gains[start_sample:end_sample] += right_gains * fade_in[:, np.newaxis]
                
            # Add left expansion with reduced gain
            if 0 <= left_pos <= 7:
                left_gains, _ = self._calculate_position_gains(left_pos)
                # Reduce the gain of new speakers to prevent sudden bursts
                left_gains = left_gains * 0.5  # Reduce by 50%
                gains[start_sample:end_sample] += left_gains * fade_in[:, np.newaxis]
                
        # Apply transition window
        gains = gains * transition_window[:, np.newaxis]
            
        # Ensure constant total gain for each time step
        for i in range(num_samples):
            total_gain = np.sum(np.abs(gains[i]))
            if total_gain > 0:
                gains[i] = gains[i] / total_gain
            
        return gains
        
    def _validate_event_times(self):
        """Validate the timing of events to ensure continuous coverage without overlaps"""
        if not self.events:
            raise ValueError("No events defined in the sequence")
            
        # Sort events by start time
        sorted_events = sorted(self.events, key=lambda x: x.start_time)
        
        # Check for gaps and overlaps
        errors = []
        warnings = []
        
        # Check first event starts at 0
        if sorted_events[0].start_time > 0:
            warnings.append(f"First event starts at {sorted_events[0].start_time}s, not at 0s")
        
        # Check for gaps and overlaps between consecutive events
        for i in range(len(sorted_events) - 1):
            current_event = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            current_end = current_event.start_time + current_event.duration
            next_start = next_event.start_time
            
            # Check for gap
            if current_end < next_start:
                warnings.append(f"Gap between events {i} and {i+1}: {next_start - current_end:.2f}s gap")
                # Extend the current event's duration to fill the gap
                current_event.duration = next_start - current_event.start_time
            
            # Check for overlap
            if current_end > next_start:
                errors.append(f"Overlap between events {i} and {i+1}: {current_end - next_start:.2f}s overlap")
        
        # Display warnings and errors
        if warnings:
            print("\n=== Timing Warnings ===")
            for warning in warnings:
                print(f"Warning: {warning}")
            print("=====================\n")
            
        if errors:
            print("\n=== Timing Errors ===")
            for error in errors:
                print(f"Error: {error}")
            print("\nEvent sequence has timing issues. Please fix the following:")
            print("1. Remove any overlaps between events")
            print("2. Ensure events are continuous or explicitly add silence")
            print("3. Consider starting the first event at 0s")
            print("=====================\n")
            raise ValueError("Event sequence has timing issues")
            
    def generate_audio(self) -> np.ndarray:
        """Generate the complete audio sequence based on events"""
        # Validate event timing before proceeding
        self._validate_event_times()
        
        # Sort events by start time
        self.events.sort(key=lambda x: x.start_time)
        
        # Calculate total duration
        total_duration = max(event.start_time + event.duration for event in self.events)
        total_samples = int(total_duration * self.sample_rate)
        
        # Initialize output arrays
        dry_output = np.zeros((total_samples, self.num_speakers))
        wet_output = np.zeros((total_samples, self.num_speakers))
        
        # Keep track of the last event's gains for filling gaps
        last_dry_gains = None
        last_wet_gains = None
        
        for event in self.events:
            start_sample = int(event.start_time * self.sample_rate)
            duration_samples = int(event.duration * self.sample_rate)
            
            # Create transition window
            transition_window = self._create_transition_window(duration_samples)
            
            if event.event_type == 'static':
                gains = self._create_static_gains(event.parameters)
                gains = gains * transition_window[:, np.newaxis]
                dry_output[start_sample:start_sample + duration_samples] = gains
                last_dry_gains = gains[-1]  # Store the last gain values
                last_wet_gains = None
                
            elif event.event_type == 'pan':
                gains = self._create_pan_gains(
                    event.parameters['start_position'],
                    event.parameters['end_position'],
                    event.duration
                )
                gains = gains * transition_window[:, np.newaxis]
                dry_output[start_sample:start_sample + duration_samples] = gains
                last_dry_gains = gains[-1]  # Store the last gain values
                last_wet_gains = None
                
            elif event.event_type == 'crossfade':
                gains = self._create_crossfade_gains(
                    event.duration,
                    event.parameters['direction']
                )
                gains = gains * transition_window[:, np.newaxis, np.newaxis]
                dry_output[start_sample:start_sample + duration_samples] = gains[:, :, 0]
                wet_output[start_sample:start_sample + duration_samples] = gains[:, :, 1]
                last_dry_gains = gains[-1, :, 0]  # Store the last gain values
                last_wet_gains = gains[-1, :, 1]
                
            elif event.event_type == 'expand':
                gains = self._create_expansion_gains(
                    event.parameters['start_position'],
                    event.duration,
                    event.parameters.get('expansion_interval', 5.0)
                )
                gains = gains * transition_window[:, np.newaxis]
                wet_output[start_sample:start_sample + duration_samples] = gains
                last_dry_gains = None
                last_wet_gains = gains[-1]  # Store the last gain values
                
            # Fill any remaining time with the last event's gains
            if start_sample + duration_samples < total_samples:
                remaining_samples = total_samples - (start_sample + duration_samples)
                if last_dry_gains is not None:
                    dry_output[start_sample + duration_samples:] = np.tile(last_dry_gains, (remaining_samples, 1))
                if last_wet_gains is not None:
                    wet_output[start_sample + duration_samples:] = np.tile(last_wet_gains, (remaining_samples, 1))
                
        # Apply dry and wet audio
        output = np.zeros((total_samples, self.num_speakers))
        
        # Ensure we have enough audio data
        if len(self.dry_audio) < total_samples:
            self.dry_audio = np.tile(self.dry_audio, int(np.ceil(total_samples / len(self.dry_audio))))
        if len(self.wet_audio) < total_samples:
            self.wet_audio = np.tile(self.wet_audio, int(np.ceil(total_samples / len(self.wet_audio))))
            
        # Apply gains to audio
        for i in range(self.num_speakers):
            output[:, i] = (self.dry_audio[:total_samples] * dry_output[:, i] +
                          self.wet_audio[:total_samples] * wet_output[:, i])
                          
        # Final safety check
        max_amp = np.max(np.abs(output))
        if max_amp > self.max_amplitude:
            print(f"Warning: Output contains spikes (max amplitude: {max_amp:.2f})")
            print("Normalizing output to prevent clipping")
            output = self._normalize_audio(output)
            
        return output
        
    def save_audio(self, output: np.ndarray, filename: str):
        """Save the generated audio to a file"""
        output_path = self.output_dir / filename
        sf.write(output_path, output * self.gain, self.sample_rate)
        
    def save_individual_speakers(self, output: np.ndarray, base_filename: str):
        """Save each speaker channel as a separate WAV file"""
        for i in range(self.num_speakers):
            speaker_output = output[:, i:i+1]  # Keep 2D shape for mono
            speaker_path = self.output_dir / f"{base_filename}_speaker_{i}.wav"
            sf.write(speaker_path, speaker_output * self.gain, self.sample_rate)
        
    def play_audio(self, output: np.ndarray):
        """Play the generated audio with gain control and display time position and event info"""
        import time
        
        # Calculate total duration
        total_duration = len(output) / self.sample_rate
        total_minutes = int(total_duration // 60)
        total_seconds = int(total_duration % 60)
        
        # Sort events by start time for easy lookup
        sorted_events = sorted(self.events, key=lambda x: x.start_time)
        
        # Start playback with the gain from config
        print(f"\nStarting playback with gain: {self.gain:.3f}")
        sd.play(output * self.gain, self.sample_rate)
        
        # Update time display while playing
        start_time = time.time()
        last_event = None
        while True:
            current_time = time.time() - start_time
            current_minutes = int(current_time // 60)
            current_seconds = int(current_time % 60)
            
            # Find current event
            current_event = None
            for event in sorted_events:
                if event.start_time <= current_time <= (event.start_time + event.duration):
                    current_event = event
                    break
            
            # Clear the line and print current time and event info
            if current_event:
                # Only print new event info when it changes
                if current_event != last_event:
                    print()  # New line for new event
                    last_event = current_event
                
                # Get position info for display
                position_str = ""
                if current_event.event_type == 'static':
                    if 'position' in current_event.parameters:
                        pos = current_event.parameters['position']
                        if pos.is_integer():
                            position_str = f" | Position: {int(pos)}"
                        else:
                            position_str = f" | Position: {pos:.1f}"
                    elif 'speakers' in current_event.parameters:
                        speakers = current_event.parameters['speakers']
                        if len(speakers) == 1:
                            position_str = f" | Position: {speakers[0]}"
                        else:
                            position_str = f" | Positions: {speakers}"
                elif current_event.event_type == 'pan':
                    start_pos = current_event.parameters['start_position']
                    end_pos = current_event.parameters['end_position']
                    if start_pos.is_integer() and end_pos.is_integer():
                        position_str = f" | From: {int(start_pos)} → To: {int(end_pos)}"
                    else:
                        position_str = f" | From: {start_pos:.1f} → To: {end_pos:.1f}"
                elif current_event.event_type == 'expand':
                    center_pos = current_event.parameters['start_position']
                    if center_pos.is_integer():
                        position_str = f" | Center: {int(center_pos)}"
                    else:
                        position_str = f" | Center: {center_pos:.1f}"
                
                # Print time, effect, and position
                print(f"\rTime: {current_minutes:02d}:{current_seconds:02d} / {total_minutes:02d}:{total_seconds:02d} | Effect: {current_event.event_type}{position_str}", end="")
            else:
                print(f"\rTime: {current_minutes:02d}:{current_seconds:02d} / {total_minutes:02d}:{total_seconds:02d} | No effect", end="")
            
            if current_time >= total_duration:
                break
            time.sleep(0.1)  # Update every 100ms
            
        print("\nPlayback complete!")
        sd.wait()
        
    def save_events(self, filename: str):
        """Save the event sequence to a JSON file"""
        events_data = []
        for event in self.events:
            events_data.append({
                'start_time': event.start_time,
                'duration': event.duration,
                'event_type': event.event_type,
                'parameters': event.parameters
            })
            
        with open(self.output_dir / filename, 'w') as f:
            json.dump(events_data, f, indent=4)
            
    def set_gain(self, gain: float):
        """Set the gain value for audio playback and saving"""
        if not 0.0 <= gain <= 1.0:
            raise ValueError("Gain must be between 0.0 and 1.0")
        self.gain = gain
        print(f"Gain set to: {gain:.3f}")

    def load_events(self, filename: str):
        """Load an event sequence from a config file (Python or JSON)"""
        config_path = SCRIPT_DIR / filename  # Use SCRIPT_DIR to get the correct path
        print(f"\nLoading configuration from: {config_path}")
        
        if filename.endswith('.py'):
            # Load Python config file
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", str(config_path))
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            # Debug print the config contents
            print("\nConfig file contents:")
            if hasattr(config, 'AUDIO_CONFIG'):
                print("AUDIO_CONFIG found:")
                print(config.AUDIO_CONFIG)
                
                # Set gain from config
                if 'gain' in config.AUDIO_CONFIG:
                    new_gain = float(config.AUDIO_CONFIG['gain'])
                    print(f"\nSetting gain from AUDIO_CONFIG: {new_gain:.3f}")
                    self.gain = new_gain
                else:
                    print("\nWarning: No gain specified in AUDIO_CONFIG")
                    if self.gain is None:
                        self.gain = 0.01  # Only set default if not specified anywhere
            else:
                print("\nWarning: AUDIO_CONFIG not found in config file")
                if self.gain is None:
                    self.gain = 0.01  # Only set default if not specified anywhere
            
            # Load events
            self.events = []
            if hasattr(config, 'EVENTS_CONFIG'):
                print("\nLoading events from EVENTS_CONFIG:")
                for event_data in config.EVENTS_CONFIG:
                    # Convert 'function' to 'event_type' for compatibility
                    event_type = event_data.pop('function')
                    print(f"\nEvent: {event_type}")
                    print(f"Parameters: {event_data['parameters']}")
                    print(f"Start time: {event_data['start_time']}")
                    print(f"Duration: {event_data['duration']}")
                    
                    # Create the event
                    event = AudioEvent(
                        start_time=event_data['start_time'],
                        duration=event_data['duration'],
                        event_type=event_type,
                        parameters=event_data['parameters']
                    )
                    self.events.append(event)
                print(f"\nLoaded {len(self.events)} events from EVENTS_CONFIG")
            else:
                print("\nWarning: EVENTS_CONFIG not found in config file")
                    
        elif filename.endswith('.json'):
            # Load JSON config file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Debug print the config contents
            print("\nConfig file contents:")
            print(config_data)
                
            # Set gain from config
            if 'gain' in config_data:
                new_gain = float(config_data['gain'])
                print(f"\nSetting gain from JSON config: {new_gain:.3f}")
                self.gain = new_gain
            else:
                print("\nWarning: No gain specified in JSON config")
                if self.gain is None:
                    self.gain = 0.01  # Only set default if not specified anywhere
                
            # Load events
            self.events = []
            for event_data in config_data.get('events', []):
                self.events.append(AudioEvent(**event_data))
            print(f"\nLoaded {len(self.events)} events from JSON config")
        else:
            raise ValueError("Config file must be either .py or .json")
            
        print(f"\nFinal gain value: {self.gain:.3f}")
        print("\nLoaded events:")
        for i, event in enumerate(self.events):
            print(f"Event {i}: {event.event_type} at {event.start_time}s for {event.duration}s")
            print(f"Parameters: {event.parameters}\n")

# Example usage:
if __name__ == "__main__":
    import os
    
    # Initialize the audio control system
    audio_control = MuseumMAudioControl()
    
    # Get the absolute path to the config file
    config_path = SCRIPT_DIR / "audio_config.py"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        exit(1)
        
    print(f"\nUsing config file: {config_path}")
    
    # Load and verify the config file
    print("\nLoading and verifying configuration...")
    try:
        # Import the config file directly to verify its contents
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", str(config_path))
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        # Verify AUDIO_CONFIG
        if not hasattr(config, 'AUDIO_CONFIG'):
            print("Error: AUDIO_CONFIG not found in config file")
            exit(1)
            
        print("\nAUDIO_CONFIG found:")
        for key, value in config.AUDIO_CONFIG.items():
            print(f"  {key}: {value}")
            
        # Verify EVENTS_CONFIG
        if not hasattr(config, 'EVENTS_CONFIG'):
            print("Error: EVENTS_CONFIG not found in config file")
            exit(1)
            
        print("\nEVENTS_CONFIG found:")
        for i, event in enumerate(config.EVENTS_CONFIG):
            print(f"\nEvent {i}:")
            print(f"  Start time: {event['start_time']}")
            print(f"  Duration: {event['duration']}")
            print(f"  Function: {event['function']}")
            print(f"  Parameters: {event['parameters']}")
            
        # Load the events into audio control
        audio_control.load_events("audio_config.py")
        
        # Generate and save the audio
        print("\nGenerating audio...")
        output = audio_control.generate_audio()
        
        print("\nSaving audio...")
        audio_control.save_audio(output, "output.wav")
        audio_control.save_individual_speakers(output, "output")
        
        # Play the audio
        print("\nPlaying audio...")
        audio_control.play_audio(output)
        
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        exit(1)
