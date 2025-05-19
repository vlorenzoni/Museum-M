import sounddevice as sd
import soundfile as sf
import os
import time
import numpy as np

# Configuration
DEVICE_NAME = "Digiface USB"  # or "Dante" depending on your setup
PLAY_DURATION = 13  # seconds before switching to next file
VOLUME_REDUCTION_DB = 20  # -20 dB reduction
GAIN_FACTOR = 10 ** (-VOLUME_REDUCTION_DB / 20)  # Convert -20 dB to linear gain

# Get the SDM files in correct MM1 to MM5 order
folder_path = "synthesized_scenarios"
sdm_files = sorted([
    f for f in os.listdir(folder_path) 
    if f.startswith('SDM_Sollazzo_Brou_church_MM') and f.endswith('.wav')
])

class AudioPlayer:
    def __init__(self, device_name, samplerate):
        self.device_name = device_name
        self.samplerate = samplerate
        self.current_buffer = None
        self.play_samples = int(PLAY_DURATION * samplerate)
        self.current_file_name = ""
        self.current_file_position = 0
        self.samples_played_in_segment = 0
        self.full_buffers = {}
        self.timeline_position = 0  # Track overall timeline position
        
    def load_all_files(self):
        """Preload all audio files"""
        for file_name in sdm_files:
            file_path = os.path.join(folder_path, file_name)
            data, _ = sf.read(file_path)
            # Apply -20 dB reduction when loading
            self.full_buffers[file_name] = data * GAIN_FACTOR
            
    def callback(self, outdata, frames, time_info, status):
        if status:
            print(status)
        
        if self.current_buffer is None:
            outdata.fill(0)
            return
        
        self.samples_played_in_segment += frames
        self.timeline_position += frames
        current_time = self.samples_played_in_segment / self.samplerate
        
        # Display progress
        file_position = (self.current_file_position / self.samplerate)
        timeline_position = (self.timeline_position / self.samplerate)
        print(f"\rPlaying: {self.current_file_name} - File Position: {file_position:.1f}s - Segment time: {current_time:.1f}s / {PLAY_DURATION}s - Timeline: {timeline_position:.1f}s", end="")
        
        # Check if it's time to switch files
        if self.samples_played_in_segment >= self.play_samples:
            self.switch_to_next_file()
        
        # Fill output buffer
        outdata[:] = self.current_buffer[:frames]
        
        # Update positions and buffer
        self.current_file_position += frames
        if len(self.current_buffer) <= frames:
            # If we reach the end of the file, loop back to beginning
            self.current_file_position = 0
            self.current_buffer = self.full_buffers[self.current_file_name].copy()
        else:
            # Remove played samples
            self.current_buffer = self.current_buffer[frames:]
        
    def switch_to_next_file(self):
        """Switch to next file maintaining current position"""
        current_index = sdm_files.index(self.current_file_name)
        next_index = (current_index + 1) % len(sdm_files)
        next_file = sdm_files[next_index]
        
        print(f"\nSwitching to: {next_file}")
        
        # Calculate equivalent position in the new file
        next_data = self.full_buffers[next_file]
        self.current_file_position = self.current_file_position % len(next_data)
        
        # Set up the new buffer from the current position
        self.current_buffer = np.roll(next_data, -self.current_file_position, axis=0)
        self.current_file_name = next_file
        self.samples_played_in_segment = 0

def main():
    # Print configuration
    print("\nPlayback Configuration:")
    print(f"- Volume reduction: -{VOLUME_REDUCTION_DB} dB (fixed)")
    print(f"- Switching interval: {PLAY_DURATION} seconds")
    print(f"- Using device: {DEVICE_NAME}")
    print(f"- Files will play in sequence: {' → '.join(sdm_files)}")
    
    # Load first file and initialize player
    first_file = sdm_files[0]
    file_path = os.path.join(folder_path, first_file)
    data, samplerate = sf.read(file_path)
    
    # Initialize audio player
    player = AudioPlayer(DEVICE_NAME, samplerate)
    player.load_all_files()  # Preload all files with -20 dB reduction
    player.current_buffer = player.full_buffers[first_file].copy()
    player.current_file_name = first_file
    
    try:
        with sd.OutputStream(
            device=DEVICE_NAME,
            channels=8,
            callback=player.callback,
            samplerate=samplerate,
            blocksize=int(samplerate * 0.1)  # 100ms blocks
        ):
            print("\nStarting playback (Press Ctrl+C to stop)")
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nPlayback stopped by user")
    except Exception as e:
        print(f"\nError during playback: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
