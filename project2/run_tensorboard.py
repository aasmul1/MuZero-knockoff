"""
Helper script to launch TensorBoard for visualizing MuZero training.

Run this script with:
python -m project2.run_tensorboard

Then open your browser at http://localhost:6006
"""

import os
import subprocess
import sys

def run_tensorboard(logdir='tensorboard_logs', port=6006):
    """
    Run TensorBoard to visualize training logs.
    
    Args:
        logdir: Directory containing TensorBoard log files
        port: Port to run TensorBoard on
    """
    # Make sure the log directory exists
    os.makedirs(logdir, exist_ok=True)
    
    print(f"Starting TensorBoard with logs from {logdir}...")
    print(f"Open your browser at http://localhost:{port}")
    
    try:
        subprocess.run([
            sys.executable, "-m", "tensorboard.main", 
            "--logdir", logdir, 
            "--port", str(port),
            "--bind_all"  # Allow access from other devices on the network
        ])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except Exception as e:
        print(f"Error starting TensorBoard: {e}")
        print("\nMake sure TensorBoard is installed with:")
        print("pip install tensorboard")

if __name__ == "__main__":
    # You can change these values if needed
    log_directory = 'tensorboard_logs'
    port_number = 6006
    
    # Get command line arguments if provided
    if len(sys.argv) > 1:
        log_directory = sys.argv[1]
    if len(sys.argv) > 2:
        port_number = int(sys.argv[2])
        
    run_tensorboard(log_directory, port_number) 