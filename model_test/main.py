import subprocess
import os

def run_script(script_path):
    """Run a Python script"""
    command = f"python {script_path}"
    subprocess.run(command, shell=True, check=True)

def main():
    # Step 1: Run text cleaning script
    print("Running text cleaning script...")
    run_script('text_cleaning.py')

    # Step 2: Train sentiment analysis model
    print("Running model training script...")
    run_script('model_training.py')

    # Step 3: Run FAST API server
    print("Starting FAST API server...")
    os.system('uvicorn app:app --reload')

if __name__ == "__main__":
    main()
