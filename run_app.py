"""
Script to run the Streamlit app
"""

import os
import subprocess

def run_streamlit_app():
    """Run the Streamlit app"""
    print("Starting RAG Chatbot Streamlit app...")
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the app.py file
    app_path = os.path.join(current_dir, "app.py")
    
    # Check if the app.py file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find app.py at {app_path}")
        return
    
    try:
        # Run the Streamlit app
        subprocess.run(["streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
    except FileNotFoundError:
        print("Error: Streamlit not found. Make sure it's installed in your environment.")
        print("You can install it with: pip install streamlit")

if __name__ == "__main__":
    run_streamlit_app()