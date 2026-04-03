import os
from dotenv import load_dotenv
import fiftyone as fo
from twelvelabs import TwelveLabs

def main():
    print("Loading environment variables...")
    load_dotenv()
    
    api_key = os.getenv("TWELVELABS_API_KEY")
    if not api_key or api_key == "your_actual_api_key_here":
        print("❌ Warning: TWELVELABS_API_KEY is not set or still set to the default template.")
        print("   Please edit the .env file and paste your actual API key.")
    else:
        try:
            client = TwelveLabs(api_key=api_key)
            print("✅ TwelveLabs client initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize TwelveLabs client: {e}")

    try:
        print("✅ FiftyOne version:", fo.__version__)
    except Exception as e:
        print(f"❌ Failed to load FiftyOne: {e}")

if __name__ == "__main__":
    main()
