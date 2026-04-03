import fiftyone as fo
import fiftyone.zoo as foz

def main():
    print("Loading the 'quickstart-video' dataset to verify setup...")
    try:
        dataset = foz.load_zoo_dataset("quickstart-video")
        print("\n✅ Dataset loaded successfully!")
        print("\nLaunching the FiftyOne App. You can view it in your browser.")
        session = fo.launch_app(dataset)
        session.wait()  # Keep the app running
    except Exception as e:
        print(f"\n❌ Error loading dataset or launching app: {e}")

if __name__ == "__main__":
    main()
