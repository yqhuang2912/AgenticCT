
import argparse
import unsloth
from src.workflow import run_agent_workflow
from src.utils.load_save_image import load_npy_to_tensor, visualize_tensor
import os

def parse_args():
    parser = argparse.ArgumentParser(description="CTRestoreAgent main script")
    parser.add_argument("--image_url", type=str, default="/data/hyq/codes/ct_restore_agent/la.png", help="Image file path to include")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # user_query = input("Please enter your query: ")
    user_query = "Please evaluate all degradation types and their severity levels in this CT image."
    
    # Visualize input image
    if args.image_url and os.path.exists(args.image_url):
        try:
            print(f"Visualizing input image: {args.image_url}")
            input_tensor = load_npy_to_tensor(args.image_url)
            input_filename = os.path.basename(args.image_url)
            vis_filename = f"/data/hyq/codes/AgenticCT/intermediates/input_{input_filename}.png"
            visualize_tensor(input_tensor, vis_filename)
            print(f"Input image visualized as {vis_filename}")
        except Exception as e:
            print(f"Failed to visualize input image: {e}")

    if not user_query.strip():
        print("No query provided. Exiting.")
        return
    
    try:
        result = run_agent_workflow(
            user_query=user_query,
            image_url=args.image_url,
            debug=args.debug
        )
        print("Workflow completed successfully.")
        print("Result:", result)
    except Exception as e:
        print(f"An error occurred: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return

if __name__ == "__main__":
    main()