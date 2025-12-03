import logging

from src.graph import build_graph
from src.config import TEAM_MEMBERS
import subprocess
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s %(levelname)s - %(message)s",
)

def enable_debug_logging():
    """
    Enable debug logging for the entire module.
    """
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)


# create the workflow graph
graph = build_graph()
    

def run_agent_workflow(user_query, image_url=None, debug=False, max_iterations=5):
    """
    Run the agent workflow with the provided user query and configurations.
    
    Args:
        user_query (str): The user's query to process.
        image_url (str, optional): URL of the image file to include in the workflow.
        debug (bool, optional): Enable debug logging if True.
        max_iterations (int, optional): Maximum number of processing iterations. Default is 5.
    
    Returns:
        The final state after the workflow completes
    """
    if debug:
        enable_debug_logging()

    logger.info(f"Starting workflow with user input: {user_query}")
    logger.info(f"Max processing iterations: {max_iterations}")
    
    # Simulate processing
    result = graph.invoke(
        {
            # Constants
            "TEAM_MEMBERS": TEAM_MEMBERS,
            "image_url": image_url,
            # Runtime Variables
            "messages": [{"role": "user", "content": user_query}],
            "current_step": 0,  # Initialize current_step
            "processing_count": 0,
            "max_processing_iterations": max_iterations,
        }
    )
    
    logger.debug(f"Final workflow state: {result}")
    logger.info("Workflow completed successfully")
    return result

if __name__ == "__main__":
    # Generate the Mermaid diagram
    mermaid_diagram = graph.get_graph().draw_mermaid()
    
    # Save the diagram to local files
    mmd_file = "workflow_diagram.mmd"
    png_file = "workflow_diagram.png"
    
    # Save the text version
    with open(mmd_file, "w") as f:
        f.write(mermaid_diagram)
    
    # Save as PNG using mermaid-cli if available
    try:
        
        # Create a temporary HTML file with the mermaid diagram
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true}});</script>
        </head>
        <body>
            <div class="mermaid">
            {mermaid_diagram}
            </div>
        </body>
        </html>
        """
        temp_html = "temp_diagram.html"
        with open(temp_html, "w") as f:
            f.write(html_content)
        
        # Try using mmdc (Mermaid CLI) if installed
        subprocess.run(["mmdc", "-i", mmd_file, "-o", png_file], check=True)
        print(f"PNG diagram saved to {png_file}")
        
        # Clean up temp file
        os.remove(temp_html)
    except Exception as e:
        print(f"Could not generate PNG: {e}")
        print("To generate PNG, install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
    
    print(f"Mermaid diagram saved to {mmd_file}")