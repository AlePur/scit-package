import json
import os
import base64
import argparse
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

def extract_outputs_from_notebook(notebook_path, cell_indices=None):
    """
    Extract outputs from specified cells in a Jupyter notebook.
    If cell_indices is None, extract from all cells.

    Returns a list of tuples (output_type, content).
    """
    # Read the notebook file
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    outputs = []

    # Iterate through cells
    for i, cell in enumerate(notebook.get('cells', [])):
        # Skip cells not in the specified indices if indices are provided
        if cell_indices is not None and i not in cell_indices:
            continue

        # Only include output cells
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                # Text output
                if 'text' in output:
                    text_content = ''.join(output['text'])
                    outputs.append(('text', text_content))

                # Stream output (stdout/stderr)
                if 'name' in output and 'text' in output:
                    stream_name = output['name']  # stdout or stderr
                    text_content = ''.join(output['text'])
                    outputs.append(('stream', f"{stream_name}: {text_content}"))

                # Image output
                if 'data' in output:
                    data = output['data']
                    # Handle PNG images
                    if 'image/png' in data:
                        img_data = data['image/png']
                        outputs.append(('image', img_data))

                    # Handle text/plain outputs
                    if 'text/plain' in data:
                        text_content = ''.join(data['text/plain'])
                        outputs.append(('text', text_content))

                    # Handle HTML outputs
                    if 'text/html' in data:
                        html_content = ''.join(data['text/html'])
                        outputs.append(('html', html_content))

    return outputs

def create_pdf_from_outputs(outputs, output_pdf_path):
    """
    Create a PDF from a list of outputs extracted from a Jupyter notebook using matplotlib.
    """
    with PdfPages(output_pdf_path) as pdf:
        for output_type, content in outputs:
            if output_type in ['text', 'stream', 'html']:
                # Create a figure for text content
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')  # Hide axes

                # Process the text content
                if output_type == 'html':
                    # Strip HTML tags for simple display
                    display_text = f"HTML Content:\n{content[:1000]}"
                    if len(content) > 1000:
                        display_text += "... (truncated)"
                else:
                    display_text = content

                # Add text to the figure
                ax.text(0.05, 0.95, display_text,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        wrap=True)

                # Save to PDF
                pdf.savefig(fig)
                plt.close(fig)

            elif output_type == 'image':
                try:
                    # Decode the base64 image data
                    img_data = base64.b64decode(content)

                    # Open the image using PIL
                    img = Image.open(BytesIO(img_data))

                    # Calculate the aspect ratio
                    width, height = img.size
                    aspect_ratio = width / height

                    # Create a new figure with appropriate dimensions
                    if aspect_ratio > 1:
                        fig, ax = plt.subplots(figsize=(8.5, 8.5/aspect_ratio))
                    else:
                        fig, ax = plt.subplots(figsize=(8.5*aspect_ratio, 8.5))

                    # Display the image
                    ax.imshow(np.array(img))
                    ax.axis('off')  # Hide axes

                    # Save to PDF
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    # Create an error message figure if image processing fails
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')
                    ax.text(0.5, 0.5, f"Error displaying image: {e}",
                            ha='center', va='center', fontsize=12)
                    pdf.savefig(fig)
                    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Extract outputs from Jupyter notebook cells and convert to PDF')
    parser.add_argument('notebook_path', help='Path to the Jupyter notebook file')
    parser.add_argument('--output', '-o', default='notebook_output.pdf', help='Output PDF file path')
    parser.add_argument('--cells', '-c', type=int, nargs='+', help='Specific cell indices to extract (0-indexed)')

    args = parser.parse_args()

    # Extract outputs from the notebook
    outputs = extract_outputs_from_notebook(args.notebook_path, args.cells)

    if not outputs:
        print("No outputs found in the specified cells.")
        return

    # Create the PDF
    create_pdf_from_outputs(outputs, args.output)
    print(f"PDF created successfully at {args.output}")

if __name__ == "__main__":
    main()
