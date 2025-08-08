import ollama
from pathlib import Path
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Functions to read grocery list, generate prompt, and write output


def read_grocery_list(input_path: Path) -> Optional[str]:
    """Read grocery list from file."""

    # Check if the input path exists
    if not input_path.exists():
        logging.error(f"Input file {input_path} does not exist.")
        return None

    try:
        grocery_list = input_path.read_text().strip()

        # Check if the grocery list is empty
        if not grocery_list:
            logging.warning("Grocery list is empty.")

        return grocery_list

    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return None


def generate_prompt(grocery_list: str) -> str:
    """Prepare prompt for categorizing grocery items."""
    return f"""
You are an assistant that categorizes and sorts grocery items.

Here is a list of grocery items:

{grocery_list}

Please:

1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.
"""


def write_output(output_path: Path, text: str) -> None:
    """Write categorized grocery list to output file."""
    try:
        output_path.write_text(text.strip())
        logging.info(
            f"Categorized grocery list has been saved to {output_path}.")
    except Exception as e:
        logging.error(f"Error writing to output file: {e}")


# Function to categorize groceries using Ollama model
def categorize_groceries(model: str, input_path: Path, output_path: Path) -> None:
    """Main function to categorize groceries using Ollama model."""
    grocery_list = read_grocery_list(input_path)
    if not grocery_list:
        logging.error("No grocery list to process.")
        return
    prompt = generate_prompt(grocery_list)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        generated_text = response.get('response', '')
        logging.info("=== Categorized Grocery List ===\n")
        print(generated_text)
        write_output(output_path, generated_text)
    except Exception as e:
        logging.error(
            f"An error occurred while generating the categorized grocery list: {e}")


if __name__ == "__main__":
    model = 'gemma3:4b'
    input_path = Path("./data/grocery_list.txt")
    output_path = Path("./categorized_grocery_list.txt")
    categorize_groceries(model, input_path, output_path)
