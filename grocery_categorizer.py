import ollama
import os

model = 'gemma3:4b'

# Paths
input_path = "./data/grocery_list.txt"
output_path = "./categorized_grocery_list.txt"

# Check if the input file exists
if not os.path.exists(input_path):
    print(f"Input file {input_path} does not exist.")
    exit(1)

# Read the input file
with open(input_path, 'r') as file:
    grocery_list = file.read().strip()


# Prepare the prompt for the model
prompt = f"""
You are an assistant that categorizes and sorts grocery items.

Here is a list of grocery items:

{grocery_list}

Please:

1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.

"""

# Call the Ollama model to generate the categorized grocery list
try:
    response = ollama.generate(model=model, prompt=prompt)
    generated_text = response.get('response', '')
    print("=== Categorized Grocery List === \n")
    print(generated_text)

    # Write the categorized grocery list to the output file
    with open(output_path, 'w') as output_file:
        output_file.write(generated_text.strip())

    print(f"\nCategorized grocery list has been saved to {output_path}.")

except Exception as e:
    print(
        f"An error occurred while generating the categorized grocery list: {e}")
