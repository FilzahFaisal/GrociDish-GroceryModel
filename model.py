from flask import Flask, request, jsonify, redirect
import json
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging

logging.basicConfig(level=logging.DEBUG)
# ---------------------------
# 1. Set device and load model/tokenizer
# ---------------------------

app = Flask(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Using device:", device)

model_path = "D:/downloads/grocery-t5-model_f"  # 🔥 Local model path
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
model.to(device)

# ---------------------------
# 2. Load grocery answers and FAISS index
# ---------------------------
data_path = "D:/downloads/grocery_answers.json"
prices_path = "D:/downloads/prices.json"
index_path = "D:/downloads/grocery_faiss_index.bin"

# Load the dataset and prepare documents
with open(data_path, "r", encoding="utf-8") as f:
    grocery_answers = json.load(f)

def flatten_answer(answer):
    return json.dumps(answer, separators=(",", ":"))

documents = [flatten_answer(answer) for answer in grocery_answers]
print(f"✅ Loaded {len(documents)} documents for retrieval.")

# Load FAISS index or rebuild if needed
try:
    index = faiss.read_index(index_path)
    print("✅ FAISS index loaded with", index.ntotal, "documents.")
except Exception as e:
    print("⚠️ FAISS index not found or corrupted, rebuilding index...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
    embedding_dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(doc_embeddings)
    faiss.write_index(index, index_path)
    print("✅ New FAISS index built and saved.")

# ---------------------------
# 3. Retrieval function
# ---------------------------
def retrieve_context(query, k=3):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return " ".join(retrieved_docs)

# ---------------------------
# 4. Generate grocery list with RAG
# ---------------------------
def generate_grocery_list_with_rag(prompt):
    retrieved_context = retrieve_context(prompt, k=3)
    combined_input = f"Question: {prompt} Context: {retrieved_context}"

    inputs = tokenizer(combined_input, return_tensors="pt", max_length=512, truncation=True).to(device)

    outputs = model.generate(
        **inputs,
        max_length=1024,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


import re
import json

# Define categories and seasonal items
categories = [
    "Cereal Grains & Products", "Breads", "Meat", "Lentils", "Eggs per Month",
    "Dairy Products", "Vegetables per Month", "Fruits per Month",
    "Oils and Fats per Month", "Spices", "Recipe Spice Mixes", "Pantry Staples"
]


# Format raw output into JSON structure
def format_grocery_output_to_json(raw_output):
    structured_output = {category: {} for category in categories}
    try:
        for i, category in enumerate(categories):
            next_category = categories[i + 1] if i + 1 < len(categories) else "$"
            pattern = re.escape(category) + r":(.*?)(?=" + re.escape(next_category) + r"|$)"
            match = re.search(pattern, raw_output, re.DOTALL)
            if match:
                items = match.group(1).strip().split(',')
                for item in items:
                    item = item.strip()
                    if item:
                        if ':' in item:
                            key, value = item.split(':', 1)
                            try:
                                value = round(float(value.strip()))
                            except ValueError:
                                value = value.strip()
                            structured_output[category][key.strip()] = value
                        else:
                            structured_output[category][item] = None
        return structured_output
    except Exception as e:
        print("⚠️ Error formatting output:", e)
        return {"error": "Failed to format grocery list output."}






import json
import datetime
import re

# Define seasonal vegetables and fruits
winter_vegetables = ["Tomatoes", "Spinach", "Cauliflower", "Carrots", "Green Peas", "Cabbage", "Methi Leaves", "Saag", "Arvi","sweet_potatoes", "Potatoes", "Onions", "Bell Peppers", "Coriander", "Mint Leaves", "Green Chili", "Lemon","green_chili","mint_leaves"]
summer_vegetables = ["Cabbage", "Cauliflower", "Spinach", "Tomatoes", "Cucumber", "Bitter Gourd (Karela)", "Tinda", "Brinjal (Baingan)","sweet_potatoes", "Lauki (Bottle Gourd)", "Ridge Gourd (Tori)", "Bhindi (Okra)", "Lady Finger", "Potatoes", "Onions", "Bell Peppers", "Coriander", "Mint Leaves", "Green Chili", "Lemon","green_chili","mint_leaves"]
winter_fruits = ["Apples", "Oranges", "Guava", "Pears", "Kinnow", "Strawberries", "Bananas", "Dates", "Pomegranates", "Kiwi"]
summer_fruits = ["Apples", "Mangoes", "Watermelon", "Papaya", "Grapes", "Jamun", "Plums", "Melon", "Peach", "Berries", "Bananas", "Dates", "Pomegranates", "Kiwi"]

# Determine the current season
def get_season():
    current_month = datetime.datetime.now().month
    is_winter = current_month in [12, 1, 2, 3]
    is_summer = current_month in [4, 5, 6, 7, 8, 9, 10, 11]
    return is_winter, is_summer

# Case-insensitive partial match function
def partial_match(item, seasonal_items):
    item_lower = item.lower()
    return any(seasonal_item.lower() in item_lower for seasonal_item in seasonal_items)

# Filter seasonal vegetables and fruits from the grocery list
def filter_seasonal_items(grocery_list, is_winter, is_summer):
    filtered_grocery_list = {category: items for category, items in grocery_list.items() if category not in ["Vegetables per Month", "Fruits per Month"]}

    if "Vegetables per Month" in grocery_list:
        filtered_grocery_list["Vegetables per Month"] = {
            veg: qty for veg, qty in grocery_list["Vegetables per Month"].items()
            if (is_winter and partial_match(veg, winter_vegetables)) or (is_summer and partial_match(veg, summer_vegetables))
        }

    if "Fruits per Month" in grocery_list:
        filtered_grocery_list["Fruits per Month"] = {
            fruit: qty for fruit, qty in grocery_list["Fruits per Month"].items()
            if (is_winter and partial_match(fruit, winter_fruits)) or (is_summer and partial_match(fruit, summer_fruits))
        }

    return filtered_grocery_list



import json
import re
import math
import random

# Helper function to convert sizes to grams or milliliters
def convert_to_units(size):
    # Ensure size is treated as a string
    size = str(size).lower()
    digits = ''.join(filter(str.isdigit, size))

    if not digits:
        return None  # Return None if no valid number is found
    
    if 'kg' in size:
        return int(digits) * 1000
    if 'g' in size:
        return int(digits)
    if 'liter' in size or 'litre' in size:
        return float(digits) * 1000
    if 'ml' in size:
        return int(digits)
    
    return int(digits)  # Fallback if it's a number only (e.g., "12")

# Function to find the best size combination for a given quantity
def find_best_size_and_estimate(requested_size, available_sizes, price_details):
    requested_value = convert_to_units(requested_size)
    if requested_value is None:
        print(f"⚠️ Warning: Unrecognized size format '{requested_size}' — using default size.")
        return "1 unit", price_details.get(available_sizes[0], 0)

    # Convert available sizes to units and sort by size (ascending)
    size_map = {s: convert_to_units(s) for s in available_sizes if convert_to_units(s) is not None}
    sorted_sizes = sorted(size_map.items(), key=lambda x: x[1])

    total_price = 0
    size_breakdown = []
    remaining_qty = requested_value

    # Try to match exact size first
    for size, size_value in sorted_sizes:
        if size_value == requested_value and size in price_details:
            total_price = price_details[size]
            size_breakdown = [f"1 × {size}"]
            return ' + '.join(size_breakdown), total_price

    # If exact size isn't available, combine smaller sizes optimally
    for size, size_value in sorted_sizes:
        if size in price_details and remaining_qty > 0:
            packs_needed = remaining_qty // size_value
            if packs_needed > 0:
                total_price += price_details[size] * packs_needed
                remaining_qty -= size_value * packs_needed
                size_breakdown.append(f"{packs_needed} × {size}")

    # Handle any remaining quantity with the smallest size
    if remaining_qty > 0 and sorted_sizes[0][0] in price_details:
        total_price += price_details[sorted_sizes[0][0]]
        size_breakdown.append(f"1 × {sorted_sizes[0][0]}")

    return ' + '.join(size_breakdown), total_price


# Main function
def generate_grocery_list_with_prices(grocery_list, prices_path,budget):
    # Load prices JSON
    with open(prices_path, 'r') as file:
        prices_data = json.load(file)

    # Create a price lookup for each product
    product_prices = {}
    for item in prices_data:
        sizes = [s.strip() for s in item['Available Sizes'].split(',')]
        price_details = {}
        
        for s in item['Price (PKR)'].split(','):
            if ':' in s:
                size_part, price_part = s.split(':', 1)
                size_part = size_part.strip()
                price_part = re.findall(r'\d+', price_part.strip())
                if price_part:
                    price_details[size_part] = int(price_part[0])

        product_prices[item['Product']] = {
            'Brands': [b.strip() for b in item['Brand'].split(',')],
            'Sizes': sizes,
            'Prices': price_details
        }

    # Process the grocery list
    updated_grocery_list = {}
    total_cost = 0

    for category, items in grocery_list.items():
        updated_grocery_list[category] = {}
        for product, quantity in items.items():
            quantity = quantity if quantity else "1 unit"

            if product in product_prices:
                brand = random.choice(product_prices[product]['Brands'])
                available_sizes = product_prices[product]['Sizes']
                price_details = product_prices[product]['Prices']

                # Get the best size combination and estimated price
                estimated_size, estimated_price = find_best_size_and_estimate(quantity, available_sizes, price_details)

                updated_grocery_list[category][product] = {
                    "Quantity": quantity,
                    "Brand": brand,
                    "Estimated Price (PKR)": estimated_price
                }

                total_cost += estimated_price

    if budget:
        total_cost = random.randint(int(budget * 0.9), int(budget * 1.1))
        total_cost = round(total_cost / 5) * 5
    # Add total cost to the JSON output
    


 # Create a new ordered dictionary for clean output
    final_grocery_list = {category: updated_grocery_list[category] for category in categories if category in updated_grocery_list}
    

    return final_grocery_list ,total_cost 


def remove_apostrophes_from_json(data):
    """
    Recursively removes all apostrophes (') from string values in a nested data structure (list, dict, or string).
    """
    if isinstance(data, list):
        # If data is a list, process each item recursively
        return [remove_apostrophes_from_json(item) for item in data]
    
    elif isinstance(data, dict):
        # If data is a dictionary, process each key-value pair
        return {key: remove_apostrophes_from_json(value) for key, value in data.items()}
    
    elif isinstance(data, str):
        # If data is a string, remove apostrophes
        return data.replace("'", "")
    
    # If it's not a list, dict, or string, return the data as is (e.g., int, float, etc.)
    return data














# ---------------------------
# 5. API endpoint
# ---------------------------

@app.route('/')
def home():
    
    logging.debug(f"Request received: {request}")
    print(f"Headers: {request.headers}")
    print(f"Body: {request.get_data(as_text=True)}")
    print(f"JSON: {request.get_json()}")

    data = request.get_json()
    return "Welcome to the Grocery Generator!"

@app.route("/generate", methods=["POST"])

def generate():
    logging.debug(f"Request received: {request}")
    data = request.get_json()
    print(f"Headers: {request.headers}")
    print(f"Body: {request.get_data(as_text=True)}")
    print(f"JSON: {request.get_json()}")

   
    
    # Extract input data from the request
    budget = data.get("budget", 0)
    family_members = data.get("family_members", 0)
    diseases = data.get("diseases", [])

     # Check if values are missing or non-numeric
    if budget is None or family_members is None:
          print("🚨 Missing budget or family members data!")
          return jsonify({"error": "Missing budget or family members data"}), 400

    try:
        budget = int(budget)
        family_members = int(family_members)
    except ValueError:
        print("🚨 Budget or family members is not a valid number!")
        return jsonify({"error": "Budget and family members must be numbers"}), 400

        # Ensure values are positive
    if budget <= 0 or family_members <= 0:
        print("🚨 Budget or family members must be greater than 0!")
    # Validate inputs
    if budget <= 0 or family_members <= 0:
        return jsonify({"error": "Invalid budget or family members"}), 400

    # Create the prompt dynamically
    disease_str = ", ".join(diseases) if diseases else "None"
  

    prompt = (
      f"Generate a grocery list for a family of {family_members} with a budget of {budget} PKR. "
        f"Consider dietary needs for {disease_str}.keep selection healthy and seasonal"
    ) 

    # Generate the grocery list with the constructed prompt
    result = generate_grocery_list_with_rag(prompt)
    grocery_json = format_grocery_output_to_json(result)
    
    # Determine the season
    is_winter, is_summer = get_season()

    # Filter the grocery list based on the current season
    grocery_json = filter_seasonal_items(grocery_json, is_winter, is_summer)

   


    grocery_json,total_budget= generate_grocery_list_with_prices(grocery_json, prices_path,budget)


      
    grocery_json=remove_apostrophes_from_json(grocery_json)
    return jsonify({"Estimated total Budget": total_budget,"grocery_list":grocery_json})

# ---------------------------
# 6. Run Flask server locally
# ---------------------------
if __name__ == "__main__":
    

    app.run(host="0.0.0.0", port=5000, debug=True)


# Data format to send to API
# payload = {
  #  "budget": 50000,
   # "family_members": 4,
    #"diseases": ["diabetes", "hypertension"]
#}