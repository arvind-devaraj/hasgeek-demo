import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Initialize the modern GenAI Client
client = genai.Client()

# =====================================================================
# 1. DEFINE SCHEMAS & TOOLS VIA PYDANTIC
# =====================================================================
# We use Pydantic to strictly define the structured payload the model must provide.
class SearchProductSchema(BaseModel):
    query: str = Field(description="The search term or product name keyword.")
    max_results: int = Field(default=3, description="Maximum number of products to return.")

class CheckInventorySchema(BaseModel):
    product_id: str = Field(description="The unique alphanumeric ID of the product (e.g., 'PROD-123').")

# The actual Python logic executed locally when the model requests it
def search_products(query: str, max_results: int = 3) -> dict:
    print(f"🛠️ [Executing Tool] Searching for '{query}' (Max: {max_results})...")
    return {"results": [{"id": "PROD-902", "name": "Wireless Ergonomic Mouse", "price": 49.99}]}

def check_inventory(product_id: str) -> dict:
    print(f"🛠️ [Executing Tool] Checking stock levels for '{product_id}'...")
    if product_id == "PROD-902":
        return {"product_id": "PROD-902", "status": "In Stock", "quantity": 14}
    return {"product_id": product_id, "status": "Out of Stock", "quantity": 0}

# Execution mapper matching native naming conventions to the local functions
tools_map = {
    "search_products": search_products,
    "check_inventory": check_inventory
}

# =====================================================================
# 2. THE MULTI-TURN LOOP USING NATIVE FUNCTION CALLS
# =====================================================================
def run_native_agent(user_prompt: str):
    # Initialize the continuous trajectory tracking state
    state = [
        {"role": "user", "parts": [user_prompt]}
    ]
    
    print(f"🛒 Customer Query: {user_prompt}\n")
    
    for iteration in range(5):
        print(f"--- Native Execution Loop {iteration + 1} ---")
        
        # Turn our ongoing tracking state list into standard Gemini Content objects
        contents = [
            types.Content(role=s["role"], parts=s["parts"]) for s in state
        ]
        
        # We supply the functions directly to the API. 
        # The Gemini platform reflects over the Python functions and any typing hints natively.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[search_products, check_inventory],
                temperature=0.0
            )
        )
        
        # IMMUTABLE RULE: Immediately append the full model response objects to the timeline tape
        state.append({"role": "model", "parts": response.candidates[0].content.parts})
        
        # Check if the model has requested structured schema actions natively
        if response.function_calls:
            for call in response.function_calls:
                # NATIVE VALIDATION BENEFIT: 'call.name' and 'call.args' are already parsed structured fields.
                # No manual Regex parsing, no custom split("Action:") strings.
                print(f"🤖 Native Tool Request Received: '{call.name}'")
                print(f"📦 Validated Payload Extracted: {call.args}")
                
                # Execute the dynamic tool function using the pre-parsed keyword dictionary
                tool_func = tools_map[call.name]
                result = tool_func(**call.args)
                
                print(f"📥 Schema Output Received: {result}\n")
                
                # Feed the validated structural output directly back into the memory sequence
                state.append({
                    "role": "user",
                    "parts": [
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": result}
                        )
                    ]
                })
            # Step into the next processing loop iteration
            continue
            
        else:
            print(f"🤖 Final Customer Response:\n{response.text}")
            break

# Execute the Native agent loop
run_native_agent("Find the wireless ergonomic mouse and see if it's currently available in stock.")