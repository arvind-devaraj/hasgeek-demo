from smolagents import CodeAgent, LiteLLMModel, tool
import datetime

# 1. Setup the Local Model (Ollama)
# Ensure you have run: ollama pull gemma3:4b
model = LiteLLMModel(
    model_id="ollama/gemma3:4b", 
    api_base="http://localhost:11434"
)

# 2. Define the "Database" Tool
@tool
def get_order_status(order_id: str) -> str:
    """
    Retrieves the real-time status and estimated delivery for a customer order.
    
    Args:
        order_id: The unique alphanumeric order tracking ID (e.g., 'ORD-123').
    """
    # Simulated Database Lookup
    mock_db = {
        "ORD-101": {"status": "In Transit", "location": "Bengaluru Hub", "eta": "2 Hours"},
        "ORD-202": {"status": "Delivered", "location": "Mumbai", "eta": "N/A"},
        "ORD-303": {"status": "Processing", "location": "Warehouse", "eta": "2 Days"}
    }
    
    order = mock_db.get(order_id.upper())
    if order:
        return f"Order {order_id} is currently '{order['status']}' at {order['location']}. Estimated arrival: {order['eta']}."
    return f"Order ID {order_id} not found. Please verify the ID and try again."

# 3. Initialize the Agent
agent = CodeAgent(
    tools=[get_order_status],
    model=model,
    add_base_tools=False # We only want it to use our specific support tool
)

# 4. Run the Customer Support Query
print("--- Customer Support Agent ---")
customer_query = "Hey, I've been waiting for my package ORD-101. Can you tell me where it is and when it will get here?"

agent.run(customer_query)