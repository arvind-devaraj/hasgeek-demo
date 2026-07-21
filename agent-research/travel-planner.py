from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env

# Mock Databases
FLIGHT_DB = {
    "Mumbai-Delhi": [
        {"flight": "6E-2012", "price": 4500, "time": "06:00 AM"},
        {"flight": "AI-802", "price": 5200, "time": "09:00 AM"}
    ],
    "Bangalore-Goa": [
        {"flight": "QP-1102", "price": 3200, "time": "11:00 AM"}
    ]
}

HOTEL_DB = {
    "Delhi": {"luxury": "The Leela Palace", "mid": "FabHotel Prime", "budget": "Zostel Delhi"},
    "Goa": {"luxury": "Taj Exotica", "mid": "Lemon Tree", "budget": "Old Quarter Hostel"}
}


@function_tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for flights between Indian cities.

    Args:
        origin: Departure city (e.g., 'Mumbai').
        destination: Arrival city (e.g., 'Delhi').
        date: Date of travel in YYYY-MM-DD format.
    """
    key = f"{origin}-{destination}"
    return str(FLIGHT_DB.get(key, "No flights found for this route."))


@function_tool
def get_hotel_recommendation(city: str, budget_level: str) -> str:
    """Recommend a hotel based on city and budget.

    Args:
        city: The city to stay in.
        budget_level: One of 'luxury', 'mid', or 'budget'.
    """
    return HOTEL_DB.get(city, {}).get(budget_level, "No specific recommendation found.")


agent = Agent(
    name="Travel Coordinator",
    instructions="Use the tools provided sequentially or in parallel to compile travel recommendations.",
    tools=[search_flights, get_hotel_recommendation],
    model="gpt-4o-mini",
)

if __name__ == "__main__":
    print("Starting Multi-Hop Tool Execution Loop...\n")

    # The Runner handles the LLM -> tool call -> result -> LLM loop internally.
    result = Runner.run_sync(
        agent,
        "I'm doing a two-leg trip: Mumbai to Delhi on May 10th, then Bangalore to Goa on May 15th. "
        "My total budget is ₹10,000 total. Find the cheapest flight for each leg and tell me how "
        "much is left over. If more than ₹3,000 is left, recommend a mid-range hotel in Delhi; "
        "otherwise recommend a budget hotel there."
    )

    print("\n--- Final Travel Recommendation ---")
    print(result.final_output)
