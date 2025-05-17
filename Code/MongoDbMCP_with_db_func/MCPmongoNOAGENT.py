from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient

# Initialize FastMCP (for tool registration)
mcp = FastMCP("IceCreamRevenue")

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mcppracdb"]
collection = db["newcol"]

# Tool to query revenue by temperature
@mcp.tool()
def query_revenue_by_temperature(Temperature: float) -> dict:
    """
    Queries the transactions collection for the first document matching the given temperature.
    """
    return collection.find_one({"Temperature": Temperature}) or {"message": f"No document found for temperature {Temperature}."}

# Tool to update revenue for a specific temperature
@mcp.tool()
def update_revenue(Temperature: float, new_revenue: float) -> dict:
    """
    Updates the revenue for a document with the given temperature.
    """
    result = collection.update_one(
        {"Temperature": Temperature},
        {"$set": {"Revenue": new_revenue}}
    )
    return {"modified_count": result.modified_count}

# Tool to delete a document by temperature
@mcp.tool()
def delete_revenue(Temperature: float) -> dict:
    """
    Deletes a document with the given temperature.
    """
    result = collection.delete_one({"Temperature": Temperature})
    return {"deleted_count": result.deleted_count}

# Tool to create an index on temperature
@mcp.tool()
def create_temperature_index() -> dict:
    """
    Creates an ascending index on the temperature field.
    """
    index_name = collection.create_index([("Temperature", 1)])
    return {"index_created": index_name}

if __name__ == "__main__":
    
    result = query_revenue_by_temperature(20.6)
    print("temperature 20.6:", result)

