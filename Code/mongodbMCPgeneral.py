from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient

mcp = FastMCP("MongoDB_MCP")

client = MongoClient("mongodb://localhost:27017/")
db = client["mcppracdb"]
collection = db["newcol"]

@mcp.tool()
def query_by_field(field_name: str, field_value: float) -> dict:
    return collection.find_one({field_name: field_value}) or {"message": "No matching document found."}

@mcp.tool()
def update_field(match_field: str, match_value: float, update_field: str, update_value: float) -> dict:
    result = collection.update_one(
        {match_field: match_value},
        {"$set": {update_field: update_value}}
    )
    return {"modified_count": result.modified_count}

@mcp.tool()
def delete_by_field(field_name: str, field_value: float) -> dict:
    result = collection.delete_one({field_name: field_value})
    return {"deleted_count": result.deleted_count}

@mcp.tool()
def create_index_on_field(field_name: str) -> dict:
    index_name = collection.create_index([(field_name, 1)])
    return {"index_created": index_name}

if __name__ == "__main__":
    result = query_by_field("Temperature", 24.6)  
    print("Query result:", result)
