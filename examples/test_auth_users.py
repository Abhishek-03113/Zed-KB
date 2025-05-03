"""
Example script demonstrating how to authenticate with the dummy test users.

This script shows how to use the test users defined in the config:
1. admin user - with admin role
2. newuser - with user role

To run this script:
    python examples/test_auth_users.py
"""

import sys
import os
import json
import requests
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the necessary modules
from src.zed_kb.auth.simple_auth import UserInfo

# Define the API endpoint (assuming default port)
API_BASE_URL = "http://localhost:8000"

def test_authentication(username, password, expected_role):
    """Test authentication for a specific user"""
    
    print(f"\nTesting authentication for user: {username}")
    print(f"Expected role: {expected_role}")
    
    # In a real application, you would use a proper authentication endpoint
    # For this example, we'll demonstrate how to include credentials in headers
    headers = {
        "X-User-ID": username,
        "X-User-Role": expected_role,
        "Authorization": f"Basic {username}:{password}"
    }
    
    # Make a request to a protected endpoint
    try:
        # This is just an example - modify to match your actual API endpoints
        response = requests.get(f"{API_BASE_URL}/api/documents", headers=headers)
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Authentication successful!")
            print(f"Response: {response.json()}")
        else:
            print(f"Authentication failed: {response.text}")
    except requests.RequestException as e:
        print(f"Request error: {e}")
        print("Make sure your API server is running (python run_server.py)")

def main():
    """Main function to demonstrate API testing with dummy users"""
    print("Testing API with dummy users")
    
    # Load user credentials from config
    config_path = PROJECT_ROOT / "src" / "zed_kb" / "config" / "schema.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    users = config["metadata"]["allowed_users"]
    
    # Test admin user
    admin_user = next((user for user in users if user["username"] == "admin"), None)
    if admin_user:
        test_authentication(
            admin_user["username"], 
            admin_user["password"], 
            admin_user["role"]
        )
    
    # Test regular user
    regular_user = next((user for user in users if user["username"] == "newuser"), None)
    if regular_user:
        test_authentication(
            regular_user["username"], 
            regular_user["password"], 
            regular_user["role"]
        )

if __name__ == "__main__":
    main()