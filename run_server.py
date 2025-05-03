#!/usr/bin/env python
"""
Command-line script to run the Zed-KB API server.
"""

import os
import sys
import argparse
import uvicorn
from dotenv import load_dotenv


def main():
    """Main function to run the Zed-KB API server."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the Zed-KB API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    
    args = parser.parse_args()
    
    # Check if environment file exists
    if os.path.exists(args.env_file):
        print(f"Loading environment from {args.env_file}")
        load_dotenv(args.env_file)
    else:
        print(f"Warning: Environment file {args.env_file} not found. Using default environment.")
    
    # Check for required environment variables
    required_vars = ["ASTRA_DB_APPLICATION_TOKEN", "GOOGLE_API_KEY", "PERMIT_IO_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Warning: Missing required environment variables: {', '.join(missing_vars)}")
        print("Some features may not work correctly.")
    
    # Run server
    print(f"Starting Zed-KB API server on {args.host}:{args.port}")
    uvicorn.run(
        "src.zed_kb.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()