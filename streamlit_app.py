#!/usr/bin/env python
"""
RAG Chat Application - Streamlit Frontend

This file contains the Streamlit frontend for the RAG chat application, which:
- Provides a user interface for document upload and management
- Offers a chat interface for interacting with the RAG system
- Communicates with the FastAPI backend via HTTP requests
- Implements user authentication with admin and regular user roles
"""

import os
import sys
import tempfile
from pathlib import Path
import time
import json
import hashlib

# Streamlit imports
import streamlit as st
import requests

# Environment variables
import dotenv

# Load environment variables
dotenv.load_dotenv()

# FastAPI backend URL
API_URL = "http://localhost:8000"


# User authentication functions
def get_users():
    """Load users from the config file"""
    try:
        config_path = os.path.join("src", "zed_kb", "config", "schema.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            return config.get("metadata", {}).get("allowed_users", [])
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")
    return []


def save_users(users):
    """Save users to the config file"""
    try:
        config_path = os.path.join("src", "zed_kb", "config", "schema.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

            config["metadata"]["allowed_users"] = users

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            return True
    except Exception as e:
        st.error(f"Error saving user data: {str(e)}")
    return False


def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate(username, password):
    """Authenticate user with username and password"""
    users = get_users()

    # Hash the provided password
    hashed_password = hash_password(password)

    # Check if user exists and password matches
    for user in users:
        if user.get("username") == username and user.get("password") == hashed_password:
            return user

    return None


def register_user(username, password, role="user"):
    """Register a new user"""
    users = get_users()

    # Check if username already exists
    for user in users:
        if user.get("username") == username:
            return False, "Username already exists"

    # Create new user
    new_user = {
        "username": username,
        "password": hash_password(password),
        "role": role
    }

    users.append(new_user)

    # Save updated users
    if save_users(users):
        return True, "User registered successfully"
    else:
        return False, "Failed to register user"


def login_signup_page():
    """Display login/signup page"""
    st.title("📚 RAG-powered Chat Application")

    # Tabs for login and signup
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # Login tab
    with tab1:
        st.header("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input(
            "Password", type="password", key="login_password")

        if st.button("Login"):
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                user = authenticate(username, password)
                if user:
                    # Store user info in session state
                    st.session_state.logged_in = True
                    st.session_state.username = user.get("username")
                    st.session_state.role = user.get("role", "user")
                    # Store the password for API authentication
                    st.session_state.password = password
                    st.success(
                        f"Logged in as {username} ({st.session_state.role})")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    # Signup tab
    with tab2:
        st.header("Sign Up")
        new_username = st.text_input("Username", key="signup_username")
        new_password = st.text_input(
            "Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password")

        # Only add role selection if there are no users (first user is admin)
        users = get_users()
        if not users:
            st.info("First user will be registered as admin")
            role = "admin"
        else:
            role = "user"  # Default role for new users

        if st.button("Sign Up"):
            if not new_username or not new_password:
                st.error("Please enter both username and password")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = register_user(
                    new_username, new_password, role)
                if success:
                    st.success(message)
                    # Auto-login after successful registration
                    st.session_state.logged_in = True
                    st.session_state.username = new_username
                    st.session_state.role = role
                    # Store the password for API authentication
                    st.session_state.password = new_password
                    st.rerun()
                else:
                    st.error(message)


def streamlit_app():
    """Streamlit application for user interface"""
    st.set_page_config(
        page_title="RAG Chat Application",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # If not logged in, show login/signup page
    if not st.session_state.logged_in:
        login_signup_page()
        return

    # User is logged in
    st.title("📚 RAG-powered Chat Application")

    # Show logout button in sidebar
    with st.sidebar:
        st.write(
            f"Logged in as: **{st.session_state.username}** ({st.session_state.role})")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            # Clear chat history on logout
            if "chat_history" in st.session_state:
                st.session_state.chat_history = []
            st.rerun()

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Check if API is available
    if not check_api_health():
        st.error(
            "⚠️ Cannot connect to API server. Please make sure the API server is running.")
        st.info("Run the API server with: `python api_server.py`")
        return

    # Admin-only section: Document management
    if st.session_state.role == "admin":
        with st.sidebar:
            st.header("📄 Document Management")

            # Document upload
            with st.expander("Upload Document", expanded=True):
                uploaded_file = st.file_uploader(
                    "Choose a file", type=["pdf", "txt", "docx"])
                security_level = st.selectbox(
                    "Security Level",
                    options=["public", "internal", "confidential"],
                    index=0
                )

                if uploaded_file is not None and st.button("Upload"):
                    with st.spinner("Uploading document..."):
                        # Create a temporary file
                        temp_file = tempfile.NamedTemporaryFile(delete=False)
                        temp_file.write(uploaded_file.getvalue())
                        temp_file.close()

                        # Send to API for processing
                        files = {"file": (uploaded_file.name, open(
                            temp_file.name, "rb"), "application/octet-stream")}
                        data = {"security_level": security_level}

                        # Create auth header
                        auth_header = json.dumps({
                            "username": st.session_state.username,
                            "password": st.session_state.get("password", "")
                        })
                        headers = {"Authorization": auth_header}

                        response = requests.post(
                            f"{API_URL}/upload",
                            files=files,
                            data=data,
                            headers=headers
                        )

                        # Clean up temporary file
                        os.unlink(temp_file.name)

                        if response.status_code == 200:
                            st.success(
                                f"Document uploaded: {uploaded_file.name}")
                        else:
                            st.error(f"Failed to upload: {response.text}")

            # List documents (admin only)
            with st.expander("Manage Documents", expanded=False):
                if st.button("Refresh Document List"):
                    with st.spinner("Loading documents..."):
                        try:
                            response = requests.get(f"{API_URL}/documents")
                            if response.status_code == 200:
                                documents = response.json()["documents"]
                                if documents:
                                    st.table({"Documents": documents})
                                else:
                                    st.info("No documents found.")
                            else:
                                st.error("Failed to load documents")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

    # Main chat interface (available to all logged-in users)
    st.header("💬 Chat")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}:** {source['title']}")
                            st.markdown(f"*{source['snippet']}*")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create auth header with current user's credentials
                    auth_header = json.dumps({
                        "username": st.session_state.username,
                        "password": st.session_state.get("password", "")
                    })

                    headers = {"Authorization": auth_header}

                    response = requests.post(
                        f"{API_URL}/chat",
                        json={"query": prompt},
                        headers=headers
                    )

                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        sources = result.get("sources", [])

                        # Display assistant response
                        st.write(answer)

                        # Display sources in an expander
                        if sources:
                            with st.expander(f"Sources ({len(sources)})"):
                                for i, source in enumerate(sources):
                                    st.markdown(
                                        f"**Source {i+1}:** {source['title']}")
                                    st.markdown(f"*{source['snippet']}*")

                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.error(f"Error: {response.text}")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Sorry, I encountered an error: {response.text}"
                        })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })


def check_api_health():
    """Check if the API server is available"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    streamlit_app()
