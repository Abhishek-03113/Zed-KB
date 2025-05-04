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
def authenticate_with_api(username, password):
    """Authenticate user with username and password via API"""
    try:
        response = requests.post(
            f"{API_URL}/login",
            json={
                "username": username,
                "password": password,  # Send the password as-is, the server will hash it
                "role": "",  # These will be filled by the server
                "security_level": ""
            }
        )
        if response.status_code == 200:
            user_data = response.json()
            return user_data.get("user", {})
        return None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return None


def register_user_with_api(username, password, role="user"):
    """Register a new user via API"""
    try:
        response = requests.post(
            f"{API_URL}/signup",
            json={
                "username": username,
                "password": password,  # Send the password as-is, the server will hash it
                "role": role,
                "security_level": "public"  # Default security level for new users
            }
        )
        if response.status_code == 200:
            return True, "User registered successfully"
        else:
            error_data = response.json()
            return False, error_data.get("error", "Failed to register user")
    except Exception as e:
        return False, f"Registration error: {str(e)}"


def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()


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
                user = authenticate_with_api(username, password)
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

        # Check if first user (admin) exists
        try:
            response = requests.get(f"{API_URL}/users")
            users_exist = response.status_code == 200 and len(
                response.json().get("users", [])) > 0
        except:
            users_exist = False

        if not users_exist:
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
                success, message = register_user_with_api(
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
                            # Create auth header for admin access
                            auth_header = json.dumps({
                                "username": st.session_state.username,
                                "password": st.session_state.get("password", "")
                            })
                            headers = {"Authorization": auth_header}

                            response = requests.get(
                                f"{API_URL}/documents",
                                headers=headers
                            )
                            if response.status_code == 200:
                                documents = response.json()["documents"]
                                if documents:
                                    st.table({"Documents": documents})
                                else:
                                    st.info("No documents found.")
                            else:
                                st.error(
                                    f"Failed to load documents: {response.text}")
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
