#!/usr/bin/env python
# Streamlit RAG Chat Application with document management

import os
import sys
import tempfile
from pathlib import Path
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import base64
import logging
from datetime import datetime
from functools import lru_cache

import streamlit as st
from streamlit_option_menu import option_menu
import requests
import httpx

# Load environment variables
import dotenv
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configuration
API_URL = os.getenv("API_URL", "zedd.up.railway.app")
if not API_URL.startswith(("http://", "https://")):
    API_URL = f"https://{API_URL}"

# Application constants
TIMEOUT_SECONDS = 10
MAX_UPLOAD_SIZE_MB = 20
ALLOWED_EXTENSIONS = ["pdf", "txt", "docx", "md", "csv"]
CACHE_TTL = 300  # seconds (5 minutes)

# UI and styling functions


def add_logo():
    """Add Zedd logo to sidebar"""
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"]::before {
            content: "Zedd";
            display: block;
            margin-left: 20px;
            margin-top: 20px;
            font-size: 30px;
            font-weight: bold;
            color: #4CAF50;
        }
        [data-testid="stSidebarNav"]::after {
            content: "An AI powered internal knowledge base";
            display: block;
            margin-left: 20px;
            margin-bottom: 20px;
            font-size: 14px;
            font-style: italic;
            color: #666;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600)
def set_page_style():
    """Set global page styling with material design influences"""
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color: white;
    }
    div.stButton > button:active {
        background-color: #3e8e41;
        color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: #f0f0f0;
    }
    /* Document card styling */
    .doc-card {
        background: linear-gradient(145deg, #ffffff, #f7f9fc);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        padding: 18px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
        border-left: 4px solid #4CAF50;
    }
    .doc-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.1);
    }
    .doc-card h4 {
        font-size: 18px;
        margin-bottom: 8px;
        color: #333;
    }
    .doc-card p {
        font-size: 14px;
        color: #666;
    }
    .doc-card.pdf-doc {
        border-left-color: #E53935;
    }
    .doc-card.docx-doc {
        border-left-color: #1E88E5;
    }
    .doc-card.txt-doc {
        border-left-color: #7CB342;
    }
    .doc-card.md-doc {
        border-left-color: #8E24AA;
    }
    .doc-card.csv-doc {
        border-left-color: #FB8C00;
    }
    .doc-card-footer {
        display: flex;
        justify-content: space-between;
        margin-top: 12px;
        align-items: center;
    }
    .doc-type-tag {
        display: inline-block;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        color: white;
    }
    .doc-type-tag.pdf {
        background-color: #E53935;
    }
    .doc-type-tag.docx {
        background-color: #1E88E5;
    }
    .doc-type-tag.txt {
        background-color: #7CB342;
    }
    .doc-type-tag.md {
        background-color: #8E24AA;
    }
    .doc-type-tag.csv {
        background-color: #FB8C00;
    }
    .doc-type-tag.unknown {
        background-color: #757575;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_icon_html(icon_name: str, color: str = "black", size: int = 24) -> str:
    """Create HTML for Material Design icons"""
    return f"""
    <span style="color: {color}; font-size: {size}px; vertical-align: middle;">
        <span class="material-icons">{icon_name}</span>
    </span>
    """


@st.cache_data(ttl=3600)
def load_css():
    """Load external CSS resources for Material Design"""
    st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500;700;900&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# API interaction functions


@st.cache_resource(ttl=600)
def get_api_session():
    """Create a persistent session for API calls to improve performance"""
    session = requests.Session()
    return session


def api_request(endpoint: str, method: str = "GET", data: dict = None,
                files: dict = None, auth: bool = True, headers: dict = None) -> Tuple[dict, int]:
    """Central function for all API requests with error handling and session reuse"""
    url = f"{API_URL}/{endpoint}"
    default_headers = {}

    session = get_api_session()

    # Add auth header if needed and available
    if auth and "username" in st.session_state and "password" in st.session_state:
        auth_header = json.dumps({
            "username": st.session_state.username,
            "password": st.session_state.password
        })
        default_headers["Authorization"] = auth_header

    if headers:
        default_headers.update(headers)

    try:
        if method.upper() == "GET":
            response = session.get(
                url, headers=default_headers, timeout=TIMEOUT_SECONDS)
        elif method.upper() == "POST":
            response = session.post(url, json=data, files=files,
                                    headers=default_headers, timeout=TIMEOUT_SECONDS)
        else:
            return {"error": f"Unsupported method: {method}"}, 400

        # Handle error responses
        if response.status_code >= 400:
            logger.error(
                f"API error: {response.status_code} - {response.text}")
            try:
                error_data = response.json()
                return error_data, response.status_code
            except:
                return {"error": f"API error: {response.status_code}"}, response.status_code

        return response.json(), response.status_code

    except requests.exceptions.Timeout:
        logger.error(f"API timeout: {url}")
        return {"error": "API request timed out. Please try again later."}, 504
    except requests.exceptions.ConnectionError:
        logger.error(f"API connection error: {url}")
        return {"error": "Could not connect to API server. Please check your internet connection."}, 503
    except Exception as e:
        logger.error(f"API request error: {str(e)}")
        return {"error": f"Error: {str(e)}"}, 500


@st.cache_data(ttl=30, show_spinner=False)
def authenticate_with_api(username: str, password: str) -> Optional[dict]:
    """Authenticate user and cache result briefly to prevent repeated calls"""
    data, status_code = api_request(
        endpoint="login",
        method="POST",
        data={
            "username": username,
            "password": password,
            "role": "",
            "security_level": ""
        },
        auth=False
    )

    if status_code == 200:
        return data.get("user", {})
    return None


def register_user_with_api(username: str, password: str, role: str = "user") -> Tuple[bool, str]:
    """Register a new user account"""
    data, status_code = api_request(
        endpoint="signup",
        method="POST",
        data={
            "username": username,
            "password": password,
            "role": role,
            "security_level": "public"
        },
        auth=False
    )

    if status_code == 200:
        return True, "User registered successfully"
    else:
        error_msg = data.get("error", "Failed to register user")
        return False, error_msg


@st.cache_data(ttl=30)
def check_api_health() -> bool:
    """Check if API server is responsive"""
    try:
        data, status_code = api_request("health", auth=False)
        return status_code == 200
    except:
        return False

# UI Component helpers


def show_success_message(message: str, key: str = None):
    """Display a styled success message"""
    st.success(message, icon="✅")


def show_error_message(message: str, key: str = None):
    """Display a styled error message"""
    st.error(message, icon="❌")


def render_chat_message(message: dict, key: str):
    """Render a chat message with user/assistant styling and source citations"""
    role = message.get("role", "")
    content = message.get("content", "")
    sources = message.get("sources", [])

    with st.container():
        col1, col2 = st.columns([1, 20])

        with col1:
            if role == "user":
                st.write("👤")
            else:
                st.write("🤖")

        with col2:
            st.markdown(f"**{role.capitalize()}**")
            # Make the welcome message larger
            if role == "assistant" and content == "I am Zedd, your assistant.":
                st.markdown(f"<h3 style='margin-top: 0;'>{content}</h3>", unsafe_allow_html=True)
            else:
                st.write(content)

            # Show sources if available (for assistant messages)
            if role == "assistant" and sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for i, source in enumerate(sources):
                        st.markdown(
                            f"**Source {i+1}:** {source.get('title', 'Unknown')}")
                        st.markdown(f"*{source.get('snippet', '')}*")
                        st.divider()


def login_signup_page():
    """Login and signup interface with form validation"""
    st.title("📚 Zedd Knowledge Base")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px 0;'>
                <h2>Welcome to Zedd</h2>
                <p>Your AI-powered Knowledge Base</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Login and signup tabs
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

        # Login tab
        with tab1:
            with st.form("login_form"):
                st.subheader("Login to your account")
                username = st.text_input(
                    "Username", key="login_username", placeholder="Enter your username")
                password = st.text_input(
                    "Password", type="password", key="login_password", placeholder="Enter your password")

                col1, col2 = st.columns([1, 1])
                with col1:
                    remember_me = st.checkbox("Remember me")

                submit_login = st.form_submit_button(
                    "Login", use_container_width=True)

                if submit_login:
                    if not username or not password:
                        st.error("Please enter both username and password")
                    else:
                        with st.spinner("Logging in..."):
                            user = authenticate_with_api(username, password)
                            if user:
                                # Store user info in session state
                                st.session_state.logged_in = True
                                st.session_state.username = user.get(
                                    "username")
                                st.session_state.role = user.get(
                                    "role", "user")
                                st.session_state.password = password
                                st.session_state.security_level = user.get(
                                    "security_level", "public")

                                show_success_message(
                                    f"Welcome back, {username}!")
                                st.rerun()
                            else:
                                show_error_message(
                                    "Invalid username or password")

        # Signup tab
        with tab2:
            with st.form("signup_form"):
                st.subheader("Create a new account")

                new_username = st.text_input(
                    "Username", key="signup_username", placeholder="Choose a username")
                new_password = st.text_input("Password", type="password", key="signup_password",
                                             placeholder="Choose a strong password")
                confirm_password = st.text_input("Confirm Password", type="password",
                                                 placeholder="Confirm your password")

                # Check if first user (admin) exists
                @st.cache_data(ttl=60)
                def check_first_user():
                    try:
                        data, status_code = api_request("users", auth=False)
                        return status_code == 200 and len(data.get("users", [])) > 0
                    except:
                        return False

                users_exist = check_first_user()

                # First user is automatically admin
                if not users_exist:
                    st.info("You'll be registered as the first admin user")
                    role = "admin"
                else:
                    role = "user"

                submit_signup = st.form_submit_button(
                    "Create Account", use_container_width=True)

                if submit_signup:
                    if not new_username or not new_password:
                        show_error_message(
                            "Please enter both username and password")
                    elif len(new_password) < 6:
                        show_error_message(
                            "Password must be at least 6 characters long")
                    elif new_password != confirm_password:
                        show_error_message("Passwords do not match")
                    else:
                        with st.spinner("Creating your account..."):
                            success, message = register_user_with_api(
                                new_username, new_password, role)
                            if success:
                                show_success_message(message)
                                # Auto-login after registration
                                st.session_state.logged_in = True
                                st.session_state.username = new_username
                                st.session_state.role = role
                                st.session_state.password = new_password
                                st.session_state.security_level = "public"
                                st.rerun()
                            else:
                                show_error_message(message)

        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; font-size: 0.8em; color: gray;'>"
            "© 2025 Zedd-KB. Built with Streamlit."
            "</div>",
            unsafe_allow_html=True
        )


@st.cache_data(ttl=60)
def fetch_users():
    """Fetch all users with caching"""
    data, status_code = api_request("users")
    if status_code == 200:
        return data.get("users", [])
    return []


def user_management_page():
    """Admin interface for user management and permissions"""
    if st.session_state.role != "admin":
        st.warning("You don't have permission to access this page.")
        return

    st.title("👥 User Management")

    # User list refresh
    refresh_users = st.button("🔄 Refresh User List")
    if refresh_users:
        fetch_users.clear()

    # Fetch and display users with caching
    with st.spinner("Loading users..."):
        users = fetch_users()

    # User table
    if users:
        st.subheader(f"Manage Users ({len(users)})")

        col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
        with col1:
            st.markdown("**Username**")
        with col2:
            st.markdown("**Role**")
        with col3:
            st.markdown("**Security Level**")
        with col4:
            st.markdown("**Actions**")

        st.markdown("---")

        # Display each user with actions
        for user in users:
            username = user.get("username")
            role = user.get("role")
            security_level = user.get("security_level", "public")

            col1, col2, col3, col4 = st.columns([3, 2, 2, 3])

            with col1:
                st.write(username)
            with col2:
                st.write(role)
            with col3:
                st.write(security_level)
            with col4:
                # Handle role change buttons
                if username != st.session_state.username:
                    if role == "admin":
                        if st.button(f"Demote to User", key=f"demote_{username}"):
                            with st.spinner(f"Updating {username} role..."):
                                data, status_code = api_request(
                                    f"remove_role/{username}")
                                if status_code == 200:
                                    show_success_message(
                                        f"User {username} demoted to user role")
                                    fetch_users.clear()
                                    st.rerun()
                                else:
                                    show_error_message(
                                        f"Failed: {data.get('error', 'Unknown error')}")
                    else:
                        if st.button(f"Promote to Admin", key=f"promote_{username}"):
                            with st.spinner(f"Updating {username} role..."):
                                data, status_code = api_request(
                                    f"give_role/{username}/admin")
                                if status_code == 200:
                                    show_success_message(
                                        f"User {username} promoted to admin role")
                                    fetch_users.clear()
                                    st.rerun()
                                else:
                                    show_error_message(
                                        f"Failed: {data.get('error', 'Unknown error')}")
    else:
        st.info("No users found.")

    # New user form
    st.markdown("---")
    st.subheader("Add New User")

    with st.form("add_user_form"):
        col1, col2 = st.columns([1, 1])
        with col1:
            new_username = st.text_input(
                "Username", placeholder="Enter username")
        with col2:
            new_password = st.text_input(
                "Password", type="password", placeholder="Enter password")

        col1, col2 = st.columns([1, 1])
        with col1:
            new_role = st.selectbox("Role", options=["user", "admin"], index=0)
        with col2:
            security_level = st.selectbox("Security Level",
                                          options=[
                                              "public", "internal", "confidential"],
                                          index=0)

        submitted = st.form_submit_button("Add User", use_container_width=True)

        if submitted:
            if not new_username or not new_password:
                show_error_message("Please fill all fields")
            else:
                with st.spinner("Adding new user..."):
                    success, message = register_user_with_api(
                        new_username, new_password, new_role)
                    if success:
                        show_success_message(
                            f"User {new_username} added successfully")
                        fetch_users.clear()
                        st.rerun()
                    else:
                        show_error_message(message)


@st.cache_data(ttl=60)
def fetch_documents():
    """Fetch document list with caching"""
    data, status_code = api_request("documents")
    if status_code == 200:
        return data.get("documents", [])
    return []


def document_management_page():
    """Admin interface for document upload and management"""
    if st.session_state.role != "admin":
        st.warning("You don't have permission to access this page.")
        return

    st.title("📄 Document Management")

    # Upload form
    st.subheader("Upload New Document")
    with st.container():
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a file to upload",
            type=ALLOWED_EXTENSIONS,
            help=f"Maximum size: {MAX_UPLOAD_SIZE_MB}MB"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            security_level = st.selectbox(
                "Security Level",
                options=["public", "internal", "confidential"],
                index=0,
                help="Determines who can access this document"
            )
        with col2:
            process_option = st.radio(
                "Processing Method",
                options=["Auto-chunk", "Use existing sections"],
                horizontal=True
            )

        if uploaded_file is not None:
            if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                st.error(
                    f"File size exceeds the maximum limit of {MAX_UPLOAD_SIZE_MB}MB")
            else:
                file_details = {
                    "Filename": uploaded_file.name,
                    "Size": f"{uploaded_file.size / 1024:.1f} KB",
                    "Type": uploaded_file.type
                }

                st.json(file_details)

                if st.button("Upload Document", use_container_width=True, type="primary"):
                    with st.spinner("Uploading and processing document..."):
                        # Handle document upload
                        temp_file = tempfile.NamedTemporaryFile(delete=False)
                        temp_file.write(uploaded_file.getvalue())
                        temp_file.close()

                        files = {"file": (uploaded_file.name, open(
                            temp_file.name, "rb"), "application/octet-stream")}
                        data = {"security_level": security_level}

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

                        os.unlink(temp_file.name)

                        if response.status_code == 200:
                            fetch_documents.clear()
                            show_success_message(
                                f"Document uploaded: {uploaded_file.name}")
                        else:
                            show_error_message(
                                f"Failed to upload: {response.text}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Document library
    st.subheader("Document Library")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "Search documents", placeholder="Enter keywords...")
    with col2:
        st.text("")
        refresh_btn = st.button("🔄 Refresh", use_container_width=True)

    # Refresh document list if requested
    if refresh_btn:
        fetch_documents.clear()

    # Get document list
    with st.spinner("Loading documents..."):
        documents = fetch_documents()

    # Display documents in grid
    if documents:
        # Filter by search term
        if search_term:
            documents = [
                doc for doc in documents if search_term.lower() in doc.lower()]

        if not documents:
            st.info("No documents match your search criteria.")
        else:
            # Create grid layout
            rows = [documents[i:i+3] for i in range(0, len(documents), 3)]

            for row in rows:
                cols = st.columns(3)
                for i, doc in enumerate(row):
                    with cols[i]:
                        doc_type = doc.split(
                            ".")[-1] if "." in doc else "unknown"
                        
                        # Set icon based on document type
                        icon = "📄"
                        if doc_type == "pdf":
                            icon = "📕"
                        elif doc_type in ["doc", "docx"]:
                            icon = "📘"
                        elif doc_type == "txt":
                            icon = "📝"
                        elif doc_type == "md":
                            icon = "📋"
                        elif doc_type == "csv":
                            icon = "📊"

                        # Get document date (would normally come from metadata)
                        doc_date = "May 4, 2025"

                        st.markdown(f"""
                        <div class='doc-card {doc_type}-doc'>
                            <h4>{icon} {doc}</h4>
                            <div class='doc-card-footer'>
                                <span class='doc-type-tag {doc_type}'>{doc_type.upper()}</span>
                                <small>Added on {doc_date}</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.button(
                                "View Details", key=f"view_{doc}", use_container_width=True)
                        with col2:
                            st.button(
                                "Delete", key=f"delete_{doc}", use_container_width=True)
    else:
        st.info("No documents have been uploaded yet.")


def chat_interface():
    """RAG chat interface for querying knowledge base"""
    st.title("💬 Zedd Chat")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Add welcome message if chat is empty
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "I am Zedd, your assistant."
        })

    # Chat options sidebar
    with st.sidebar:
        st.subheader("Chat Options")

        # Model selection for admin users
        model_params = {}
        if st.session_state.role == "admin":
            llm_model = st.selectbox(
                "Select Model",
                options=["Gemini", "OpenAI", "Claude"],
                index=0,
                help="Select which AI model to use"
            )
            model_params["model"] = llm_model.lower()

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Lower is more factual, higher is more creative"
            )
            model_params["temperature"] = temperature

        # RAG results count for all users
        results_to_consider = st.slider(
            "Results to Consider",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of document chunks to consider"
        )
        model_params["results_to_consider"] = results_to_consider

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = [{
                "role": "assistant",
                "content": "Chat history cleared. How can I help you today?"
            }]
            st.rerun()

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            render_chat_message(msg, f"msg_{i}")

    # Chat input
    st.markdown("---")
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Update chat display
        with chat_container:
            render_chat_message(
                st.session_state.chat_history[-1], f"msg_user_{time.time()}")

        # Get AI response
        with st.spinner("Thinking..."):
            try:
                # Prepare request with parameters
                data = {"query": user_input}
                data.update(model_params)

                data, status_code = api_request(
                    endpoint="chat",
                    method="POST",
                    data=data
                )

                if status_code == 200:
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])

                    # Add response to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                    # Update display with new message
                    with chat_container:
                        render_chat_message(
                            st.session_state.chat_history[-1], f"msg_bot_{time.time()}")
                else:
                    error_message = data.get("error", "Failed to get response")
                    st.error(f"Error: {error_message}")

                    # Add error message to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {error_message}"
                    })
            except Exception as e:
                st.error(f"Error: {str(e)}")

                # Add error message to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}"
                })

        # Update UI
        st.rerun()


def profile_page():
    """User profile and account management"""
    st.title("👤 Your Profile")

    col1, col2 = st.columns([1, 2])

    with col1:
        # User avatar
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; 
                      background-color: white; border-radius: 10px; 
                      box-shadow: 0 0 10px rgba(0,0,0,0.1);">
                <div style="width: 100px; height: 100px; border-radius: 50%; 
                          background-color: #4CAF50; margin: 0 auto; 
                          display: flex; align-items: center; justify-content: center; 
                          color: white; font-size: 3em;">
                    {st.session_state.username[0].upper()}
                </div>
                <h3>{st.session_state.username}</h3>
                <p>{st.session_state.role}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Account details
        st.subheader("Account Information")
        st.markdown(f"**Username:** {st.session_state.username}")
        st.markdown(f"**Role:** {st.session_state.role}")
        st.markdown(f"**Security Level:** {st.session_state.security_level}")

        if st.button("Change Password", use_container_width=True):
            st.warning("Password change functionality coming soon!")


def main_app():
    """Main application with navigation menu"""
    with st.sidebar:
        add_logo()
        st.markdown("---")

        # Navigation options based on role
        menu_options = ["Chat", "Profile"]
        if st.session_state.role == "admin":
            menu_options.extend(["Document Manager", "User Manager"])

        # Initialize current page state
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Chat"

        # Nav menu with state preservation
        selected = option_menu(
            "Navigation",
            menu_options,
            icons=["chat-dots", "person-circle",
                   "file-earmark-text", "people"],
            menu_icon="list",
            default_index=menu_options.index(st.session_state.current_page),
        )

        # Update current page state
        if selected != st.session_state.current_page:
            st.session_state.current_page = selected

        # User info and logout
        st.markdown("---")
        st.write(f"Logged in as: **{st.session_state.username}**")
        st.write(f"Role: **{st.session_state.role}**")

        if st.button("Logout", use_container_width=True):
            # Clean session state on logout
            for key in list(st.session_state.keys()):
                if key not in ["logged_in", "username", "role", "password"]:
                    del st.session_state[key]

            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.password = None
            st.rerun()

    # Render selected page
    if selected == "Chat":
        chat_interface()
    elif selected == "Document Manager":
        document_management_page()
    elif selected == "User Manager":
        user_management_page()
    elif selected == "Profile":
        profile_page()


def streamlit_app():
    """Main application entry point"""
    st.set_page_config(
        page_title="Zedd Knowledge Base",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply styles
    set_page_style()
    load_css()

    # Init session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Check API health only for logged-in users
    if st.session_state.logged_in:
        api_healthy = check_api_health()
        if not api_healthy:
            st.warning(
                "⚠️ API server may be unavailable or experiencing issues. Some features may not work correctly.")

    # Show appropriate view based on login state
    if not st.session_state.logged_in:
        login_signup_page()
        return

    # Launch main app for logged-in users
    main_app()


if __name__ == "__main__":
    streamlit_app()
