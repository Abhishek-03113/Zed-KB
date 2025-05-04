# Zedd-KB: AI-Powered Knowledge Base with Fine-Grained AI Access Control

Zedd-KB is a secure, AI-powered internal knowledge base designed for organizations that require robust, fine-grained authorization controls over AI actions and data access. Built for the [Permit.io AI Access Control Challenge](https://www.permit.io/), Zedd-KB demonstrates how externalized authorization can safeguard sensitive information and AI capabilities in real-world applications.

## Why Zedd-KB?

Modern AI systems can access, summarize, and generate insights from vast internal data. Without proper controls, this power can lead to data leaks, unauthorized actions, or compliance violations. Zedd-KB solves this by integrating [Permit.io](https://www.permit.io/) for externalized, policy-driven access control, ensuring that:

- **Sensitive answers (e.g., HR, legal) require higher clearance**
- **AI cannot summarize or quote documents unless explicitly allowed**
- **All AI actions are subject to fine-grained, dynamic permission checks**

## Key Features

- **Document Ingestion & Chunking:** Supports PDF, DOCX, HTML, TXT, and more. Documents are split into semantically meaningful chunks for efficient retrieval.
- **Metadata Extraction & Security Classification:** Each document and chunk is tagged with security levels, allowed roles, and access policies.
- **Vector Database Integration:** Uses Pinecone for scalable, semantic search over internal documents.
- **RAG Pipeline with Gemini LLM:** Retrieval-Augmented Generation (RAG) answers user queries using only authorized, relevant content.
- **Permit.io Authorization:** All sensitive AI actions (search, upload, chat, admin) are protected by Permit.io policies, enabling approval workflows and dynamic access control.
- **Tiered Access Model:** Users are assigned roles (admin, user) and security levels. Only authorized users can access, quote, or export sensitive information.

## Architecture Overview

- **FastAPI Backend:** Handles authentication, document management, and RAG chat endpoints. Integrates with Permit.io for all permission checks.
- **Streamlit Frontend:** User-friendly interface for chat, document upload, and admin management.
- **MongoDB:** Stores user accounts and roles.
- **Pinecone:** Vector database for semantic document retrieval.
- **Permit.io:** Centralized, externalized authorization for all AI and data actions.

## Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Zed-KB
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install PyMuPDF unstructured markdown google-generativeai
```

### 3. Configure Environment Variables
- Copy `.env.example` to `.env` and fill in your API keys (Google, OpenAI, Pinecone, Permit.io, MongoDB, etc).

### 4. Run the Backend
```bash
python main.py
```

### 5. Run the Frontend
```bash
cd Zedd-frontend
streamlit run main.py
```

## Testing the App

To make testing easy, use these credentials:

**Admin:**
- username: `admin`
- password: `2025DEVChallenge`

**User:**
- username: `newuser`
- password: `2025DEVChallenge`

## How Authorization Works (with Permit.io)

- **Every API endpoint** (document upload, chat, user management) checks permissions via Permit.io before executing sensitive actions.
- **Roles and security levels** are enforced at both the API and AI layer. For example, only admins can upload or view all documents; users can only access public data.
- **Approval workflows** can be added for exporting or sharing generated insights, requiring explicit admin consent.
- **Permit.io policies** are externalized and can be updated without redeploying the app, enabling dynamic, auditable access control.

## Example Use Cases

- **Employee queries HR:** If a user asks about HR policies, the AI only returns public information unless the user has higher clearance.
- **Legal document summary:** Summarization or quoting of legal documents is blocked unless the document's metadata and the user's role allow it.
- **Exporting insights:** Attempting to export or share generated content can trigger an approval workflow via Permit.io.

## Extending & Customizing

- **Add new roles or security levels** in Permit.io and update your policies as your organization grows.
- **Integrate additional AI models** (OpenAI, Claude, etc.) by extending the vector store and LLM modules.
- **Customize approval workflows** for any sensitive AI action.

## Benefits of Externalized Authorization (vs. Traditional Approaches)

- **Centralized policy management:** Update access rules in Permit.io without code changes.
- **Dynamic, context-aware permissions:** Grant or revoke access instantly, even for new AI capabilities.
- **Auditability:** All access decisions are logged and traceable.
- **Separation of concerns:** Security logic is decoupled from application logic, reducing risk and complexity.

## Contributing

Pull requests and feedback are welcome! Please open an issue to discuss your ideas or report bugs.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built for the Permit.io AI Access Control Challenge. Learn more at [permit.io](https://www.permit.io/).*
