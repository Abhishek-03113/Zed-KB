# Document Processing Module

This module handles all aspects of document ingestion, processing, and preparation for the Zed-KB knowledge base system.

## Overview

The document processing pipeline transforms raw documents into searchable, security-aware chunks with extracted metadata. This module serves as the foundation for document management within Zed-KB.

## Components

### document_loader.py

Handles the loading and initial processing of various document formats:
- PDF documents
- Microsoft Word documents (.docx)
- HTML content
- Plain text files
- Markdown documents

The loader extracts raw text while preserving essential structural information.

### document_chunker.py

Splits documents into semantically meaningful chunks for efficient retrieval:
- Intelligent chunking based on content boundaries
- Configurable chunk sizes and overlaps
- Header/section-aware chunking to maintain context
- Special handling for code, tables, and other structured content

### metadata_extractor.py

Extracts valuable metadata from documents to enhance search and filtering:
- Document type, creation date, and modification date
- Author information and source details
- Security classification tags
- Topic extraction and categorization
- Named entity recognition for key concepts

### document_indexer.py

Manages the document indexing process:
- Coordinates the document processing pipeline
- Assigns unique identifiers to documents and chunks
- Handles security classification and access controls
- Manages document versioning and updates

## Usage

```python
from zed_kb.document_processing.document_loader import DocumentLoader
from zed_kb.document_processing.document_chunker import DocumentChunker
from zed_kb.document_processing.metadata_extractor import MetadataExtractor
from zed_kb.document_processing.document_indexer import DocumentIndexer

# Create processing pipeline
loader = DocumentLoader()
chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
metadata_extractor = MetadataExtractor()
indexer = DocumentIndexer()

# Process a document
document = loader.load("path/to/document.pdf")
chunks = chunker.split_document(document)
metadata = metadata_extractor.extract(document)
indexer.index(document, chunks, metadata, security_level="confidential")
```

## Security Features

The document processing module supports Zed-KB's tiered security model:
- Document-level security classification
- Chunk-level security inheritance
- Metadata scrubbing for sensitive information
- Access control integration