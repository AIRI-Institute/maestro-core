# mmar-mapi

Multimodal architectures Maestro API — common pure/IO utilities for multi-modal architectures team.

## Installation

```bash
pip install mmar-mapi
```

## Overview

A Python library providing foundational utilities for building conversational AI systems, with particular focus on medical/healthcare applications.

### Core Features

- **Chat Management**: Framework for managing conversations with flexible message types
- **File Storage**: Resource management with deduplication support
- **Document Processing**: Content extraction from various document types
- **LLM Integration**: Interfaces for Large Language Model communication
- **XML Parsing**: Utilities for medical diagnostic data

## Quick Start

```python
from mmar_mapi import Chat, HumanMessage, AIMessage, make_content

# Create a chat session
chat = Chat(
    context=Context(session_id="session-123", user_id="user-456"),
    messages=[
        HumanMessage(content=make_content("Hello, AI!")),
        AIMessage(content=make_content("How can I help you today?"))
    ]
)
```

```python
from mmar_mapi import FileStorage, ResourceId

# Store and retrieve files
storage = FileStorage()
resource_id = storage.upload(b"file content")
content = storage.download(resource_id)
```

## Main Modules

### Chat System (`mmar_mapi.models.chat`)
- `Chat` — Conversation container with context and messages
- `HumanMessage`, `AIMessage`, `MiscMessage` — Message types
- `Content` — Flexible content supporting text, resources, commands, and widgets
- `Widget` — UI components for interactive conversations

### File Storage (`mmar_mapi.file_storage`)
- `FileStorageAPI` — Abstract interface for file operations
- `FileStorage` — Implementation with deduplication
- `FileStorageBasic` — Simple file access without storage
- `ResourceId` — Type-safe resource identifiers

### Models (`mmar_mapi.models`)
- `TrackInfo`, `DomainInfo` — Track and domain categorization
- `DiagnosticsXMLTagEnum`, `MTRSXMLTagEnum`, `UncertaintyXMLTagEnum` — Medical XML tags

### Utilities (`mmar_mapi.utils`)
- `make_session_id()` — Generate unique session identifiers
- `chunked()` — Split iterables into chunks
- `XMLParser` — XML processing utilities

### Services (`mmar_mapi.services`)

Service APIs for integrating with external systems:

**LLM Hub** (`llm_hub.py`)
- `LLMHubAPI` — Interface for LLM communication
- `LLMCallProps` — Call configuration
- `LLMPayload` — Message payload with attachments
- `LLMResponseExt` — Extended response format

**Document Extractor** (`document_extractor.py`)
- `DocumentExtractorAPI` — Extract content from documents
- `DocExtractionSpec` — Extraction configuration
- `ExtractedTable`, `ExtractedPicture`, `ExtractedPageImage` — Extraction results

**Chat Manager** (`chat_manager.py`)
- `ChatManagerAPI` — Manage domains, tracks, and chat responses

**Text Processing** (`text_generator.py`, `text_processor.py`)
- `TextGeneratorAPI` — Generate text from chat
- `TextProcessorAPI` — Process text with chat context

**Content Interpreter** (`content_interpreter.py`)
- `ContentInterpreterAPI` — Interpret content resources
- `ContentInterpreterRemoteAPI` — Remote content interpretation
- `ContentInterpreterRemoteResponse` — Remote response format

**Additional Services**
- `BinaryClassifiersAPI` — Binary text classification
- `TranslatorAPI` — Language translation
- `CriticAPI` — Text evaluation
- `TextExtractorAPI` — Extract text from resources
