# Requirements Document

## Introduction

The Backlog Synthesizer is a multi-agent system that reads meeting transcripts, architecture documents, and existing backlog tickets, then produces structured user stories with acceptance criteria. The system uses an orchestrator pattern with specialized sub-agents for parsing, gap detection, and story writing, backed by a memory engine for state management and audit logging. This project demonstrates AI-first engineering practices including iterative problem framing, modular agent design, automated evaluation, and robust error handling.

## Glossary

- **Orchestrator_Agent**: The central coordination agent that receives all inputs, decides which sub-agent to invoke and in what order, manages session state, and writes to the Memory_Engine after each step
- **Parser_Agent**: A sub-agent responsible for ingesting transcripts and architecture documents, extracting decisions, pain points, and feature requests
- **Gap_Detection_Agent**: A sub-agent that compares newly extracted requests against the existing backlog to flag duplicates and conflicts
- **Story_Writer_Agent**: A sub-agent that produces structured user stories, epics, acceptance criteria, and feature tags from extracted information
- **Memory_Engine**: The persistence layer comprising short-term session state, long-term vector store, and an audit log
- **Short_Term_Memory**: An in-session state store (Python dict or Redis) holding current processing context
- **Long_Term_Memory**: A vector store (Chroma or FAISS) providing semantic search over backlog items and extracted information
- **Audit_Log**: A sequential record of every agent action taken during a session, showing how conclusions were reached
- **Meeting_Transcript**: A text or PDF document containing notes or recordings from a meeting
- **Architecture_Document**: A Confluence or wiki export describing system architecture decisions
- **Backlog_Ticket**: An existing work item from JIRA or GitHub Issues, provided as JSON
- **User_Story**: A structured output artifact containing a role, goal, benefit statement, and acceptance criteria
- **Epic**: A grouping of related User_Stories under a common theme or feature area
- **Golden_Dataset**: A curated set of 3-5 sample transcripts with hand-written ideal outputs used for evaluation
- **Session**: A single end-to-end execution of the Backlog Synthesizer processing a set of inputs to produce output stories

## Requirements

### Requirement 1: Document Ingestion

**User Story:** As a product manager, I want to provide meeting transcripts, architecture docs, and existing tickets as input, so that the system can analyze all relevant context for story generation.

#### Acceptance Criteria

1. WHEN a Meeting_Transcript in text format (.txt or .md) is provided, THE Parser_Agent SHALL extract the full text content and pass it to the processing pipeline within 5 seconds for files up to 10MB
2. WHEN a Meeting_Transcript in PDF format is provided, THE Parser_Agent SHALL convert the PDF to text using the document parsing Tool interface and extract the full text content, preserving paragraph boundaries
3. WHEN an Architecture_Document in HTML wiki export format is provided, THE Parser_Agent SHALL strip HTML markup, preserve heading hierarchy, and extract the content as structured text sections
4. WHEN Backlog_Tickets in JSON format are provided, THE Orchestrator_Agent SHALL validate the JSON against the expected ticket schema and load valid tickets into the Long_Term_Memory for comparison
5. IF an input document is malformed or unreadable, THEN THE Parser_Agent SHALL return an error object containing the document filename, the failure reason, and the byte offset or line number where parsing failed (if applicable)
6. IF a Backlog_Ticket JSON file fails schema validation, THEN THE Orchestrator_Agent SHALL reject the invalid entries, log the validation errors, and continue processing valid tickets

### Requirement 2: Orchestration and Routing

**User Story:** As a system operator, I want the orchestrator to coordinate sub-agents in the correct order, so that each processing step has the context it needs from prior steps.

#### Acceptance Criteria

1. WHEN a new Session is initiated with input documents, THE Orchestrator_Agent SHALL invoke the Parser_Agent before any other sub-agent
2. WHEN the Parser_Agent completes extraction, THE Orchestrator_Agent SHALL invoke the Gap_Detection_Agent with the extracted items and existing backlog data from Long_Term_Memory
3. WHEN the Gap_Detection_Agent completes analysis, THE Orchestrator_Agent SHALL invoke the Story_Writer_Agent with the deduplicated and prioritized items from the gap report
4. WHEN a sub-agent completes its invocation, THE Orchestrator_Agent SHALL write the sub-agent's output to the Memory_Engine before invoking the next sub-agent
5. IF a sub-agent invocation fails with a transient error, THEN THE Orchestrator_Agent SHALL retry the invocation up to 3 times with exponential backoff starting at 1 second (1s, 2s, 4s) before reporting the failure
6. IF a sub-agent invocation fails with a permanent error (authentication failure, invalid configuration, or schema violation), THEN THE Orchestrator_Agent SHALL NOT retry and SHALL immediately log the failure to the Audit_Log and halt the pipeline with an error result

### Requirement 3: Information Extraction

**User Story:** As a product manager, I want the system to extract decisions, pain points, and feature requests from raw documents, so that I do not have to manually comb through meeting notes.

#### Acceptance Criteria

1. WHEN a Meeting_Transcript is processed, THE Parser_Agent SHALL identify and extract decision items as structured records containing the decision text, source chunk index, character offset within the chunk, and a confidence score between 0.0 and 1.0
2. WHEN a Meeting_Transcript is processed, THE Parser_Agent SHALL identify and extract pain points as structured records containing the description, affected stakeholder (if identifiable), source chunk index, and a confidence score between 0.0 and 1.0
3. WHEN a Meeting_Transcript is processed, THE Parser_Agent SHALL identify and extract feature requests as structured records containing the request description, requester (if identifiable), source chunk index, and a confidence score between 0.0 and 1.0
4. WHEN an Architecture_Document is processed, THE Parser_Agent SHALL extract technical constraints and architectural decisions as structured records containing the constraint or decision text, the source section heading, and the type classification (constraint, decision, or principle)
5. THE Parser_Agent SHALL chunk documents into segments of no more than 2000 tokens with an overlap of 200 tokens between consecutive chunks to avoid splitting extracted items at boundaries
6. IF no decisions, pain points, or feature requests are identified in a document, THEN THE Parser_Agent SHALL return an empty extraction result with a metadata note indicating the document was processed but yielded no items

### Requirement 4: Duplicate and Conflict Detection

**User Story:** As a product manager, I want the system to detect when a new request duplicates or conflicts with an existing ticket, so that I avoid creating redundant or contradictory work items.

#### Acceptance Criteria

1. WHEN the Gap_Detection_Agent receives extracted feature requests, THE Gap_Detection_Agent SHALL compute semantic similarity against all Backlog_Tickets in the Long_Term_Memory within 30 seconds per request
2. WHEN a semantic similarity score exceeds 0.85 between a new request and an existing Backlog_Ticket, THE Gap_Detection_Agent SHALL flag the request as a potential duplicate and include the matching ticket identifier
3. WHEN two items have a semantic similarity score between 0.50 and 0.85 and contain mutually exclusive statements regarding the same feature attribute, THE Gap_Detection_Agent SHALL flag both items as conflicting and include a description identifying the contradicting statements
4. WHEN the Gap_Detection_Agent completes analysis of all extracted feature requests, THE Gap_Detection_Agent SHALL produce a gap report listing new items, duplicates, and conflicts with a confidence score between 0.0 and 1.0 for each classification
5. IF no existing Backlog_Tickets are available for comparison, THEN THE Gap_Detection_Agent SHALL mark all extracted items as new with a confidence score of 1.0 without duplicate or conflict checking
6. IF the semantic similarity computation fails or exceeds 30 seconds for a given request, THEN THE Gap_Detection_Agent SHALL mark that request as unprocessed in the gap report with an error indication describing the failure reason

### Requirement 5: User Story Generation

**User Story:** As a product manager, I want the system to produce well-structured user stories with acceptance criteria, so that my development team has clear, actionable work items.

#### Acceptance Criteria

1. WHEN the Story_Writer_Agent receives deduplicated items, THE Story_Writer_Agent SHALL produce a User_Story for each item containing a role, goal, and benefit statement in the format "As a [role], I want [goal], so that [benefit]"
2. WHEN the Story_Writer_Agent produces a User_Story, THE Story_Writer_Agent SHALL generate between 2 and 10 acceptance criteria for each User_Story, where each acceptance criterion describes a single testable condition
3. WHEN the Story_Writer_Agent produces a User_Story, THE Story_Writer_Agent SHALL assign between 1 and 5 feature tags to each User_Story based on keywords and topics identified in the source item content
4. WHEN two or more User_Stories share at least one feature tag, THE Story_Writer_Agent SHALL group them under a common Epic with a title that summarizes the shared functionality in 60 characters or fewer
5. THE Story_Writer_Agent SHALL format all output using a consistent JSON schema containing fields for title, user_story, acceptance_criteria, tags, and epic
6. IF a deduplicated item contains insufficient detail to populate the role, goal, or benefit statement, THEN THE Story_Writer_Agent SHALL produce the User_Story with placeholder text indicating the missing element and tag the story with a "needs-refinement" label

### Requirement 6: Memory and State Management

**User Story:** As a system operator, I want session state persisted and searchable, so that the system can reference prior context and I can audit how conclusions were reached.

#### Acceptance Criteria

1. THE Memory_Engine SHALL store all intermediate results from the current Session in Short_Term_Memory accessible by session identifier
2. THE Memory_Engine SHALL index all generated User_Stories and extracted items in Long_Term_Memory for semantic search
3. WHEN a new item is stored in Long_Term_Memory, THE Memory_Engine SHALL generate and store an embedding vector for the item content
4. THE Audit_Log SHALL record every sub-agent invocation with a timestamp, agent name, input summary (max 500 characters), output summary (max 500 characters), and duration in milliseconds
5. WHEN a Session completes, THE Memory_Engine SHALL retain the Audit_Log and Long_Term_Memory entries for a minimum of 30 days beyond the session lifetime
6. IF the Short_Term_Memory store is unavailable, THEN THE Memory_Engine SHALL fall back to an in-process Python dictionary, log a warning to the Audit_Log, and continue processing
7. WHEN a user requests Audit_Log entries for a given session identifier, THE Memory_Engine SHALL return all log entries for that session in chronological order
8. IF a user requests Audit_Log entries for an invalid or expired session identifier, THEN THE Memory_Engine SHALL return an empty result set with a message indicating the session was not found or has expired

### Requirement 7: Error Handling and Resilience

**User Story:** As a system operator, I want the system to handle failures gracefully with retries and clear error reporting, so that transient issues do not cause silent data loss.

#### Acceptance Criteria

1. IF an LLM API call fails with a transient error (HTTP 429, 500, 502, 503, or 504, or a network timeout), THEN THE Orchestrator_Agent SHALL retry the call up to 3 times with exponential backoff starting at 1 second with a maximum backoff of 8 seconds
2. IF all retry attempts for a sub-agent are exhausted, THEN THE Orchestrator_Agent SHALL log the failure to the Audit_Log including the error type, last error message, and number of attempts, and return a partial result with a status field set to "partial_failure" and an errors array listing each failed step
3. IF an LLM API call fails with a permanent error (HTTP 401, 403, or 404), THEN THE Orchestrator_Agent SHALL NOT retry and SHALL immediately log the failure and halt the pipeline with a status of "permanent_failure"
4. IF the Long_Term_Memory vector store does not respond within 10 seconds, THEN THE Gap_Detection_Agent SHALL treat it as unreachable, skip duplicate detection, and annotate the output with a warning that gap analysis was not performed
5. THE Orchestrator_Agent SHALL enforce a timeout of 120 seconds per sub-agent invocation
6. IF a sub-agent invocation exceeds the 120-second timeout, THEN THE Orchestrator_Agent SHALL terminate the invocation and treat it as a transient failure eligible for retry

### Requirement 8: Evaluation Framework

**User Story:** As an evaluator, I want an automated evaluation pipeline with a golden dataset, so that I can measure the quality of the system output against known-good results.

#### Acceptance Criteria

1. THE Evaluation_Framework SHALL include a Golden_Dataset of at least 3 sample Meeting_Transcripts with corresponding hand-written ideal User_Story outputs, where each sample contains at least one decision, one pain point, and one feature request
2. WHEN an evaluation run is triggered, THE Evaluation_Framework SHALL execute the full pipeline against each Golden_Dataset entry and compare outputs to expected results using both the keyword overlap score and the LLM-as-judge score
3. THE Evaluation_Framework SHALL compute a keyword overlap score as a normalized value between 0.0 and 1.0, calculated by dividing the number of matching keywords found in both generated and expected acceptance criteria by the total number of keywords in the expected acceptance criteria, using case-insensitive token matching
4. THE Evaluation_Framework SHALL support an LLM-as-judge mode that independently scores each generated User_Story on relevance, completeness, and clarity, each on an integer scale of 1 to 5, where 1 indicates the criterion is not met and 5 indicates the criterion is fully met
5. WHEN an evaluation run completes, THE Evaluation_Framework SHALL produce a summary report in JSON format containing per-case keyword overlap scores, per-case LLM-as-judge scores for each dimension, and aggregate metrics including mean and minimum scores across all test cases
6. IF the pipeline fails to produce output for a Golden_Dataset entry during an evaluation run, THEN THE Evaluation_Framework SHALL record the failure reason for that entry, assign it a score of 0 for all metrics, and continue processing the remaining entries

### Requirement 9: Modular Tool Abstractions

**User Story:** As a developer, I want each agent's tools to be defined behind clear interfaces, so that I can swap implementations without changing agent logic.

#### Acceptance Criteria

1. THE Parser_Agent SHALL access document parsing through a Tool interface that defines callable methods for PDF-to-text conversion and text chunking, such that the Parser_Agent code contains no direct references to the underlying PDF library or chunking implementation
2. THE Gap_Detection_Agent SHALL access embedding computation and vector search through a Tool interface that defines callable methods for generating embeddings and querying similar items, such that the Gap_Detection_Agent code contains no direct references to the underlying vector store implementation
3. THE Story_Writer_Agent SHALL access LLM generation through a Tool interface that defines callable methods for text generation, such that the Story_Writer_Agent code contains no direct references to the underlying model provider
4. WHEN a Tool implementation is replaced with an alternative that conforms to the same Tool interface, THE system SHALL require zero modifications to the agent module source code that invokes the Tool
5. THE system SHALL define each Tool interface with typed method signatures specifying input parameter types, return value types, and a defined error type that all implementations must raise on failure
6. IF a Tool implementation encounters an implementation-specific error, THEN THE Tool SHALL translate it into the interface-defined error type before propagating to the invoking agent
7. THE system SHALL provide a configuration mechanism that binds a concrete Tool implementation to each Tool interface without requiring changes to agent source code

### Requirement 10: Output Serialization

**User Story:** As a product manager, I want the final output serialized in a structured format, so that I can import the stories into my project management tool.

#### Acceptance Criteria

1. THE Story_Writer_Agent SHALL serialize the final output as a JSON document conforming to a published JSON schema definition
2. THE Story_Writer_Agent SHALL produce JSON output that satisfies the round-trip property: deserializing and re-serializing the output SHALL produce a semantically equivalent JSON document
3. WHEN the output contains multiple Epics, THE Story_Writer_Agent SHALL include an index array at the top level listing all Epic titles and the count of their constituent User_Stories
4. THE output JSON schema SHALL include fields for epic_title (string), stories (array of User_Story objects), and metadata (object containing session_id as string and timestamp as ISO 8601 datetime string)
5. IF serialization fails for any item, THEN THE Story_Writer_Agent SHALL return an error object containing the item title, the field that failed serialization, and the error description, while successfully serialized items are still included in the output
