# KEP-003: RAG-Enhanced Knowledge Assistant for Contextual Help

## Summary

This enhancement proposal introduces a Retrieval-Augmented Generation (RAG) system to the Brok chatbot to provide contextual knowledge assistance and improved tool discovery. The current system relies on short-term context (KEP-001) and has no persistent knowledge beyond the current session. This proposal adds a simple RAG implementation using ramalama to enhance user assistance with tool usage examples, community knowledge, and interactive help while maintaining the project's core philosophy of simplicity and incremental development.

**Status**: Proposed

## Motivation

### Current Limitations

1. **Limited Tool Discovery**: Users don't know what tools are available or how to use them effectively
2. **No Persistent Knowledge**: Bot cannot remember successful interactions or learn from past conversations
3. **Repetitive Questions**: Common questions about strims.gg, tool usage, and bot capabilities require repeated explanations
4. **Poor Tool Examples**: Current tool descriptions are static and lack contextual usage examples
5. **No Community Context**: Bot lacks knowledge about strims.gg community, common topics, and user patterns

### Use Cases

- **Tool Assistance**: "How do I check weather?" → Bot retrieves and shares relevant tool examples
- **Community Knowledge**: "What is WSG?" → Bot retrieves strims.gg-specific information
- **Interactive Help**: "What can you do?" → Bot provides contextual capabilities based on current conversation
- **Learning Enhancement**: Bot remembers successful tool interactions for future reference
- **Troubleshooting**: "Calculator not working" → Bot retrieves common solutions and examples

## Goals

### Primary Goals

- Implement simple RAG system using ramalama for knowledge retrieval
- Create curated knowledge base with tool examples and community information
- Enhance tool discovery and usage through contextual examples
- Provide interactive help system that learns from successful interactions
- Maintain backward compatibility with existing prompt and context systems

### Secondary Goals

- Foundation for future advanced knowledge features
- Improved user onboarding and bot capability discovery
- Community-specific knowledge integration
- Performance optimization for real-time chat responses

## Non-Goals

- Complex vector databases or heavy ML infrastructure
- Real-time web scraping or dynamic content fetching
- Advanced RAG techniques (re-ranking, hybrid search, etc.)
- User-specific personalization beyond session scope
- Integration with external knowledge bases or APIs
- Full conversation memory persistence across bot restarts

## Proposal

### Increment A: RAG Foundation and Static Knowledge Base

**Objective**: Implement basic RAG infrastructure with curated static knowledge

#### Actionable Goals

1. **RAG Infrastructure Setup**
   - [ ] Add ramalama dependency and configuration to `pyproject.toml`
   - [ ] Create `KnowledgeRetriever` class in `brok/knowledge/retriever.py`
   - [ ] Add `RAG_ENABLED` feature flag (default: false) to `BotConfig`
   - [ ] Implement simple vector store initialization with ramalama

2. **Static Knowledge Base Creation**
   - [ ] Create `knowledge/base/` directory with curated content:
     - `tool_examples.md` - Comprehensive tool usage examples and patterns
     - `community_faq.md` - strims.gg community information and common questions  
     - `bot_capabilities.md` - Bot features, commands, and usage guide
     - `troubleshooting.md` - Common issues and solutions
   - [ ] Implement knowledge base loading and indexing on startup

3. **Basic Retrieval Integration**
   - [ ] Add `retrieve_relevant_knowledge()` method to find relevant information
   - [ ] Integrate retrieval results into prompt context (compatible with KEP-002 XML)
   - [ ] Add knowledge source attribution in responses

**Success Criteria**: RAG system can retrieve and include relevant knowledge in bot responses

### Increment B: Dynamic Learning and Tool Enhancement

**Objective**: Add learning capabilities and enhanced tool assistance

#### Actionable Goals

1. **Successful Interaction Learning**
   - [ ] Track successful tool executions with context in `interaction_memory.json`
   - [ ] Store user questions that led to helpful responses
   - [ ] Implement periodic knowledge base updates from learned interactions

2. **Enhanced Tool Discovery**
   - [ ] Integrate RAG with `ToolRegistry.get_tools_description()` for contextual examples
   - [ ] Add tool usage pattern recognition and suggestions
   - [ ] Provide contextual tool recommendations based on user queries

3. **Interactive Help System**
   - [ ] Implement help command with RAG-enhanced responses
   - [ ] Add capability discovery based on conversation context
   - [ ] Create dynamic tool usage examples from successful interactions

**Success Criteria**: Measurable improvement in tool usage success rate and user help interactions

### Increment C: Performance Optimization and Production Readiness

**Objective**: Optimize RAG performance for real-time chat requirements

#### Actionable Goals

1. **Performance Optimization**
   - [ ] Implement caching for frequent knowledge retrievals
   - [ ] Optimize vector search with relevance filtering
   - [ ] Add retrieval timeout and fallback mechanisms

2. **Knowledge Base Management**
   - [ ] Automated knowledge base validation and health checks
   - [ ] Knowledge freshness tracking and update notifications
   - [ ] Memory usage monitoring and bounds enforcement

3. **Production Features**
   - [ ] Comprehensive error handling and graceful degradation
   - [ ] RAG performance metrics and monitoring
   - [ ] Knowledge source versioning and rollback capabilities

**Success Criteria**: <100ms retrieval latency, robust error handling, production-ready monitoring

## Design Details

### Knowledge Base Structure

```
knowledge/
├── base/                    # Static curated knowledge
│   ├── tool_examples.md    # Tool usage patterns and examples
│   ├── community_faq.md    # strims.gg community information
│   ├── bot_capabilities.md # Bot features and commands
│   └── troubleshooting.md  # Common issues and solutions
├── learned/                # Dynamic learned knowledge
│   ├── interaction_memory.json  # Successful interactions
│   └── tool_patterns.json      # Learned tool usage patterns
└── config/
    └── knowledge_config.yaml   # RAG configuration and weights
```

### Configuration

```python
@dataclass
class RAGConfig:
    """Configuration for RAG knowledge system."""
    enabled: bool = False
    knowledge_base_path: str = "knowledge/"
    max_retrieved_chunks: int = 3
    retrieval_threshold: float = 0.7
    learning_enabled: bool = True
    cache_ttl_seconds: int = 300

@dataclass
class BotConfig:
    # Existing fields...
    rag: RAGConfig = field(default_factory=RAGConfig)
```

### API Design

```python
class KnowledgeRetriever:
    """RAG-based knowledge retrieval system using ramalama."""
    
    async def retrieve_relevant_knowledge(
        self, 
        query: str, 
        context: str | None = None,
        max_chunks: int = 3
    ) -> list[KnowledgeChunk]:
        """Retrieve relevant knowledge for a query."""
        
    async def add_successful_interaction(
        self,
        user_query: str,
        bot_response: str,
        tools_used: list[str],
        success_indicators: dict[str, Any]
    ) -> None:
        """Learn from successful interactions."""
        
    def get_tool_examples(self, tool_name: str) -> list[str]:
        """Get contextual examples for a specific tool."""
```

### Integration with Existing Systems

#### Prompt Enhancement (KEP-002 Compatible)

```xml
<prompt version="1.0">
  <system role="assistant" name="brok">
    You are brok, a helpful AI assistant with access to knowledge and tools.
  </system>
  
  <knowledge_context>
    <retrieved_info source="tool_examples">
      Weather tool examples: Use "weather london" or {"tool": "weather", "params": {"city": "london"}}
    </retrieved_info>
    <retrieved_info source="community_faq">
      WSG = Winnie Swim Gang, a popular strims.gg community
    </retrieved_info>
  </knowledge_context>
  
  <tools><!-- existing tool definitions --></tools>
  <context><!-- existing KEP-001 context --></context>
  <request><!-- user query --></request>
</prompt>
```

#### Tool Registry Enhancement

```python
# Enhanced tool descriptions with RAG examples
def get_tools_description_with_examples(self) -> str:
    """Get tool descriptions enhanced with RAG examples."""
    base_description = self.get_tools_description()
    
    # Add contextual examples from RAG
    for tool_name in self.get_available_tools():
        examples = self._knowledge_retriever.get_tool_examples(tool_name)
        # Enhance description with examples
    
    return enhanced_description
```

## Knowledge Base Content Strategy

### Tool Examples (`tool_examples.md`)

```markdown
# Tool Usage Examples

## Weather Tool
### Basic Usage
- "What's the weather in London?" → {"tool": "weather", "params": {"city": "London"}}
- "Check weather for NYC" → Natural language detection works
- "Is it raining in Tokyo?" → Context-aware weather queries

### Common Patterns
- Always specify city name clearly
- Bot can handle abbreviations (NYC, LA, etc.)
- Works with "weather", "temperature", "forecast" keywords

### Troubleshooting
- "City not found" → Try full city name or add country
- Timeout errors → Weather service may be down, try again
```

### Community FAQ (`community_faq.md`)

```markdown
# strims.gg Community Knowledge

## Common Terms
- **WSG**: Winnie Swim Gang - popular community group
- **Pepe**: Common emote and meme reference
- **DGG**: Destiny.gg community overlap
- **Coomer**: Meme reference, not explicit content

## Chat Culture
- Fast-moving chat with lots of emotes
- Community in-jokes and references
- Gaming and politics discussion common
```

### Bot Capabilities (`bot_capabilities.md`)

```markdown
# Brok Bot Capabilities

## Available Tools
- **Weather**: Current weather for any city
- **Calculator**: Math calculations and expressions
- **DateTime**: Current time, timezone conversions

## Commands
- `@brok help` - Show available capabilities
- `@brok weather [city]` - Get weather information
- `@brok calc [expression]` - Perform calculations

## Keywords
- Responds to: `!bot`, `!ask`, `@brok`
- Natural language tool detection
- Context-aware responses
```

## Implementation Strategy

### Phase 1: Minimal Viable RAG (Increment A)

1. **ramalama Integration**
   ```bash
   pip install ramalama
   ```

2. **Simple Knowledge Loading**
   ```python
   # Load static markdown files into vector store
   knowledge_files = ["tool_examples.md", "community_faq.md"]
   for file in knowledge_files:
       chunks = load_and_chunk_markdown(file)
       vector_store.add_documents(chunks)
   ```

3. **Basic Retrieval**
   ```python
   # Simple retrieval in prompt building
   if rag_config.enabled:
       relevant_info = await knowledge_retriever.retrieve(user_query)
       prompt_context += f"\nRelevant Knowledge:\n{relevant_info}"
   ```

### Phase 2: Learning Integration (Increment B)

1. **Success Tracking**
   ```python
   # Track successful tool executions
   if tool_result.success:
       await knowledge_retriever.add_successful_interaction(
           user_query=original_query,
           bot_response=response,
           tools_used=[tool_name],
           success_indicators={"tool_success": True}
       )
   ```

2. **Dynamic Examples**
   ```python
   # Generate examples from learned interactions
   learned_examples = get_learned_tool_examples(tool_name)
   static_examples = get_static_tool_examples(tool_name)
   return combine_examples(static_examples, learned_examples)
   ```

## Testing Strategy

### Unit Testing
- Knowledge base loading and indexing
- Vector search and retrieval accuracy
- RAG integration with existing prompt system
- Learning system for successful interactions

### Integration Testing
- End-to-end knowledge retrieval in chat responses
- Tool discovery improvement measurement
- Prompt compatibility with KEP-002 XML formatting
- Performance testing with realistic knowledge base sizes

### Manual Testing
- User help query improvement evaluation
- Tool usage pattern recognition validation
- Community knowledge relevance assessment
- Response quality comparison (with/without RAG)

## Metrics and Success Criteria

### Success Metrics
- **Tool Discovery**: 25% increase in successful tool usage
- **Help Queries**: 40% improvement in help request satisfaction
- **Response Relevance**: Manual evaluation showing improved contextual responses
- **Performance**: <100ms retrieval latency in 95th percentile
- **Knowledge Coverage**: 80% of common questions answerable from knowledge base

### Monitoring Points
- RAG retrieval latency and cache hit rates
- Knowledge base usage patterns and popular queries
- Tool usage success rate before/after RAG enhancement
- User satisfaction indicators (fewer repeated questions)
- Memory usage and vector store performance

## Rollout Plan

### Increment A: Static Knowledge Foundation
- Deploy with `RAG_ENABLED=false` by default
- Enable in development environment for testing
- Validate zero impact on existing functionality
- Create initial curated knowledge base

### Increment B: Learning and Enhancement
- Enable RAG for tool-related queries first
- Gradual expansion to community knowledge
- Monitor performance and user feedback
- Iterate on knowledge base content

### Increment C: Production Optimization
- Performance optimization and monitoring
- Full RAG integration with all query types
- Knowledge base automation and maintenance
- Feature enabled by default based on metrics

## Dependencies

### Internal Dependencies
- KEP-001 enhanced context system for integration
- KEP-002 XML prompt formatting for knowledge injection
- Current `ToolRegistry` system for enhanced tool descriptions
- Existing `BotConfig` configuration management

### External Dependencies
- **ramalama**: RAG framework for vector search and retrieval
- **markdown**: Knowledge base content parsing
- Standard library: `json`, `pathlib`, `asyncio` for knowledge management

## Alternative Approaches Considered

### Alternative 1: Complex Vector Database (Pinecone, Weaviate)
- **Pros**: Advanced features, scalability, cloud management
- **Cons**: Over-engineering for chat bot scope, external dependencies, cost

### Alternative 2: LLM-Based Knowledge Generation
- **Pros**: Dynamic knowledge creation, no static content maintenance
- **Cons**: Hallucination risks, computational overhead, less controllable

### Alternative 3: External Knowledge APIs (Wikipedia, etc.)
- **Pros**: Vast knowledge, always current information
- **Cons**: API dependencies, rate limits, relevance filtering challenges

### Alternative 4: In-Memory Search (No RAG)
- **Pros**: Simple implementation, no external dependencies
- **Cons**: Limited semantic search, poor scalability, no learning

## Graduation Criteria

### Alpha (Increment A)
- [ ] RAG infrastructure implemented with ramalama
- [ ] Static knowledge base loaded and indexed
- [ ] Basic retrieval integration with existing prompt system
- [ ] Feature flag functional with zero impact when disabled
- [ ] Unit test coverage > 80% for knowledge components

### Beta (Increment B)
- [ ] Dynamic learning system operational
- [ ] Enhanced tool discovery measurably improved
- [ ] Interactive help system functional
- [ ] Integration testing complete with realistic scenarios
- [ ] Performance within acceptable bounds (<100ms retrieval)

### Stable (Increment C)
- [ ] Production-ready performance optimization
- [ ] Comprehensive error handling and monitoring
- [ ] Knowledge base maintenance automation
- [ ] Positive user feedback and usage metrics
- [ ] Feature enabled by default with successful rollout

## Risk Mitigation

### Technical Risks
- **RAG Latency**: Implement caching and async retrieval with fallbacks
- **Vector Store Size**: Monitor memory usage and implement pruning strategies
- **Knowledge Quality**: Curated content validation and community feedback loops

### Operational Risks
- **Dependency Management**: Pin ramalama version, test upgrades thoroughly
- **Knowledge Maintenance**: Automate validation, version control for knowledge base
- **Feature Complexity**: Maintain simple interfaces, gradual feature expansion

### Rollback Strategy
- Feature flag allows instant disable if issues arise
- Static knowledge base can be rolled back to previous versions
- Graceful degradation when RAG system unavailable
- Clear success criteria before advancing increments 