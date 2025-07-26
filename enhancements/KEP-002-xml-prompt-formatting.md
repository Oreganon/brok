# KEP-002: XML-Formatted Prompt Structure for Enhanced Context Organization

## Summary

This enhancement proposal introduces XML-based prompt formatting to improve how context, tools, and instructions are structured for LLM consumption. The current system uses simple text concatenation with `\n\n` separators, creating ambiguous boundaries between prompt sections. XML markup provides clearer semantic separation while maintaining backward compatibility and improving LLM comprehension.

**Status**: Draft - Planning Phase

## Motivation

### Current Limitations

1. **Poor Section Boundaries**: Unclear where system instructions end and context begins
2. **Unstructured Context**: Chat history lacks metadata (timestamps, sender roles, message types) 
3. **Flat Tool Descriptions**: Tools listed as plain text without clear parameter structures
4. **Limited Extensibility**: Difficult to add rich metadata or new prompt sections

### Benefits

- **Enhanced LLM Comprehension**: Structured sections with clear boundaries and metadata
- **Improved Tool Calling**: Structured schemas with examples increase parameter accuracy
- **Better Debugging**: XML structure makes prompt inspection significantly easier
- **Future Extensibility**: Easy addition of new sections without breaking compatibility

## Goals

- Replace text concatenation with structured XML formatting
- Improve LLM understanding of context boundaries and tool schemas  
- Maintain full backward compatibility with existing prompt templates
- Provide configurable XML formatting with feature flags
- Support rich metadata integration from KEP-001 context system

## Non-Goals

- Complex XML schema validation or processing overhead
- Breaking changes to existing configurations or LLM providers
- Advanced templating languages beyond simple XML markup

## Proposal

### Increment A: XML Foundation

**Objective**: Add optional XML formatting with feature flag

**Goals:**
- [ ] Implement `XMLPromptTemplate` extending `PromptTemplate`
- [ ] Add `XML_PROMPT_FORMATTING` feature flag (default: false)
- [ ] Ensure identical output when flag disabled (round-trip test equality)
- [ ] Basic XML sections: `<system>`, `<tools>`, `<context>`, `<request>`

**Success Criteria**: XML available as opt-in with zero behavior change when disabled

### Increment B: Enhanced Context Structure  

**Objective**: Integrate KEP-001 context with XML metadata

**Goals:**
- [ ] Individual `<message>` elements with sender, timestamp, type attributes
- [ ] Integration with existing `ContextMessage` dataclass
- [ ] Priority-based message ordering within XML structure

**Success Criteria**: Measurable improvement in context understanding via A/B testing

### Increment C: Structured Tool Integration

**Objective**: Enhanced tool descriptions with schemas and examples

**Goals:**  
- [ ] `<tool>` elements with structured parameter definitions
- [ ] Usage examples and validation rules within tool XML
- [ ] Integration with existing `ToolRegistry` system

**Success Criteria**: Significant improvement in tool calling accuracy (>10% success rate increase)

### Increment D: Performance Optimization

**Objective**: Optimize XML generation and validate token efficiency

**Goals:**
- [ ] Efficient XML generation with minimal string concatenation  
- [ ] Token overhead measurement with `tiktoken` (verify <20% increase)
- [ ] Context truncation optimization within XML boundaries
- [ ] Memory usage monitoring and bounds validation

**Success Criteria**: <5ms XML generation overhead, <20% token increase

### Increment E: Advanced Features (Future)

**Objective**: Advanced prompt engineering capabilities  

**Goals:**
- [ ] Conditional sections based on context
- [ ] Prompt versioning and schema evolution
- [ ] Custom XML namespaces for extensions

## Design Details

### XML Structure Example

```xml
<prompt version="1.0">
  <system role="assistant" name="brok">
    You are brok, a helpful AI assistant. Keep responses concise.
  </system>
  
  <tools count="1">
    <tool name="weather" category="information">
      <description>Get current weather for a location</description>
      <parameters>
        <parameter name="city" type="string" required="true">
          City name for weather lookup
          <example>London</example>
        </parameter>
      </parameters>
      <usage_example><![CDATA[
        {"tool": "weather", "params": {"city": "London"}}
      ]]></usage_example>
    </tool>
  </tools>
  
  <context window_size="10">
    <message sender="alice" timestamp="2025-01-26T15:30:00Z" type="user">
      @brok what's the weather?
    </message>
    <message sender="brok" timestamp="2025-01-26T15:30:15Z" type="bot" tool_used="weather">
      🤖 Current weather in London: Partly cloudy, 15°C
    </message>
  </context>
  
  <request sender="alice">
    <user_input>What about tomorrow?</user_input>
    <response_prompt>Assistant:</response_prompt>
  </request>
</prompt>
```

### Configuration

```python
@dataclass 
class XMLPromptConfig:
    """Configuration for XML prompt formatting."""
    enabled: bool = False
    include_metadata: bool = True  # timestamps, confidence scores
    include_examples: bool = True  # tool usage examples
    version: str = "1.0"

@dataclass
class BotConfig:
    # Existing fields...
    xml_prompt: XMLPromptConfig = field(default_factory=XMLPromptConfig)
```

**Environment Variables:**
```bash
XML_PROMPT_FORMATTING=false  # Main toggle - only this exposed initially
```

## Implementation

### Testing Strategy

**Unit Testing:**
- XML template creation and validation
- XML schema validation with `xmlschema` library in CI
- Backward compatibility (round-trip equality when flag off)

**Performance Testing:**  
- Token overhead measurement: `tiktoken` analysis on 1000 random prompts
- Token-limit regression testing (ensure prompts fit within provider context limits)
- XML generation performance (<5ms overhead target)

**Integration Testing:**
- A/B testing framework for XML vs text prompt comparison
- Tool calling accuracy measurement with structured vs flat descriptions

### Rollout Plan

**Increment A**: Feature flag deployment (disabled by default)
- Development environment testing with XML enabled
- Zero-impact validation in production with flag off
- **Fallback Plan**: Auto-disable if error rate >2% increase

**Increments B-C**: Gradual beta testing  
- Select user groups with XML enabled
- Performance validation with real workloads
- **Flag Removal**: After 2 stable releases and positive metrics

**Increment D**: Performance optimization and production readiness
- Full monitoring operational  
- Documentation and best practices published

## Metrics

**Success Metrics:**
- Tool calling success rate improvement (target: >10% increase)
- Context understanding improvements in A/B tests
- XML generation performance (<5ms overhead)
- Token efficiency (verify <20% increase via empirical measurement)

**Monitoring:**
- Feature flag adoption rates
- XML vs text prompt error rates  
- Tool calling accuracy deltas
- Token usage differences

## Dependencies

**Internal:**
- Current `PromptTemplate` system
- KEP-001 enhanced context (`ContextMessage` objects)  
- Existing `ToolRegistry` and schema system

**External:**
- None (uses Python stdlib for XML generation)
- Optional: `xmlschema` for validation in testing

## Graduation Criteria

**Alpha (A)**: XML foundation with feature flag, zero impact when disabled
**Beta (B-C)**: Context and tool integration, measurable improvements  
**Stable (D)**: Performance optimized, production monitoring operational
**Production**: Feature enabled by default based on positive metrics

## Appendix: Technical Details

### Token Overhead Analysis
- Empirical measurement plan using `tiktoken` on representative prompt corpus
- XML tags benefit from BPE merging in many tokenizers (reduces actual overhead)
- Trade-off: +15% tokens for significantly improved structure and debugging

### XML vs JSON Rationale  
- XML chosen over JSON for human readability and debugging capabilities
- Many LLMs pre-trained on HTML/XML markup (Anthropic, OpenAI models)
- Structured markup more important than minimal token count for this use case

### Rollback Strategy
- Feature flag allows instant rollback if issues arise
- Gradual rollout with error rate monitoring  
- Clear success criteria before advancing increments 