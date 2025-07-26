use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

use crate::tools::ToolManager;

// Custom error type for better error handling
#[derive(Debug, Clone)]
pub enum InferenceError {
    Network(String),
    Parse(String),
    Provider(String),
    Tool(String),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::Network(msg) => write!(f, "Network error: {msg}"),
            InferenceError::Parse(msg) => write!(f, "Parse error: {msg}"),
            InferenceError::Provider(msg) => write!(f, "Provider error: {msg}"),
            InferenceError::Tool(msg) => write!(f, "Tool error: {msg}"),
        }
    }
}

impl std::error::Error for InferenceError {}

// Common prompt template to avoid duplication
fn format_chat_prompt(
    prompt: String,
    message_history: &[String],
    user_context: Option<&str>,
) -> String {
    let context = if message_history.is_empty() {
        "No previous messages.".to_string()
    } else {
        format!("Recent chat history:\n{}", message_history.join("\n"))
    };

    let user_info = user_context.unwrap_or("");

    format!("You are a chat bot named 'brok'. Respond with ONE short sentence only. Put your response in <reply></reply> tags.

If something is funny, add 'LUL' at the end.
If something is weird, add 'PeepoWeird' at the end.
If you don't know, add 'FeelsPepoMan' at the end.

{user_info}Recent messages: {context}

Question: {prompt}

Response:")
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub sender: String,
    pub response_tx: mpsc::UnboundedSender<String>,
}

#[derive(Debug, Clone)]
pub enum ProviderType {
    Ollama,
    LlamaCpp,
}

// Trait for different AI providers
#[async_trait::async_trait]
pub trait InferenceProvider: Send + Sync {
    async fn generate_response(
        &self,
        prompt: String,
        message_history: &[String],
        user_context: Option<&str>,
    ) -> Result<String, InferenceError>;
}

// Ollama API structures
#[derive(Serialize, Deserialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct OllamaResponse {
    response: String,
}

// llama.cpp API structures
#[derive(Serialize, Deserialize)]
struct LlamaCppRequest {
    prompt: String,
    n_predict: i32,
    temperature: f32,
    stop: Vec<String>,
    repeat_last_n: i32,
    repeat_penalty: f32,
    top_k: i32,
    top_p: f32,
}

#[derive(Serialize, Deserialize)]
struct LlamaCppResponse {
    content: String,
    generation_settings: Option<serde_json::Value>,
    model: Option<String>,
    prompt: Option<String>,
    stopped_eos: Option<bool>,
    stopped_limit: Option<bool>,
    stopped_word: Option<bool>,
    stopping_word: Option<String>,
    tokens_cached: Option<i32>,
    tokens_evaluated: Option<i32>,
    tokens_predicted: Option<i32>,
    truncated: Option<bool>,
}

// Ollama provider implementation
pub struct OllamaProvider {
    client: Client,
    api_endpoint: String,
    model: String,
}

impl OllamaProvider {
    pub fn new(client: Client, api_host: String, model: String) -> Self {
        let api_endpoint = format!("http://{api_host}/api/generate");
        Self {
            client,
            api_endpoint,
            model,
        }
    }
}

#[async_trait::async_trait]
impl InferenceProvider for OllamaProvider {
    async fn generate_response(
        &self,
        prompt: String,
        message_history: &[String],
        user_context: Option<&str>,
    ) -> Result<String, InferenceError> {
        let formatted_prompt = format_chat_prompt(prompt, message_history, user_context);

        let request = OllamaRequest {
            model: self.model.clone(),
            prompt: formatted_prompt,
            stream: false,
        };

        println!(
            "[DEBUG] Sending Ollama request to API at {}",
            self.api_endpoint
        );
        println!(
            "[DEBUG] Request body: {}",
            serde_json::to_string(&request).unwrap_or_else(|_| "Failed to serialize".to_string())
        );

        let response = self
            .client
            .post(&self.api_endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                println!("[DEBUG] HTTP request failed: {e}");
                InferenceError::Network(e.to_string())
            })?;

        println!(
            "[DEBUG] Received response with status: {}",
            response.status()
        );

        if !response.status().is_success() {
            return Err(InferenceError::Provider(format!(
                "Ollama API returned status: {}",
                response.status()
            )));
        }

        let ollama_response: OllamaResponse = response
            .json()
            .await
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        println!("[DEBUG] Ollama response: {}", ollama_response.response);

        Ok(ollama_response.response)
    }
}

// llama.cpp provider implementation
pub struct LlamaCppProvider {
    client: Client,
    api_endpoint: String,
}

impl LlamaCppProvider {
    pub fn new(client: Client, api_host: String) -> Self {
        let api_endpoint = format!("http://{api_host}/completion");
        Self {
            client,
            api_endpoint,
        }
    }
}

#[async_trait::async_trait]
impl InferenceProvider for LlamaCppProvider {
    async fn generate_response(
        &self,
        prompt: String,
        message_history: &[String],
        user_context: Option<&str>,
    ) -> Result<String, InferenceError> {
        let formatted_prompt = format_chat_prompt(prompt, message_history, user_context);

        let request = LlamaCppRequest {
            prompt: formatted_prompt,
            n_predict: 128,
            temperature: 0.7,
            stop: vec!["</reply>".to_string(), "\n\n".to_string()],
            repeat_last_n: 64,
            repeat_penalty: 1.1,
            top_k: 40,
            top_p: 0.95,
        };

        println!(
            "[DEBUG] Sending llama.cpp request to API at {}",
            self.api_endpoint
        );
        println!(
            "[DEBUG] Request body: {}",
            serde_json::to_string(&request).unwrap_or_else(|_| "Failed to serialize".to_string())
        );

        let response = self
            .client
            .post(&self.api_endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                println!("[DEBUG] HTTP request failed: {e}");
                InferenceError::Network(e.to_string())
            })?;

        println!(
            "[DEBUG] Received response with status: {}",
            response.status()
        );

        if !response.status().is_success() {
            return Err(InferenceError::Provider(format!(
                "llama.cpp API returned status: {}",
                response.status()
            )));
        }

        let llamacpp_response: LlamaCppResponse = response
            .json()
            .await
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        println!("[DEBUG] llama.cpp response: {}", llamacpp_response.content);

        Ok(llamacpp_response.content)
    }
}

// Inference manager that handles tool calls and AI responses
pub struct InferenceManager {
    provider: Arc<dyn InferenceProvider>,
    tool_manager: ToolManager,
}

impl InferenceManager {
    pub fn new(provider: Arc<dyn InferenceProvider>, tool_manager: ToolManager) -> Self {
        Self {
            provider,
            tool_manager,
        }
    }

    pub async fn get_ai_response(
        &self,
        prompt: String,
        message_history: &[String],
        user_context: Option<String>,
    ) -> Result<String, InferenceError> {
        println!("[DEBUG] Getting AI response for prompt: {prompt}");

        // Check if this is a tool call
        if let Some(tool_call) = self.tool_manager.detect_tool_call(&prompt) {
            println!("[DEBUG] Detected tool call: {tool_call:?}");

            match self.tool_manager.execute_tool(tool_call).await {
                Ok(result) => {
                    // Pass tool result to LLM for response generation
                    let tool_context = format!("Tool call result: {}", result.content);
                    let enhanced_prompt = format!("The user asked: \"{prompt}\" and I retrieved this information: {tool_context}. Please provide a natural response.");
                    return self
                        .get_llm_response(enhanced_prompt, message_history, user_context)
                        .await;
                }
                Err(e) => {
                    return Err(InferenceError::Tool(e.to_string()));
                }
            }
        }

        self.get_llm_response(prompt, message_history, user_context)
            .await
    }

    async fn get_llm_response(
        &self,
        prompt: String,
        message_history: &[String],
        user_context: Option<String>,
    ) -> Result<String, InferenceError> {
        let user_context_str = user_context.as_deref();

        let response = self
            .provider
            .generate_response(prompt, message_history, user_context_str)
            .await?;

        // Parse the reply from <reply></reply> tags
        let parsed_reply = self.parse_reply(&response);
        println!("[DEBUG] Parsed reply: {parsed_reply}");
        Ok(parsed_reply)
    }

    fn parse_reply(&self, response: &str) -> String {
        // Check if response is too long (indicates multiple responses bug)
        if response.len() > 500 {
            println!(
                "[DEBUG] Response too long ({} chars), likely contains multiple examples",
                response.len()
            );
            // Try to extract just the first meaningful reply
            if let Some(first_reply) = self.extract_first_valid_reply(response) {
                return first_reply;
            }
        }

        // First try to find <reply></reply> tags
        if let Some(start) = response.find("<reply>") {
            if let Some(end) = response.find("</reply>") {
                if end > start {
                    let reply_start = start + "<reply>".len();
                    let reply = response[reply_start..end].trim().to_string();

                    // Validate the reply is reasonable
                    if reply.len() > 200 {
                        println!("[DEBUG] Reply too long, truncating: {reply}");
                        let truncated = reply
                            .split_whitespace()
                            .take(10)
                            .collect::<Vec<_>>()
                            .join(" ");
                        // Ensure we don't exceed limit when adding suffix
                        let max_len = 200 - " FeelsPepoMan".len();
                        if truncated.len() > max_len {
                            return format!("{} FeelsPepoMan", &truncated[..max_len]);
                        } else {
                            return format!("{truncated} FeelsPepoMan");
                        }
                    }

                    return reply;
                }
            }
        }

        // Fallback: extract the last sentence if no tags found
        let lines: Vec<&str> = response.lines().collect();
        if let Some(last_line) = lines.last() {
            let trimmed = last_line.trim();
            if !trimmed.is_empty() && trimmed.ends_with('.') {
                return trimmed.to_string();
            }
        }

        // Final fallback: return a safe default response
        "FeelsPepoMan".to_string()
    }

    fn extract_first_valid_reply(&self, response: &str) -> Option<String> {
        // Look for the first occurrence of <reply></reply> before any "User question:" patterns
        let lines: Vec<&str> = response.lines().collect();
        let mut reply_lines = Vec::new();

        for line in lines {
            if line.contains("User question:") || line.contains("Example:") {
                break; // Stop when we hit example patterns
            }
            reply_lines.push(line);
        }

        let truncated_response = reply_lines.join("\n");

        // Try to parse from the truncated response
        if let Some(start) = truncated_response.find("<reply>") {
            if let Some(end) = truncated_response.find("</reply>") {
                if end > start {
                    let reply_start = start + "<reply>".len();
                    let reply = truncated_response[reply_start..end].trim();
                    if !reply.is_empty() && reply.len() < 100 {
                        return Some(reply.to_string());
                    }
                }
            }
        }

        None
    }
}

// The inference worker function
pub async fn inference_worker(
    mut inference_rx: mpsc::UnboundedReceiver<InferenceRequest>,
    inference_manager: Arc<InferenceManager>,
    message_history: Arc<Mutex<Vec<String>>>,
    available_users: Arc<Mutex<std::collections::HashSet<String>>>,
    mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) {
    println!("[DEBUG] Inference worker started");

    loop {
        tokio::select! {
            request = inference_rx.recv() => {
                match request {
                    Some(request) => {
                        println!(
                            "[DEBUG] Processing inference request from: {}",
                            request.sender
                        );

                        // Get current message history
                        let history = {
                            let history_guard = message_history.lock().await;
                            history_guard.clone()
                        };

                        // Detect users in the prompt and get their info
                        let user_context = get_user_context(&request.prompt, available_users.clone()).await;

                        // Process inference
                        match inference_manager
                            .get_ai_response(request.prompt, &history, user_context)
                            .await
                        {
                            Ok(response) => {
                                println!("[DEBUG] Inference completed, sending response");
                                if let Err(e) = request.response_tx.send(response) {
                                    eprintln!("[DEBUG] Failed to send response: {e}");
                                }
                            }
                            Err(e) => {
                                eprintln!("[DEBUG] Inference failed: {e}");
                                let error_response =
                                    "Sorry, I encountered an error processing your request.".to_string();
                                if let Err(e) = request.response_tx.send(error_response) {
                                    eprintln!("[DEBUG] Failed to send error response: {e}");
                                }
                            }
                        }
                    }
                    None => {
                        println!("[DEBUG] Inference channel closed, stopping worker");
                        break;
                    }
                }
            }
            _ = &mut shutdown_rx => {
                println!("[DEBUG] Received shutdown signal, stopping inference worker");
                break;
            }
        }
    }

    println!("[DEBUG] Inference worker stopped");
}

// Helper function to get user context
async fn get_user_context(
    prompt: &str,
    available_users: Arc<Mutex<std::collections::HashSet<String>>>,
) -> Option<String> {
    use tokio::fs::read_to_string;

    let users = available_users.lock().await;
    let mut detected_users = Vec::new();

    for user in users.iter() {
        if prompt.contains(user) {
            detected_users.push(user.clone());
        }
    }

    if detected_users.is_empty() {
        return None;
    }

    let mut user_context = String::new();
    user_context.push_str("User information:\n");

    for username in &detected_users {
        let user_info = read_to_string(format!("users/{username}"))
            .await
            .unwrap_or_default();
        if !user_info.trim().is_empty() {
            user_context.push_str(&format!("{username}: {user_info}\n"));
        } else {
            user_context.push_str(&format!("{username}: No information available\n"));
        }
    }

    Some(user_context)
}

// Factory function to create providers
pub fn create_provider(
    provider_type: ProviderType,
    client: Client,
    api_host: String,
    model: Option<String>,
) -> Arc<dyn InferenceProvider> {
    match provider_type {
        ProviderType::Ollama => {
            let model = model.unwrap_or_else(|| "granite3.3:2b".to_string());
            Arc::new(OllamaProvider::new(client, api_host, model))
        }
        ProviderType::LlamaCpp => Arc::new(LlamaCppProvider::new(client, api_host)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tokio::sync::Mutex;

    // Mock provider for testing
    struct MockProvider {
        response: Result<String, InferenceError>,
    }

    #[async_trait::async_trait]
    impl InferenceProvider for MockProvider {
        async fn generate_response(
            &self,
            _prompt: String,
            _message_history: &[String],
            _user_context: Option<&str>,
        ) -> Result<String, InferenceError> {
            self.response.clone()
        }
    }

    impl MockProvider {
        fn new_success(response: String) -> Self {
            Self {
                response: Ok(response),
            }
        }

        fn new_failure(error: InferenceError) -> Self {
            Self {
                response: Err(error),
            }
        }
    }

    #[test]
    fn test_format_chat_prompt_with_history() {
        let prompt = "What's the weather?".to_string();
        let history = vec![
            "user1: Hello".to_string(),
            "brok: Hi there!".to_string(),
            "user2: How are you?".to_string(),
        ];

        let result = format_chat_prompt(prompt.clone(), &history, None);

        assert!(result.contains("What's the weather?"));
        assert!(result.contains("user1: Hello"));
        assert!(result.contains("brok: Hi there!"));
        assert!(result.contains("user2: How are you?"));
        assert!(result.contains("<reply></reply>"));
    }

    #[test]
    fn test_format_chat_prompt_empty_history() {
        let prompt = "Test prompt".to_string();
        let history = vec![];

        let result = format_chat_prompt(prompt.clone(), &history, None);

        assert!(result.contains("Test prompt"));
        assert!(result.contains("No previous messages"));
    }

    #[test]
    fn test_format_chat_prompt_with_user_context() {
        let prompt = "Tell me about user1".to_string();
        let history = vec![];
        let user_context = Some("user1: Likes programming and coffee");

        let result = format_chat_prompt(prompt.clone(), &history, user_context);

        assert!(result.contains("Tell me about user1"));
        assert!(result.contains("user1: Likes programming and coffee"));
    }

    #[test]
    fn test_parse_reply_valid_tags() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success("test".to_string()));
        let manager = InferenceManager::new(provider, tool_manager);

        let response = "Some text <reply>Hello world!</reply> more text";
        let result = manager.parse_reply(response);

        assert_eq!(result, "Hello world!");
    }

    #[test]
    fn test_parse_reply_no_tags_fallback() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success("test".to_string()));
        let manager = InferenceManager::new(provider, tool_manager);

        let response = "This is a complete sentence.";
        let result = manager.parse_reply(response);

        assert_eq!(result, "This is a complete sentence.");
    }

    #[test]
    fn test_parse_reply_fallback_to_default() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success("test".to_string()));
        let manager = InferenceManager::new(provider, tool_manager);

        let response = "Invalid response format";
        let result = manager.parse_reply(response);

        assert_eq!(result, "FeelsPepoMan");
    }

    #[test]
    fn test_parse_reply_too_long_response() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success("test".to_string()));
        let manager = InferenceManager::new(provider, tool_manager);

        // Create a response that's way too long (over 500 chars)
        let long_response = "User question: What's the weather? <reply>It's sunny</reply> User question: How are you? <reply>I'm good</reply>".repeat(10);
        let result = manager.parse_reply(&long_response);

        // Should extract first valid reply before patterns
        assert_eq!(result, "It's sunny");
    }

    #[test]
    fn test_parse_reply_truncate_long_valid_reply() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success("test".to_string()));
        let manager = InferenceManager::new(provider, tool_manager);

        let long_reply =
            "This is a very long response that exceeds the 200 character limit ".repeat(5);
        let response = format!("<reply>{long_reply}</reply>");
        let result = manager.parse_reply(&response);

        // Should be truncated and have FeelsPepoMan suffix
        assert!(result.len() <= 200);
        assert!(result.ends_with("FeelsPepoMan"));
    }

    #[test]
    fn test_extract_first_valid_reply_stops_at_example() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success("test".to_string()));
        let manager = InferenceManager::new(provider, tool_manager);

        let response = "<reply>Valid answer</reply>\nUser question: example\n<reply>Should not use this</reply>";
        let result = manager.extract_first_valid_reply(response);

        assert_eq!(result, Some("Valid answer".to_string()));
    }

    #[tokio::test]
    async fn test_get_user_context_with_users() {
        let mut users = HashSet::new();
        users.insert("alice".to_string());
        users.insert("bob".to_string());
        let available_users = Arc::new(Mutex::new(users));

        // Create test user files
        tokio::fs::create_dir_all("users").await.ok();
        tokio::fs::write("users/alice", "Alice is a developer")
            .await
            .ok();
        tokio::fs::write("users/bob", "Bob likes coffee").await.ok();

        let prompt = "Tell me about alice and bob";
        let result = get_user_context(prompt, available_users).await;

        assert!(result.is_some());
        let context = result.unwrap();
        assert!(context.contains("alice: Alice is a developer"));
        assert!(context.contains("bob: Bob likes coffee"));

        // Cleanup
        tokio::fs::remove_file("users/alice").await.ok();
        tokio::fs::remove_file("users/bob").await.ok();
        tokio::fs::remove_dir("users").await.ok();
    }

    #[tokio::test]
    async fn test_get_user_context_no_users_mentioned() {
        let users = HashSet::new();
        let available_users = Arc::new(Mutex::new(users));

        let prompt = "What's the weather like?";
        let result = get_user_context(prompt, available_users).await;

        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_get_user_context_missing_user_files() {
        let mut users = HashSet::new();
        users.insert("nonexistent".to_string());
        let available_users = Arc::new(Mutex::new(users));

        let prompt = "Tell me about nonexistent";
        let result = get_user_context(prompt, available_users).await;

        assert!(result.is_some());
        let context = result.unwrap();
        assert!(context.contains("nonexistent: No information available"));
    }

    #[tokio::test]
    async fn test_inference_manager_successful_response() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success(
            "<reply>Hello world!</reply>".to_string(),
        ));
        let manager = InferenceManager::new(provider, tool_manager);

        let result = manager
            .get_ai_response("Hello".to_string(), &["user1: Hi".to_string()], None)
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello world!");
    }

    #[tokio::test]
    async fn test_inference_manager_network_error() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_failure(InferenceError::Network(
            "Connection failed".to_string(),
        )));
        let manager = InferenceManager::new(provider, tool_manager);

        let result = manager
            .get_ai_response("Hello".to_string(), &[], None)
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), InferenceError::Network(_)));
    }

    #[tokio::test]
    async fn test_inference_manager_with_weather_tool() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success(
            "<reply>The weather is sunny!</reply>".to_string(),
        ));
        let manager = InferenceManager::new(provider, tool_manager);

        // This should trigger weather tool detection but fail due to network
        // The important thing is that it attempts to use the tool
        let result = manager
            .get_ai_response("What's the weather in London?".to_string(), &[], None)
            .await;

        // Should either succeed with tool result or fail with tool error
        // Both are acceptable since we're testing the flow, not the actual API
        assert!(result.is_ok() || matches!(result.unwrap_err(), InferenceError::Tool(_)));
    }

    #[tokio::test]
    async fn test_inference_manager_with_calculator_tool() {
        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success(
            "<reply>The answer is 8!</reply>".to_string(),
        ));
        let manager = InferenceManager::new(provider, tool_manager);

        let result = manager
            .get_ai_response("What is 2 + 6?".to_string(), &[], None)
            .await;

        assert!(result.is_ok());
        // Should contain the LLM response about the calculation
        let response = result.unwrap();
        assert!(response.contains("answer") || response.contains("8"));
    }

    #[test]
    fn test_inference_error_display() {
        let error = InferenceError::Network("Connection timeout".to_string());
        assert_eq!(error.to_string(), "Network error: Connection timeout");

        let error = InferenceError::Parse("Invalid JSON".to_string());
        assert_eq!(error.to_string(), "Parse error: Invalid JSON");

        let error = InferenceError::Provider("Model not found".to_string());
        assert_eq!(error.to_string(), "Provider error: Model not found");

        let error = InferenceError::Tool("Calculator failed".to_string());
        assert_eq!(error.to_string(), "Tool error: Calculator failed");
    }

    #[test]
    fn test_create_provider_ollama() {
        let client = reqwest::Client::new();
        let provider = create_provider(
            ProviderType::Ollama,
            client,
            "localhost:11434".to_string(),
            Some("test-model".to_string()),
        );

        // Should create OllamaProvider successfully
        // We can't test the exact type but we can verify it was created
        let _ = provider; // Just verify it doesn't panic
    }

    #[test]
    fn test_create_provider_llama_cpp() {
        let client = reqwest::Client::new();
        let provider = create_provider(
            ProviderType::LlamaCpp,
            client,
            "localhost:8080".to_string(),
            None,
        );

        // Should create LlamaCppProvider successfully
        let _ = provider; // Just verify it doesn't panic
    }

    #[test]
    fn test_create_provider_ollama_default_model() {
        let client = reqwest::Client::new();
        let provider = create_provider(
            ProviderType::Ollama,
            client,
            "localhost:11434".to_string(),
            None, // No model specified
        );

        // Should create OllamaProvider with default model successfully
        let _ = provider; // Just verify it doesn't panic
    }

    #[tokio::test]
    async fn test_inference_worker_shutdown() {
        let (_inference_tx, inference_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success("test".to_string()));
        let inference_manager = Arc::new(InferenceManager::new(provider, tool_manager));

        let message_history = Arc::new(Mutex::new(Vec::new()));
        let available_users = Arc::new(Mutex::new(HashSet::new()));

        // Start the worker
        let worker_handle = tokio::spawn(async move {
            inference_worker(
                inference_rx,
                inference_manager,
                message_history,
                available_users,
                shutdown_rx,
            )
            .await;
        });

        // Send shutdown signal
        shutdown_tx.send(()).unwrap();

        // Worker should complete quickly
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(100), worker_handle).await;

        assert!(
            result.is_ok(),
            "Worker should shutdown gracefully within timeout"
        );
    }

    #[tokio::test]
    async fn test_inference_worker_request_processing() {
        let (inference_tx, inference_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, _shutdown_rx) = tokio::sync::oneshot::channel();

        let tool_manager = crate::tools::ToolManager::new(reqwest::Client::new());
        let provider = Arc::new(MockProvider::new_success(
            "<reply>Test response</reply>".to_string(),
        ));
        let inference_manager = Arc::new(InferenceManager::new(provider, tool_manager));

        let message_history = Arc::new(Mutex::new(vec!["user1: Hello".to_string()]));
        let available_users = Arc::new(Mutex::new(HashSet::new()));

        // Start the worker (it will be dropped when test ends)
        let _worker_handle = tokio::spawn(async move {
            inference_worker(
                inference_rx,
                inference_manager,
                message_history,
                available_users,
                _shutdown_rx,
            )
            .await;
        });

        // Send a request
        let (response_tx, mut response_rx) = mpsc::unbounded_channel();
        let request = InferenceRequest {
            prompt: "Test prompt".to_string(),
            sender: "testuser".to_string(),
            response_tx,
        };

        inference_tx.send(request).unwrap();

        // Should receive a response
        let response =
            tokio::time::timeout(std::time::Duration::from_millis(100), response_rx.recv()).await;

        assert!(response.is_ok());
        assert_eq!(response.unwrap().unwrap(), "Test response");

        // Cleanup
        shutdown_tx.send(()).ok();
    }
}
