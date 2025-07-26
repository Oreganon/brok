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
