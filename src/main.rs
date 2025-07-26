use clap::Parser;
use reqwest::Client;
use std::collections::HashSet;
use std::fs::{read_dir, read_to_string};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time;
use wsgg::Connection;

mod inference;
mod tools;

use inference::{
    create_provider, inference_worker, InferenceManager, InferenceRequest, ProviderType,
};
use tools::ToolManager;

/// AI Chat Bot
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Location of the file containing the cookie for the bot to use
    #[arg(short, long)]
    cookie: String,

    /// Use the dev environement (chat2.strims.gg)
    #[arg(short, long, default_value_t = false)]
    dev: bool,

    /// API endpoint host and port (e.g., localhost:8080)
    #[arg(long, default_value = "localhost:8080")]
    api_host: String,

    /// AI provider type: ollama or llama-cpp
    #[arg(long, default_value = "ollama")]
    provider: String,

    /// Model name (only used with Ollama provider)
    #[arg(long)]
    model: Option<String>,
}

#[derive(Debug, Clone)]
struct PendingMessage {
    id: u64,
    content: String,
    timestamp: std::time::Instant,
}

#[derive(Debug, Clone)]
struct UserUpdateRequest {
    username: String,
    old_info: String,
    chat_context: String,
}

struct App {
    message_history: Arc<Mutex<Vec<String>>>,
    inference_tx: mpsc::UnboundedSender<InferenceRequest>,
    pending_messages: Arc<Mutex<Vec<PendingMessage>>>,
    next_message_id: Arc<Mutex<u64>>,
    available_users: Arc<Mutex<HashSet<String>>>,
    user_update_tx: mpsc::UnboundedSender<UserUpdateRequest>,
}

impl App {
    fn new(
        inference_tx: mpsc::UnboundedSender<InferenceRequest>,
        user_update_tx: mpsc::UnboundedSender<UserUpdateRequest>,
    ) -> App {
        App {
            message_history: Arc::new(Mutex::new(Vec::new())),
            inference_tx,
            pending_messages: Arc::new(Mutex::new(Vec::new())),
            next_message_id: Arc::new(Mutex::new(1)),
            available_users: Arc::new(Mutex::new(HashSet::new())),
            user_update_tx,
        }
    }

    async fn add_message_to_history(&self, sender: &str, message: &str) {
        let formatted_message = format!("{sender}: {message}");
        let mut history = self.message_history.lock().await;
        history.push(formatted_message);

        // Keep only last 10 messages
        if history.len() > 10 {
            history.remove(0);
        }
    }

    async fn queue_inference(
        &self,
        prompt: String,
        sender: String,
    ) -> mpsc::UnboundedReceiver<String> {
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        let request = InferenceRequest {
            prompt,
            sender,
            response_tx,
        };

        if let Err(e) = self.inference_tx.send(request) {
            eprintln!("[DEBUG] Failed to queue inference: {e}");
        }

        response_rx
    }

    async fn add_pending_message(&self, content: String) -> u64 {
        let mut id_guard = self.next_message_id.lock().await;
        let message_id = *id_guard;
        *id_guard += 1;
        drop(id_guard);

        let pending_message = PendingMessage {
            id: message_id,
            content,
            timestamp: std::time::Instant::now(),
        };

        let mut pending = self.pending_messages.lock().await;
        pending.push(pending_message);
        println!("[DEBUG] Added pending message with ID: {message_id}");
        println!(
            "[DEBUG] Current pending queue ({} messages):",
            pending.len()
        );
        for (index, msg) in pending.iter().enumerate() {
            println!("[DEBUG]   {}: ID {} - {}", index + 1, msg.id, msg.content);
        }
        message_id
    }

    async fn confirm_message_sent(&self, message_id: u64, expected_content: &str) {
        let mut pending = self.pending_messages.lock().await;
        if let Some(index) = pending
            .iter()
            .position(|msg| msg.id == message_id && msg.content == expected_content)
        {
            let removed = pending.remove(index);
            println!("[DEBUG] Confirmed message sent and removed from pending: ID {} with content validation", removed.id);
        } else {
            println!(
                "[DEBUG] Message ID {message_id} not found or content mismatch - not removing from pending"
            );
        }
    }

    async fn get_messages_to_retry(&self) -> Vec<PendingMessage> {
        let pending = self.pending_messages.lock().await;
        let retry_threshold = std::time::Duration::from_secs(5);

        pending
            .iter()
            .filter(|msg| msg.timestamp.elapsed() > retry_threshold)
            .cloned()
            .collect()
    }

    async fn check_for_echo(&self, received_message: &str) -> Option<u64> {
        let pending = self.pending_messages.lock().await;
        pending
            .iter()
            .find(|msg| msg.content == received_message)
            .map(|msg| msg.id)
    }

    async fn load_available_users(&self) -> Result<(), String> {
        match read_dir("users") {
            Ok(entries) => {
                let mut users = self.available_users.lock().await;
                users.clear();
                for entry in entries.flatten() {
                    if let Some(filename) = entry.file_name().to_str() {
                        users.insert(filename.to_string());
                    }
                }
                println!("[DEBUG] Loaded {} users", users.len());
                Ok(())
            }
            Err(e) => {
                eprintln!("[DEBUG] Failed to read users directory: {e}");
                Err(e.to_string())
            }
        }
    }

    async fn detect_users_in_message(&self, message: &str) -> Vec<String> {
        let users = self.available_users.lock().await;
        let mut detected_users = Vec::new();

        for user in users.iter() {
            if message.contains(user) {
                detected_users.push(user.clone());
            }
        }

        detected_users
    }

    async fn read_user_info(&self, username: &str) -> String {
        tokio::fs::read_to_string(format!("users/{username}"))
            .await
            .unwrap_or_default()
    }

    async fn queue_user_update(&self, username: String, old_info: String, chat_context: String) {
        let request = UserUpdateRequest {
            username,
            old_info,
            chat_context,
        };

        if let Err(e) = self.user_update_tx.send(request) {
            eprintln!("[DEBUG] Failed to queue user update: {e}");
        }
    }
}

async fn user_update_worker(
    mut user_update_rx: mpsc::UnboundedReceiver<UserUpdateRequest>,
    inference_manager: Arc<InferenceManager>,
    mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) {
    println!("[DEBUG] User update worker started");

    loop {
        tokio::select! {
            request = user_update_rx.recv() => {
                match request {
                    Some(request) => {
                        println!("[DEBUG] Processing user update for: {}", request.username);

                        let update_prompt = format!(
                            "This is the info for {}. Only add information about {} to this file. Update the user information based on the recent chat context. Return only the updated user information in a concise format.

Old user info for {}: {}

Keep the new user info short. 5 sentences max.

Recent chat context:
{}

The new user information for {} should be 5 sentences at max.

Please provide updated user information for {} only:", 
                            request.username,
                            request.username,
                            request.username,
                            if request.old_info.trim().is_empty() { "No previous information" } else { &request.old_info },
                            request.chat_context,
                            request.username,
                            request.username
                        );

                        match inference_manager
                            .get_ai_response(update_prompt, &[], None)
                            .await
                        {
                            Ok(updated_info) => {
                                // Write the updated info back to the user file using tokio::fs
                                let file_path = format!("users/{}", request.username);
                                if let Err(e) = tokio::fs::write(&file_path, updated_info.trim()).await {
                                    eprintln!(
                                        "[DEBUG] Failed to write user info for {}: {}",
                                        request.username, e
                                    );
                                } else {
                                    println!("[DEBUG] Updated user info for: {}", request.username);
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "[DEBUG] Failed to get updated user info for {}: {}",
                                    request.username, e
                                );
                            }
                        }
                    }
                    None => {
                        println!("[DEBUG] User update channel closed, stopping worker");
                        break;
                    }
                }
            }
            _ = &mut shutdown_rx => {
                println!("[DEBUG] Received shutdown signal, stopping user update worker");
                break;
            }
        }
    }

    println!("[DEBUG] User update worker stopped");
}

#[tokio::main]
async fn main() {
    let bot_account = "brok";

    let args = Args::parse();

    // Create inference channel
    let (inference_tx, inference_rx) = mpsc::unbounded_channel();

    // Create user update channel
    let (user_update_tx, user_update_rx) = mpsc::unbounded_channel();

    // Parse provider type
    let provider_type = match args.provider.to_lowercase().as_str() {
        "ollama" => ProviderType::Ollama,
        "llama-cpp" | "llamacpp" => ProviderType::LlamaCpp,
        _ => {
            eprintln!("[ERROR] Invalid provider type: {}", args.provider);
            eprintln!("[ERROR] Valid options: ollama, llama-cpp");
            std::process::exit(1);
        }
    };

    // Validate model flag usage
    if matches!(provider_type, ProviderType::LlamaCpp) && args.model.is_some() {
        eprintln!("[WARNING] --model flag is ignored for llama-cpp provider");
        eprintln!("[WARNING] llama.cpp server serves a single pre-loaded model");
    }

    println!("[DEBUG] Using provider: {provider_type:?}");

    let app = Arc::new(App::new(inference_tx, user_update_tx));

    // Load available users
    if let Err(e) = app.load_available_users().await {
        eprintln!("[WARNING] Failed to load users: {e}");
    }

    // Create inference provider and manager
    let client = Client::new();
    let tool_manager = ToolManager::new(client.clone());
    let provider = create_provider(provider_type, client, args.api_host.clone(), args.model);
    let inference_manager = InferenceManager::new(provider, tool_manager);
    let inference_manager = Arc::new(inference_manager);

    // Create shutdown channels
    let (inference_shutdown_tx, inference_shutdown_rx) = tokio::sync::oneshot::channel();
    let (user_shutdown_tx, user_shutdown_rx) = tokio::sync::oneshot::channel();

    // Start inference worker
    let message_history = Arc::clone(&app.message_history);
    let available_users = Arc::clone(&app.available_users);
    let inference_manager_clone = Arc::clone(&inference_manager);
    let inference_handle = tokio::spawn(async move {
        inference_worker(
            inference_rx,
            inference_manager_clone,
            message_history,
            available_users,
            inference_shutdown_rx,
        )
        .await;
    });

    // Start user update worker
    let user_inference_manager = Arc::clone(&inference_manager);
    let user_handle = tokio::spawn(async move {
        user_update_worker(user_update_rx, user_inference_manager, user_shutdown_rx).await;
    });

    // Setup signal handling for Linux
    let shutdown_signal = async {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler");
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("Failed to install SIGINT handler");

        tokio::select! {
            _ = sigterm.recv() => println!("[DEBUG] Received SIGTERM"),
            _ = sigint.recv() => println!("[DEBUG] Received SIGINT"),
        }
    };

    println!("[DEBUG] Reading cookie from: {}", args.cookie);
    let cookie: String = read_to_string(args.cookie).unwrap().parse().unwrap();
    println!("[DEBUG] Cookie loaded successfully");

    let mut conn = if args.dev {
        println!("Running in test environement");
        println!("[DEBUG] Creating dev connection");
        Connection::new_dev(cookie.as_str()).unwrap()
    } else {
        println!("Running in production environement");
        println!("[DEBUG] Creating production connection");
        Connection::new(cookie.as_str()).unwrap()
    };
    println!("[DEBUG] Connection established, starting main loop");

    // Store pending responses to handle them asynchronously
    let mut pending_responses: Vec<(mpsc::UnboundedReceiver<String>, String)> = Vec::new();

    // Run main loop with shutdown handling
    tokio::select! {
        _ = shutdown_signal => {
            println!("[DEBUG] Shutdown signal received, stopping workers...");

            // Send shutdown signals to workers
            let _ = inference_shutdown_tx.send(());
            let _ = user_shutdown_tx.send(());

            // Wait for workers to finish
            let _ = tokio::join!(inference_handle, user_handle);

            println!("[DEBUG] All workers stopped, exiting");
            return;
        }
        _ = async {
            loop {
        // Check for completed inferences first
        let mut completed_indices = Vec::new();
        let mut processed_messages = Vec::new(); // Store the original messages that were processed

        for (i, (rx, original_message)) in pending_responses.iter_mut().enumerate() {
            if let Ok(response) = rx.try_recv() {
                println!("[DEBUG] Inference completed, sending response");

                // Wait before sending to avoid rate limiting
                time::sleep(time::Duration::from_millis(500)).await;

                println!("[DEBUG] Sending response to chat: {response}");

                // Add to pending messages before sending
                let message_id = app.add_pending_message(response.clone()).await;

                if let Err(e) = conn.send(&response) {
                    eprintln!("[DEBUG] Failed to send response: {e}");
                } else {
                    println!("[DEBUG] Response sent successfully with ID: {message_id}");
                }

                // Add bot's response to history
                app.add_message_to_history(bot_account, &response).await;

                completed_indices.push(i);

                // Store the original message for user processing
                processed_messages.push(original_message.clone());
            }
        }

        // Remove completed responses (in reverse order to maintain indices)
        for &i in completed_indices.iter().rev() {
            pending_responses.remove(i);
        }

        // Process user updates for completed messages
        for message in processed_messages {
            let detected_users = app.detect_users_in_message(&message).await;
            if !detected_users.is_empty() {
                let history = {
                    let history_guard = app.message_history.lock().await;
                    history_guard.clone()
                };
                let chat_context = history.join("\n");

                for username in detected_users {
                    let old_info = app.read_user_info(&username).await;
                    app.queue_user_update(username, old_info, chat_context.clone())
                        .await;
                }
            }
        }

        // Check for messages that need to be retried
        let messages_to_retry = app.get_messages_to_retry().await;
        for retry_msg in messages_to_retry {
            println!(
                "[DEBUG] Retrying message ID {}: {}",
                retry_msg.id, retry_msg.content
            );

            // Update timestamp before retrying
            app.add_pending_message(retry_msg.content.clone()).await;
            app.confirm_message_sent(retry_msg.id, &retry_msg.content)
                .await; // Remove old entry

            if let Err(e) = conn.send(&retry_msg.content) {
                eprintln!("[DEBUG] Failed to send retry message: {e}");
            }

            // Small delay between retries
            time::sleep(time::Duration::from_millis(100)).await;
        }

        // Handle new messages (non-blocking check)
        println!("[DEBUG] Waiting for message...");
        let msg_result = conn.read_msg();

        match msg_result {
            Ok(msg) => {
                println!("[DEBUG] Received message");

                let data: String = msg.message.to_string();
                println!("[DEBUG] Message content: {data}");
                println!("[DEBUG] Message from user: {}", msg.sender);

                // Check if this message is an echo of our own message
                if msg.sender == bot_account {
                    if let Some(message_id) = app.check_for_echo(&data).await {
                        app.confirm_message_sent(message_id, &data).await;
                    }
                }

                // Add all messages to history for context
                app.add_message_to_history(&msg.sender, &data).await;

                if data.starts_with(&format!("@{bot_account}")) {
                    println!("[DEBUG] Message is directed at bot");
                    let rest = data
                        .strip_prefix(&format!("@{bot_account} "))
                        .unwrap_or(&data)
                        .to_string();
                    println!("[DEBUG] Extracted prompt: {rest}");

                    // Queue inference asynchronously
                    let response_rx = app.queue_inference(rest, msg.sender.clone()).await;
                    pending_responses.push((response_rx, data.clone()));
                    println!("[DEBUG] Inference queued, continuing to process messages");
                } else {
                    println!("[DEBUG] Message not directed at bot, ignoring");
                }
            }
            Err(e) => {
                eprintln!("[DEBUG] Error reading message: {e}");
                // Add small delay when no message to prevent busy looping
                time::sleep(time::Duration::from_millis(10)).await;
            }
        }
    }
        } => {}
    }
}
