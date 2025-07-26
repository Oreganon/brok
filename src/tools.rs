use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Serialize, Deserialize, Debug)]
pub struct WeatherResponse {
    pub current_condition: Vec<CurrentCondition>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CurrentCondition {
    #[serde(rename = "FeelsLikeC")]
    pub feels_like_c: String,
    #[serde(rename = "FeelsLikeF")]
    pub feels_like_f: String,
    pub cloudcover: String,
    pub humidity: String,
    #[serde(rename = "localObsDateTime")]
    pub local_obs_date_time: String,
    pub observation_time: String,
    #[serde(rename = "precipInches")]
    pub precip_inches: String,
    #[serde(rename = "precipMM")]
    pub precip_mm: String,
    pub pressure: String,
    #[serde(rename = "pressureInches")]
    pub pressure_inches: String,
    #[serde(rename = "temp_C")]
    pub temp_c: String,
    #[serde(rename = "temp_F")]
    pub temp_f: String,
    #[serde(rename = "uvIndex")]
    pub uv_index: String,
    pub visibility: String,
    #[serde(rename = "visibilityMiles")]
    pub visibility_miles: String,
    #[serde(rename = "weatherCode")]
    pub weather_code: String,
    #[serde(rename = "weatherDesc")]
    pub weather_desc: Vec<WeatherDesc>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WeatherDesc {
    pub value: String,
}

#[derive(Debug, Clone)]
pub enum ToolCall {
    Weather { location: String },
    Calculator { expression: String },
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub success: bool,
    pub content: String,
}

pub struct ToolManager {
    client: Client,
}

impl ToolManager {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    pub async fn calculate(&self, expression: &str) -> Result<ToolResult, String> {
        println!("[DEBUG] Calculating expression: {expression}");

        // Input sanitization - only allow safe mathematical characters
        let allowed_chars = "0123456789+-*/().^ ";
        if !expression.chars().all(|c| allowed_chars.contains(c)) {
            return Ok(ToolResult {
                success: false,
                content: "Invalid characters in expression. Only numbers, +, -, *, /, ^, (, ), and spaces are allowed.".to_string(),
            });
        }

        // Additional safety: check for dangerous patterns
        if expression.contains("..") || expression.contains("//") || expression.len() > 200 {
            return Ok(ToolResult {
                success: false,
                content: "Expression contains unsafe patterns or is too long.".to_string(),
            });
        }

        // Use bc for calculation
        let output = Command::new("bc")
            .arg("-l")
            .arg("-q")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        match output {
            Ok(mut child) => {
                use std::io::Write;
                if let Some(stdin) = child.stdin.as_mut() {
                    let _ = stdin.write_all(expression.as_bytes());
                    let _ = stdin.write_all(b"\nquit\n");
                }

                match child.wait_with_output() {
                    Ok(output) => {
                        if output.status.success() {
                            let result = String::from_utf8_lossy(&output.stdout).trim().to_string();
                            if result.is_empty() {
                                Ok(ToolResult {
                                    success: false,
                                    content: "No result from calculation".to_string(),
                                })
                            } else {
                                Ok(ToolResult {
                                    success: true,
                                    content: format!("{expression} = {result}"),
                                })
                            }
                        } else {
                            let error = String::from_utf8_lossy(&output.stderr);
                            Ok(ToolResult {
                                success: false,
                                content: format!("Calculation error: {error}"),
                            })
                        }
                    }
                    Err(e) => Ok(ToolResult {
                        success: false,
                        content: format!("Failed to execute calculation: {e}"),
                    }),
                }
            }
            Err(e) => Ok(ToolResult {
                success: false,
                content: format!("Failed to start calculator: {e}"),
            }),
        }
    }

    pub async fn get_weather(&self, location: &str) -> Result<ToolResult, String> {
        println!("[DEBUG] Getting weather for location: {location}");

        let url = format!("https://wttr.in/{location}?format=j1");
        println!("[DEBUG] Weather API URL: {url}");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        println!("[DEBUG] Weather API response status: {}", response.status());

        if response.status().is_success() {
            let response_text = response.text().await.map_err(|e| e.to_string())?;
            println!("[DEBUG] Raw weather API response: {response_text}");

            let weather_data: WeatherResponse =
                serde_json::from_str(&response_text).map_err(|e| e.to_string())?;

            if let Some(current) = weather_data.current_condition.first() {
                let weather_desc = current
                    .weather_desc
                    .first()
                    .map(|desc| desc.value.clone())
                    .unwrap_or("Unknown".to_string());

                let result = format!(
                    "Current weather in {}: {} with temperature {}°C (observed at {})",
                    location, weather_desc, current.temp_c, current.observation_time
                );

                Ok(ToolResult {
                    success: true,
                    content: result,
                })
            } else {
                Ok(ToolResult {
                    success: false,
                    content: "No weather data available".to_string(),
                })
            }
        } else {
            Ok(ToolResult {
                success: false,
                content: "Failed to fetch weather data".to_string(),
            })
        }
    }

    pub fn detect_tool_call(&self, prompt: &str) -> Option<ToolCall> {
        let prompt_lower = prompt.to_lowercase();

        // Check for calculator requests - only trigger on clear math expressions
        let has_math_operators = prompt_lower.contains('+')
            || prompt_lower.contains('-')
            || prompt_lower.contains('*')
            || prompt_lower.contains('/')
            || prompt_lower.contains('^')
            || prompt_lower.contains('(')
            || prompt_lower.contains(')');
        let has_numbers = prompt_lower.matches(char::is_numeric).count() > 0;
        let is_math_question = (prompt_lower.starts_with("what is ")
            || prompt_lower.starts_with("what's ")
            || prompt_lower.starts_with("calculate ")
            || prompt_lower.starts_with("solve "))
            && has_math_operators
            && has_numbers;
        let is_direct_math = has_math_operators
            && has_numbers
            && prompt_lower.chars().filter(|c| c.is_alphabetic()).count() < 5; // Very few letters

        if is_math_question || is_direct_math {
            // Extract mathematical expression
            let expression = if let Some(pos) = prompt_lower.find("calculate ") {
                prompt[pos + "calculate ".len()..].trim().to_string()
            } else if let Some(pos) = prompt_lower.find("what is ") {
                prompt[pos + "what is ".len()..].trim().to_string()
            } else if let Some(pos) = prompt_lower.find("what's ") {
                prompt[pos + "what's ".len()..].trim().to_string()
            } else if let Some(pos) = prompt_lower.find("solve ") {
                prompt[pos + "solve ".len()..].trim().to_string()
            } else {
                // Try to extract the mathematical part
                prompt.trim().to_string()
            };

            // Clean up common question endings
            let clean_expression = expression
                .trim_end_matches('?')
                .trim_end_matches('.')
                .trim()
                .to_string();

            println!("[DEBUG] Extracted expression: '{clean_expression}'");

            Some(ToolCall::Calculator {
                expression: clean_expression,
            })
        }
        // Simple pattern matching for weather requests
        else if prompt_lower.contains("weather") {
            // Extract location - look for "weather in X" or "weather for X"
            let location = if let Some(pos) = prompt_lower.find("weather in ") {
                let start = pos + "weather in ".len();
                prompt[start..]
                    .split_whitespace()
                    .next()
                    .unwrap_or("London")
                    .to_string()
            } else if let Some(pos) = prompt_lower.find("weather for ") {
                let start = pos + "weather for ".len();
                prompt[start..]
                    .split_whitespace()
                    .next()
                    .unwrap_or("London")
                    .to_string()
            } else {
                "London".to_string() // Default location
            };

            // Clean the location string - remove any special characters that might cause URL issues
            let clean_location = location
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect::<String>();

            println!("[DEBUG] Extracted location: '{location}', cleaned: '{clean_location}'");

            Some(ToolCall::Weather {
                location: clean_location,
            })
        } else {
            None
        }
    }

    pub async fn execute_tool(&self, tool_call: ToolCall) -> Result<ToolResult, String> {
        match tool_call {
            ToolCall::Weather { location } => self.get_weather(&location).await,
            ToolCall::Calculator { expression } => self.calculate(&expression).await,
        }
    }
}
