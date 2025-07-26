use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone)]
pub enum ToolError {
    HttpRequest(String),
    JsonParsing(String),
    ProcessExecution(String),
    InvalidInput(String),
    NoResult,
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::HttpRequest(msg) => write!(f, "HTTP request failed: {msg}"),
            ToolError::JsonParsing(msg) => write!(f, "JSON parsing failed: {msg}"),
            ToolError::ProcessExecution(msg) => write!(f, "Process execution failed: {msg}"),
            ToolError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            ToolError::NoResult => write!(f, "No result returned"),
        }
    }
}

impl std::error::Error for ToolError {}

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
    pub content: String,
}

pub struct ToolManager {
    client: Client,
}

impl ToolManager {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    pub async fn calculate(&self, expression: &str) -> Result<ToolResult, ToolError> {
        println!("[DEBUG] Calculating expression: {expression}");

        // Input sanitization - only allow safe mathematical characters
        let allowed_chars = "0123456789+-*/().^ ";
        if !expression.chars().all(|c| allowed_chars.contains(c)) {
            return Err(ToolError::InvalidInput(
                "Only numbers, +, -, *, /, ^, (, ), and spaces are allowed".to_string(),
            ));
        }

        // Additional safety: check for dangerous patterns
        if expression.contains("..") || expression.contains("//") || expression.len() > 200 {
            return Err(ToolError::InvalidInput(
                "Expression contains unsafe patterns or is too long".to_string(),
            ));
        }

        // Use bc for calculation
        let output = Command::new("bc")
            .arg("-l")
            .arg("-q")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| ToolError::ProcessExecution(format!("Failed to start bc: {e}")))?;

        let mut child = output;
        use std::io::Write;
        if let Some(stdin) = child.stdin.as_mut() {
            let _ = stdin.write_all(expression.as_bytes());
            let _ = stdin.write_all(b"\nquit\n");
        }

        let output = child
            .wait_with_output()
            .map_err(|e| ToolError::ProcessExecution(format!("Failed to wait for bc: {e}")))?;

        if output.status.success() {
            let result = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if result.is_empty() {
                Err(ToolError::NoResult)
            } else {
                Ok(ToolResult {
                    content: format!("{expression} = {result}"),
                })
            }
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(ToolError::ProcessExecution(format!(
                "bc calculation error: {error}"
            )))
        }
    }

    pub async fn get_weather(&self, location: &str) -> Result<ToolResult, ToolError> {
        println!("[DEBUG] Getting weather for location: {location}");

        let url = format!("https://wttr.in/{location}?format=j1");
        println!("[DEBUG] Weather API URL: {url}");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::HttpRequest(format!("Request failed: {e}")))?;
        println!("[DEBUG] Weather API response status: {}", response.status());

        if response.status().is_success() {
            let response_text = response
                .text()
                .await
                .map_err(|e| ToolError::HttpRequest(format!("Failed to read response: {e}")))?;
            println!("[DEBUG] Raw weather API response: {response_text}");

            let weather_data: WeatherResponse =
                serde_json::from_str(&response_text).map_err(|e| {
                    ToolError::JsonParsing(format!("Weather data parsing failed: {e}"))
                })?;

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

                Ok(ToolResult { content: result })
            } else {
                Err(ToolError::NoResult)
            }
        } else {
            Err(ToolError::HttpRequest(format!(
                "HTTP {}: Failed to fetch weather data",
                response.status()
            )))
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
                // Take all remaining words for multi-word locations
                prompt[start..]
                    .trim()
                    .split('?')
                    .next()
                    .unwrap_or("London")
                    .trim()
                    .to_string()
            } else if let Some(pos) = prompt_lower.find("weather for ") {
                let start = pos + "weather for ".len();
                // Take all remaining words for multi-word locations
                prompt[start..]
                    .trim()
                    .split('?')
                    .next()
                    .unwrap_or("London")
                    .trim()
                    .to_string()
            } else {
                "London".to_string() // Default location
            };

            // Clean the location string - allow spaces and common location characters
            let clean_location = location
                .chars()
                .filter(|c| {
                    c.is_alphanumeric()
                        || *c == ' '
                        || *c == '-'
                        || *c == '_'
                        || *c == ','
                        || *c == '.'
                })
                .collect::<String>()
                .trim()
                .to_string();

            println!("[DEBUG] Extracted location: '{location}', cleaned: '{clean_location}'");

            Some(ToolCall::Weather {
                location: if clean_location.is_empty() {
                    "London".to_string()
                } else {
                    clean_location
                },
            })
        } else {
            None
        }
    }

    pub async fn execute_tool(&self, tool_call: ToolCall) -> Result<ToolResult, ToolError> {
        match tool_call {
            ToolCall::Weather { location } => self.get_weather(&location).await,
            ToolCall::Calculator { expression } => self.calculate(&expression).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::Client;

    fn create_test_tool_manager() -> ToolManager {
        ToolManager::new(Client::new())
    }

    #[test]
    fn test_detect_tool_call_calculator() {
        let manager = create_test_tool_manager();

        // Test simple math expression
        let result = manager.detect_tool_call("what is 2 + 3?");
        assert!(matches!(result, Some(ToolCall::Calculator { .. })));
        if let Some(ToolCall::Calculator { expression }) = result {
            assert_eq!(expression, "2 + 3");
        }

        // Test calculation keyword
        let result = manager.detect_tool_call("calculate 10 * 5");
        assert!(matches!(result, Some(ToolCall::Calculator { .. })));
        if let Some(ToolCall::Calculator { expression }) = result {
            assert_eq!(expression, "10 * 5");
        }

        // Test direct math expression
        let result = manager.detect_tool_call("25 / 5");
        assert!(matches!(result, Some(ToolCall::Calculator { .. })));
        if let Some(ToolCall::Calculator { expression }) = result {
            assert_eq!(expression, "25 / 5");
        }

        // Test solve keyword
        let result = manager.detect_tool_call("solve 2^3");
        assert!(matches!(result, Some(ToolCall::Calculator { .. })));
        if let Some(ToolCall::Calculator { expression }) = result {
            assert_eq!(expression, "2^3");
        }

        // Test non-math expressions
        let result = manager.detect_tool_call("hello world");
        assert!(result.is_none());

        let result = manager.detect_tool_call("what is your name?");
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_tool_call_weather() {
        let manager = create_test_tool_manager();

        // Test weather in location
        let result = manager.detect_tool_call("what's the weather in London?");
        assert!(matches!(result, Some(ToolCall::Weather { .. })));
        if let Some(ToolCall::Weather { location }) = result {
            assert_eq!(location, "London");
        }

        // Test weather for location
        let result = manager.detect_tool_call("weather for New York");
        assert!(matches!(result, Some(ToolCall::Weather { .. })));
        if let Some(ToolCall::Weather { location }) = result {
            assert_eq!(location, "New York");
        }

        // Test weather with multi-word location
        let result = manager.detect_tool_call("weather in San Francisco please");
        assert!(matches!(result, Some(ToolCall::Weather { .. })));
        if let Some(ToolCall::Weather { location }) = result {
            assert_eq!(location, "San Francisco please");
        }

        // Test default weather location
        let result = manager.detect_tool_call("what's the weather?");
        assert!(matches!(result, Some(ToolCall::Weather { .. })));
        if let Some(ToolCall::Weather { location }) = result {
            assert_eq!(location, "London");
        }

        // Test non-weather request
        let result = manager.detect_tool_call("hello world");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_calculate_valid_expressions() {
        let manager = create_test_tool_manager();

        // Test simple addition
        let result = manager.calculate("2 + 3").await;
        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("2 + 3 = 5"));

        // Test multiplication
        let result = manager.calculate("4 * 5").await;
        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("4 * 5 = 20"));

        // Test division
        let result = manager.calculate("10 / 2").await;
        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("10 / 2 = 5"));
    }

    #[tokio::test]
    async fn test_calculate_invalid_inputs() {
        let manager = create_test_tool_manager();

        // Test invalid characters
        let result = manager.calculate("2 + 3; rm -rf /").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));

        // Test dangerous patterns
        let result = manager.calculate("2 + 3 .. /etc/passwd").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));

        // Test too long expression
        let long_expr = "1+".repeat(150);
        let result = manager.calculate(&long_expr).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));

        // Test expression with letters
        let result = manager.calculate("abc + def").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[test]
    fn test_tool_error_display() {
        let error = ToolError::InvalidInput("test error".to_string());
        assert_eq!(error.to_string(), "Invalid input: test error");

        let error = ToolError::HttpRequest("network error".to_string());
        assert_eq!(error.to_string(), "HTTP request failed: network error");

        let error = ToolError::JsonParsing("parse error".to_string());
        assert_eq!(error.to_string(), "JSON parsing failed: parse error");

        let error = ToolError::ProcessExecution("exec error".to_string());
        assert_eq!(error.to_string(), "Process execution failed: exec error");

        let error = ToolError::NoResult;
        assert_eq!(error.to_string(), "No result returned");
    }

    #[test]
    fn test_location_parsing_edge_cases() {
        let manager = create_test_tool_manager();

        // Test location with punctuation
        let result = manager.detect_tool_call("weather in New York, NY?");
        if let Some(ToolCall::Weather { location }) = result {
            assert_eq!(location, "New York, NY");
        }

        // Test empty location extraction
        let result = manager.detect_tool_call("weather in ");
        if let Some(ToolCall::Weather { location }) = result {
            assert_eq!(location, "London"); // Should default to London
        }

        // Test location with special characters that get filtered
        let result = manager.detect_tool_call("weather in Tokyo@#$%");
        if let Some(ToolCall::Weather { location }) = result {
            assert_eq!(location, "Tokyo"); // Special chars should be filtered
        }
    }

    #[test]
    fn test_math_detection_edge_cases() {
        let manager = create_test_tool_manager();

        // Test expression with too many letters (should not trigger math)
        let result = manager.detect_tool_call("hello world + test");
        assert!(result.is_none());

        // Test expression with minimal letters (should trigger math)
        let result = manager.detect_tool_call("a + b");
        assert!(result.is_none()); // This should not trigger since 'a' and 'b' are not numbers

        // Test complex valid expression
        let result = manager.detect_tool_call("(2 + 3) * 4 ^ 2");
        assert!(matches!(result, Some(ToolCall::Calculator { .. })));
        if let Some(ToolCall::Calculator { expression }) = result {
            assert_eq!(expression, "(2 + 3) * 4 ^ 2");
        }

        // Test question mark removal
        let result = manager.detect_tool_call("what is 5 + 5?");
        if let Some(ToolCall::Calculator { expression }) = result {
            assert_eq!(expression, "5 + 5");
        }
    }
}
