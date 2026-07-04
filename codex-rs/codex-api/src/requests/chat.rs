use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::headers::build_session_headers;
use crate::requests::headers::insert_header;
use crate::requests::headers::subagent_header;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionSource;
use http::HeaderMap;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;

/// Assembled request body plus headers for Chat Completions streaming calls.
pub struct ChatRequest {
    pub body: Value,
    pub headers: HeaderMap,
}

pub struct ChatRequestBuilder<'a> {
    model: &'a str,
    instructions: &'a str,
    input: &'a [ResponseItem],
    tools: &'a [Value],
    session_id: Option<String>,
    thread_id: Option<String>,
    session_source: Option<SessionSource>,
}

impl<'a> ChatRequestBuilder<'a> {
    pub fn new(
        model: &'a str,
        instructions: &'a str,
        input: &'a [ResponseItem],
        tools: &'a [Value],
    ) -> Self {
        Self {
            model,
            instructions,
            input,
            tools,
            session_id: None,
            thread_id: None,
            session_source: None,
        }
    }

    pub fn session_id(mut self, id: Option<String>) -> Self {
        self.session_id = id;
        self
    }

    pub fn thread_id(mut self, id: Option<String>) -> Self {
        self.thread_id = id;
        self
    }

    pub fn session_source(mut self, source: Option<SessionSource>) -> Self {
        self.session_source = source;
        self
    }

    pub fn build(self, _provider: &Provider) -> Result<ChatRequest, ApiError> {
        let mut messages = Vec::<Value>::new();
        messages.push(json!({"role": "system", "content": self.instructions}));

        let input = self.input;
        let mut reasoning_by_anchor_index: HashMap<usize, String> = HashMap::new();
        let mut last_emitted_role: Option<&str> = None;
        for item in input {
            match item {
                ResponseItem::Message { role, .. } => last_emitted_role = Some(role.as_str()),
                ResponseItem::FunctionCall { .. } | ResponseItem::LocalShellCall { .. } => {
                    last_emitted_role = Some("assistant")
                }
                ResponseItem::FunctionCallOutput { .. } => last_emitted_role = Some("tool"),
                ResponseItem::Reasoning { .. }
                | ResponseItem::Other
                | ResponseItem::AdditionalTools { .. }
                | ResponseItem::AgentMessage { .. }
                | ResponseItem::ToolSearchCall { .. }
                | ResponseItem::ToolSearchOutput { .. }
                | ResponseItem::ImageGenerationCall { .. }
                | ResponseItem::CompactionTrigger { .. }
                | ResponseItem::ContextCompaction { .. } => {}
                ResponseItem::CustomToolCall { .. } => {}
                ResponseItem::CustomToolCallOutput { .. } => {}
                ResponseItem::WebSearchCall { .. } => {}
                ResponseItem::Compaction { .. } => {}
            }
        }

        let mut last_user_index: Option<usize> = None;
        for (idx, item) in input.iter().enumerate() {
            if let ResponseItem::Message { role, .. } = item
                && role == "user"
            {
                last_user_index = Some(idx);
            }
        }

        if !matches!(last_emitted_role, Some("user")) {
            for (idx, item) in input.iter().enumerate() {
                if let Some(u_idx) = last_user_index
                    && idx <= u_idx
                {
                    continue;
                }

                if let ResponseItem::Reasoning {
                    content: Some(items),
                    ..
                } = item
                {
                    let mut text = String::new();
                    for entry in items {
                        match entry {
                            ReasoningItemContent::ReasoningText { text: segment }
                            | ReasoningItemContent::Text { text: segment } => {
                                text.push_str(segment)
                            }
                        }
                    }
                    if text.trim().is_empty() {
                        continue;
                    }

                    let mut attached = false;
                    if idx > 0
                        && let ResponseItem::Message { role, .. } = &input[idx - 1]
                        && role == "assistant"
                    {
                        reasoning_by_anchor_index
                            .entry(idx - 1)
                            .and_modify(|v| v.push_str(&text))
                            .or_insert(text.clone());
                        attached = true;
                    }

                    if !attached && idx + 1 < input.len() {
                        match &input[idx + 1] {
                            ResponseItem::FunctionCall { .. }
                            | ResponseItem::LocalShellCall { .. } => {
                                reasoning_by_anchor_index
                                    .entry(idx + 1)
                                    .and_modify(|v| v.push_str(&text))
                                    .or_insert(text.clone());
                            }
                            ResponseItem::Message { role, .. } if role == "assistant" => {
                                reasoning_by_anchor_index
                                    .entry(idx + 1)
                                    .and_modify(|v| v.push_str(&text))
                                    .or_insert(text.clone());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let mut last_assistant_text: Option<String> = None;

        for (idx, item) in input.iter().enumerate() {
            match item {
                ResponseItem::Message { role, content, .. } => {
                    let mut text = String::new();
                    let mut items: Vec<Value> = Vec::new();
                    let mut saw_image = false;

                    for c in content {
                        match c {
                            ContentItem::InputText { text: t }
                            | ContentItem::OutputText { text: t } => {
                                text.push_str(t);
                                items.push(json!({"type":"text","text": t}));
                            }
                            ContentItem::InputImage { image_url, .. } => {
                                saw_image = true;
                                items.push(
                                    json!({"type":"image_url","image_url": {"url": image_url}}),
                                );
                            }
                        }
                    }

                    if role == "assistant" {
                        if let Some(prev) = &last_assistant_text
                            && prev == &text
                        {
                            continue;
                        }
                        last_assistant_text = Some(text.clone());
                    }

                    let content_value = if role == "assistant" {
                        json!(text)
                    } else if saw_image {
                        json!(items)
                    } else {
                        json!(text)
                    };

                    let mut msg = json!({"role": role, "content": content_value});
                    if role == "assistant"
                        && let Some(reasoning) = reasoning_by_anchor_index.get(&idx)
                        && let Some(obj) = msg.as_object_mut()
                    {
                        obj.insert("reasoning".to_string(), json!(reasoning));
                    }
                    messages.push(msg);
                }
                ResponseItem::FunctionCall {
                    name,
                    arguments,
                    call_id,
                    ..
                } => {
                    let reasoning = reasoning_by_anchor_index.get(&idx).map(String::as_str);
                    let tool_call = json!({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
                    });
                    push_tool_call_message(&mut messages, tool_call, reasoning);
                }
                ResponseItem::LocalShellCall {
                    call_id, action, ..
                } => {
                    let reasoning = reasoning_by_anchor_index.get(&idx).map(String::as_str);
                    let tool_call = json!({
                        "id": call_id.clone().unwrap_or_default(),
                        "type": "local_shell_call",
                        "action": action,
                    });
                    push_tool_call_message(&mut messages, tool_call, reasoning);
                }
                ResponseItem::FunctionCallOutput {
                    call_id, output, ..
                } => {
                    let content_value =
                        match &output.body {
                            FunctionCallOutputBody::ContentItems(items) => {
                                let mapped: Vec<Value> = items
                                .iter()
                                .map(|it| match it {
                                    FunctionCallOutputContentItem::InputText { text } => {
                                        json!({"type":"text","text": text})
                                    }
                                    FunctionCallOutputContentItem::InputImage { image_url, .. } => {
                                        json!({"type":"image_url","image_url": {"url": image_url}})
                                    }
                                    FunctionCallOutputContentItem::EncryptedContent { .. } => {
                                        json!({"type":"text","text": "[encrypted]"})
                                    }
                                })
                                .collect();
                                json!(mapped)
                            }
                            FunctionCallOutputBody::Text(text) => json!(text),
                        };

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content_value,
                    }));
                }
                ResponseItem::CustomToolCall {
                    call_id,
                    name,
                    input,
                    ..
                } => {
                    let tool_call = json!({
                        "id": call_id,
                        "type": "custom",
                        "custom": {
                            "name": name,
                            "input": input,
                        }
                    });
                    let reasoning = reasoning_by_anchor_index.get(&idx).map(String::as_str);
                    push_tool_call_message(&mut messages, tool_call, reasoning);
                }
                ResponseItem::CustomToolCallOutput {
                    call_id, output, ..
                } => {
                    let content_value =
                        match &output.body {
                            FunctionCallOutputBody::Text(text) => json!(text),
                            FunctionCallOutputBody::ContentItems(items) => {
                                let mapped: Vec<Value> = items
                                .iter()
                                .map(|it| match it {
                                    FunctionCallOutputContentItem::InputText { text } => {
                                        json!({"type":"text","text": text})
                                    }
                                    FunctionCallOutputContentItem::InputImage { image_url, .. } => {
                                        json!({"type":"image_url","image_url": {"url": image_url}})
                                    }
                                    FunctionCallOutputContentItem::EncryptedContent { .. } => {
                                        json!({"type":"text","text": "[encrypted]"})
                                    }
                                })
                                .collect();
                                json!(mapped)
                            }
                        };
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content_value,
                    }));
                }
                // Skip Responses-API-specific items that have no Chat Completions equivalent
                ResponseItem::AdditionalTools { .. }
                | ResponseItem::AgentMessage { .. }
                | ResponseItem::ToolSearchCall { .. }
                | ResponseItem::ToolSearchOutput { .. }
                | ResponseItem::Reasoning { .. }
                | ResponseItem::WebSearchCall { .. }
                | ResponseItem::ImageGenerationCall { .. }
                | ResponseItem::Compaction { .. }
                | ResponseItem::CompactionTrigger { .. }
                | ResponseItem::ContextCompaction { .. }
                | ResponseItem::Other => {
                    continue;
                }
            }
        }

        let payload = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "tools": self.tools,
        });

        let mut headers = build_session_headers(self.session_id, self.thread_id);
        if let Some(subagent) = subagent_header(&self.session_source) {
            insert_header(&mut headers, "x-openai-subagent", &subagent);
        }

        Ok(ChatRequest {
            body: payload,
            headers,
        })
    }
}

fn push_tool_call_message(messages: &mut Vec<Value>, tool_call: Value, reasoning: Option<&str>) {
    // Chat Completions requires that tool calls are grouped into a single assistant message
    // (with `tool_calls: [...]`) followed by tool role responses.
    if let Some(Value::Object(obj)) = messages.last_mut()
        && obj.get("role").and_then(Value::as_str) == Some("assistant")
        && obj.get("content").is_some_and(Value::is_null)
        && let Some(tool_calls) = obj.get_mut("tool_calls").and_then(Value::as_array_mut)
    {
        tool_calls.push(tool_call);
        if let Some(reasoning) = reasoning {
            if let Some(Value::String(existing)) = obj.get_mut("reasoning") {
                if !existing.is_empty() {
                    existing.push('\n');
                }
                existing.push_str(reasoning);
            } else {
                obj.insert(
                    "reasoning".to_string(),
                    Value::String(reasoning.to_string()),
                );
            }
        }
        return;
    }

    let mut msg = json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [tool_call],
    });
    if let Some(reasoning) = reasoning
        && let Some(obj) = msg.as_object_mut()
    {
        obj.insert("reasoning".to_string(), json!(reasoning));
    }
    messages.push(msg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Provider;
    use codex_protocol::models::FunctionCallOutputContentItem;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::models::LocalShellAction;
    use codex_protocol::models::LocalShellExecAction;
    use codex_protocol::models::LocalShellStatus;
    use http::HeaderMap;
    use pretty_assertions::assert_eq;
    use std::time::Duration;

    fn provider() -> Provider {
        Provider {
            name: "test".to_string(),
            base_url: "https://example.com/v1".to_string(),
            query_params: None,
            headers: HeaderMap::new(),
            retry: crate::provider::RetryConfig {
                max_attempts: 1,
                base_delay: Duration::from_millis(1),
                retry_429: false,
                retry_5xx: true,
                retry_transport: true,
            },
            stream_idle_timeout: Duration::from_secs(1),
        }
    }

    fn build(input: &[ResponseItem]) -> Value {
        let builder = ChatRequestBuilder::new("m", "sys-instructions", input, &[]);
        builder
            .build(&provider())
            .expect("build should succeed")
            .body
    }

    fn messages_from(body: &Value) -> &Vec<Value> {
        body["messages"]
            .as_array()
            .unwrap_or_else(|| panic!("messages array missing: {body}"))
    }

    fn user_message(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn assistant_message(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: text.to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn reasoning_item(text: &str) -> ResponseItem {
        ResponseItem::Reasoning {
            id: None,
            summary: Vec::new(),
            content: Some(vec![ReasoningItemContent::ReasoningText {
                text: text.to_string(),
            }]),
            encrypted_content: None,
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn function_call(name: &str, call_id: &str, arguments: &str) -> ResponseItem {
        ResponseItem::FunctionCall {
            id: None,
            name: name.to_string(),
            namespace: None,
            arguments: arguments.to_string(),
            call_id: call_id.to_string(),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn function_call_output(call_id: &str, text: &str) -> ResponseItem {
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: call_id.to_string(),
            output: FunctionCallOutputPayload::from_content_items(vec![
                FunctionCallOutputContentItem::InputText {
                    text: text.to_string(),
                },
            ]),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn local_shell_call() -> ResponseItem {
        ResponseItem::LocalShellCall {
            id: Some("id1".to_string()),
            call_id: Some("shell-c1".to_string()),
            status: LocalShellStatus::InProgress,
            action: LocalShellAction::Exec(LocalShellExecAction {
                command: vec!["echo".to_string()],
                timeout_ms: Some(1_000),
                working_directory: None,
                env: None,
                user: None,
            }),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    #[test]
    fn prepends_system_message_with_instructions() {
        let body = build(&[user_message("hi")]);
        let messages = messages_from(&body);
        assert_eq!(messages[0]["role"], Value::String("system".into()));
        assert_eq!(
            messages[0]["content"],
            Value::String("sys-instructions".into())
        );
        assert_eq!(messages[1]["role"], Value::String("user".into()));
    }

    #[test]
    fn maps_user_and_assistant_messages() {
        let body = build(&[user_message("u1"), assistant_message("a1")]);
        let messages = messages_from(&body);
        // system, user, assistant
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], Value::String("user".into()));
        assert_eq!(messages[1]["content"], Value::String("u1".into()));
        assert_eq!(messages[2]["role"], Value::String("assistant".into()));
        assert_eq!(messages[2]["content"], Value::String("a1".into()));
    }

    #[test]
    fn omits_reasoning_when_none_present() {
        let body = build(&[user_message("u1"), assistant_message("a1")]);
        let messages = messages_from(&body);
        let assistant = messages
            .iter()
            .find(|m| m["role"] == "assistant")
            .expect("assistant message present");
        assert!(assistant.get("reasoning").is_none());
    }

    #[test]
    fn attaches_reasoning_to_previous_assistant() {
        let body = build(&[
            user_message("u1"),
            assistant_message("a1"),
            reasoning_item("rA"),
        ]);
        let messages = messages_from(&body);
        let assistant = messages
            .iter()
            .find(|m| m["role"] == "assistant")
            .expect("assistant message present");
        assert_eq!(assistant["content"], Value::String("a1".into()));
        assert_eq!(assistant["reasoning"], Value::String("rA".into()));
    }

    #[test]
    fn attaches_reasoning_to_function_call_anchor() {
        let body = build(&[
            user_message("u1"),
            reasoning_item("rFunc"),
            function_call("f", "c1", "{}"),
        ]);
        let messages = messages_from(&body);
        let assistant = messages
            .iter()
            .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
            .expect("tool-call assistant message present");
        assert_eq!(assistant["reasoning"], Value::String("rFunc".into()));
        let tool_calls = assistant["tool_calls"]
            .as_array()
            .expect("tool_calls array");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["type"], Value::String("function".into()));
        assert_eq!(tool_calls[0]["function"]["name"], Value::String("f".into()));
    }

    #[test]
    fn attaches_reasoning_to_local_shell_call() {
        let body = build(&[
            user_message("u1"),
            reasoning_item("rShell"),
            local_shell_call(),
        ]);
        let messages = messages_from(&body);
        let assistant = messages
            .iter()
            .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
            .expect("tool-call assistant message present");
        assert_eq!(assistant["reasoning"], Value::String("rShell".into()));
        assert_eq!(
            assistant["tool_calls"][0]["type"],
            Value::String("local_shell_call".into())
        );
    }

    #[test]
    fn drops_reasoning_when_last_role_is_user() {
        let body = build(&[
            assistant_message("aPrev"),
            reasoning_item("rHist"),
            user_message("uNew"),
        ]);
        let messages = messages_from(&body);
        assert!(messages.iter().all(|msg| msg.get("reasoning").is_none()));
    }

    #[test]
    fn ignores_reasoning_before_last_user() {
        let body = build(&[
            user_message("u1"),
            assistant_message("a1"),
            user_message("u2"),
            reasoning_item("rAfterU1"),
        ]);
        let messages = messages_from(&body);
        assert!(messages.iter().all(|msg| msg.get("reasoning").is_none()));
    }

    #[test]
    fn skips_empty_reasoning_segments() {
        let body = build(&[
            user_message("u1"),
            assistant_message("a1"),
            reasoning_item(""),
            reasoning_item("   "),
        ]);
        let messages = messages_from(&body);
        let assistant = messages
            .iter()
            .find(|m| m["role"] == "assistant")
            .expect("assistant message present");
        assert!(assistant.get("reasoning").is_none());
    }

    #[test]
    fn suppresses_duplicate_assistant_messages() {
        let body = build(&[assistant_message("dup"), assistant_message("dup")]);
        let messages = messages_from(&body);
        let assistant_count = messages
            .iter()
            .filter(|m| m["role"] == "assistant" && m.get("tool_calls").is_none())
            .count();
        assert_eq!(assistant_count, 1);
    }

    #[test]
    fn groups_multiple_function_calls_into_one_assistant_message() {
        let body = build(&[
            user_message("u1"),
            function_call("f1", "c1", "{}"),
            function_call("f2", "c2", "{}"),
        ]);
        let messages = messages_from(&body);
        let tool_call_messages: Vec<_> = messages
            .iter()
            .filter(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
            .collect();
        assert_eq!(tool_call_messages.len(), 1);
        let tool_calls = tool_call_messages[0]["tool_calls"]
            .as_array()
            .expect("tool_calls array");
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(
            tool_calls[0]["function"]["name"],
            Value::String("f1".into())
        );
        assert_eq!(
            tool_calls[1]["function"]["name"],
            Value::String("f2".into())
        );
    }

    #[test]
    fn emits_tool_role_for_function_call_output() {
        let body = build(&[
            user_message("u1"),
            function_call("f", "c1", "{}"),
            function_call_output("c1", "result-text"),
        ]);
        let messages = messages_from(&body);
        let tool_messages: Vec<_> = messages.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(tool_messages.len(), 1);
        assert_eq!(tool_messages[0]["tool_call_id"], Value::String("c1".into()));
        let content = tool_messages[0]["content"]
            .as_array()
            .expect("content as array (content_items path)");
        assert_eq!(content[0]["type"], Value::String("text".into()));
        assert_eq!(content[0]["text"], Value::String("result-text".into()));
    }

    #[test]
    fn sets_stream_and_model_in_payload() {
        let body = build(&[user_message("hi")]);
        assert_eq!(body["model"], Value::String("m".into()));
        assert_eq!(body["stream"], Value::Bool(true));
        assert!(body["tools"].as_array().is_some_and(std::vec::Vec::is_empty));
    }

    #[test]
    fn includes_session_and_thread_headers() {
        let input = [user_message("hi")];
        let builder = ChatRequestBuilder::new("m", "sys", &input, &[])
            .session_id(Some("sess-1".to_string()))
            .thread_id(Some("thr-1".to_string()));
        let request = builder.build(&provider()).expect("build succeeds");
        assert!(
            request
                .headers
                .get("session-id")
                .is_some_and(|v| v == "sess-1")
        );
        assert!(
            request
                .headers
                .get("thread-id")
                .is_some_and(|v| v == "thr-1")
        );
    }

    #[test]
    fn maps_image_input_to_content_array() {
        let msg = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![
                ContentItem::InputText {
                    text: "look at this".to_string(),
                },
                ContentItem::InputImage {
                    image_url: "https://img.example.com/x.png".to_string(),
                    detail: None,
                },
            ],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        };
        let body = build(&[msg]);
        let messages = messages_from(&body);
        let user = messages
            .iter()
            .find(|m| m["role"] == "user")
            .expect("user message present");
        let content = user["content"].as_array().expect("content as array");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], Value::String("text".into()));
        assert_eq!(content[1]["type"], Value::String("image_url".into()));
        assert_eq!(
            content[1]["image_url"]["url"],
            Value::String("https://img.example.com/x.png".into())
        );
    }

    #[test]
    fn skips_responses_only_item_variants() {
        let input = vec![
            user_message("u1"),
            ResponseItem::WebSearchCall {
                id: None,
                status: None,
                action: None,
                internal_chat_message_metadata_passthrough: None,
            },
            assistant_message("a1"),
        ];
        let body = build(&input);
        let messages = messages_from(&body);
        // system, user, assistant — no web_search entry
        assert_eq!(messages.len(), 3);
        assert!(messages.iter().all(|m| m.get("type").is_none()));
    }
}
