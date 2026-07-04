#![allow(clippy::expect_used)]

use std::sync::Arc;

use codex_core::ModelClient;
use codex_core::Prompt;
use codex_core::ResponseEvent;
use codex_login::CodexAuth;
use codex_login::auth::AgentIdentityAuthPolicy;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::WireApi;
use codex_otel::SessionTelemetry;
use codex_protocol::ThreadId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::LocalShellExecAction;
use codex_protocol::models::LocalShellStatus;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionSource;
use core_test_support::TestCodexResponsesRequestKind;
use core_test_support::load_default_config_for_test;
use core_test_support::responses_metadata as test_responses_metadata;
use core_test_support::skip_if_no_network;
use futures::StreamExt;
use serde_json::Value;
use tempfile::TempDir;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

fn mock_chat_provider(server_uri: String) -> ModelProviderInfo {
    ModelProviderInfo {
        name: "mock".into(),
        base_url: Some(format!("{server_uri}/v1")),
        env_key: None,
        env_key_instructions: None,
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: WireApi::Chat,
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        request_max_retries: Some(0),
        stream_max_retries: Some(0),
        stream_idle_timeout_ms: Some(5_000),
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
    }
}

async fn run_request(input: Vec<ResponseItem>) -> Value {
    let server = MockServer::start().await;

    let template = ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(
            "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n\n",
            "text/event-stream",
        );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(template)
        .expect(1)
        .mount(&server)
        .await;

    let provider = mock_chat_provider(server.uri());

    let codex_home = TempDir::new().expect("failed to create TempDir");
    let mut config = load_default_config_for_test(&codex_home).await;
    config.model_provider_id = provider.name.clone();
    config.model_provider = provider.clone();
    config.show_raw_agent_reasoning = true;
    let effort = config.model_reasoning_effort.clone();
    let summary = config.model_reasoning_summary;
    let model = codex_core::test_support::get_model_offline(config.model.as_deref());
    config.model = Some(model.clone());
    let config = Arc::new(config);

    let thread_id = ThreadId::new();
    let auth_manager =
        codex_core::test_support::auth_manager_from_auth(CodexAuth::from_api_key("Test API Key"));
    let auth_mode = codex_otel::TelemetryAuthMode::ApiKey;
    let session_source = SessionSource::Exec;
    let model_info =
        codex_core::test_support::construct_model_info_offline(model.as_str(), &config);
    let session_telemetry = SessionTelemetry::new(
        thread_id,
        model.as_str(),
        model_info.slug.as_str(),
        /*account_id*/ None,
        Some("test@test.com".to_string()),
        Some(auth_mode),
        "test_originator".to_string(),
        /*log_user_prompts*/ false,
        "test".to_string(),
        session_source.clone(),
    );

    let client = ModelClient::new(
        Some(auth_manager),
        AgentIdentityAuthPolicy::JwtOnly,
        thread_id,
        provider.clone(),
        session_source.clone(),
        "test_originator".to_string(),
        config.model_verbosity,
        /*enable_request_compression*/ false,
        /*include_timing_metrics*/ false,
        /*beta_features_header*/ None,
        /*item_ids_enabled*/ false,
        /*attestation_provider*/ None,
    );
    let responses_metadata = test_responses_metadata(
        "11111111-1111-4111-8111-111111111111",
        &thread_id.to_string(),
        &thread_id.to_string(),
        /*turn_id*/ None,
        format!("{thread_id}:0"),
        &session_source,
        /*parent_thread_id*/ None,
        TestCodexResponsesRequestKind::Turn,
    );
    let mut client_session = client.new_session();

    let mut prompt = Prompt::default();
    prompt.input = input;

    let mut stream = client_session
        .stream(
            &prompt,
            &model_info,
            &session_telemetry,
            effort,
            summary.unwrap_or(model_info.default_reasoning_summary),
            /*service_tier*/ None,
            &responses_metadata,
            &codex_rollout_trace::InferenceTraceContext::disabled(),
        )
        .await
        .expect("stream chat failed");
    while let Some(event) = stream.next().await {
        if let Ok(ResponseEvent::Completed { .. }) = event {
            break;
        }
    }

    let all_requests = server.received_requests().await.expect("received requests");
    let requests: Vec<_> = all_requests
        .iter()
        .filter(|req| req.method == "POST" && req.url.path().ends_with("/chat/completions"))
        .collect();
    let request = requests
        .first()
        .unwrap_or_else(|| panic!("expected POST request to /chat/completions"));
    request.body_json().expect("invalid json body")
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

fn function_call() -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: "f".to_string(),
        namespace: None,
        arguments: "{}".to_string(),
        call_id: "c1".to_string(),
        internal_chat_message_metadata_passthrough: None,
    }
}

fn local_shell_call() -> ResponseItem {
    ResponseItem::LocalShellCall {
        id: Some("id1".to_string()),
        call_id: None,
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

fn messages_from(body: &Value) -> Vec<Value> {
    body["messages"]
        .as_array()
        .unwrap_or_else(|| panic!("messages array missing"))
        .clone()
}

fn first_assistant(messages: &[Value]) -> &Value {
    messages
        .iter()
        .find(|msg| msg["role"] == "assistant")
        .unwrap_or_else(|| panic!("assistant message not present"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn omits_reasoning_when_none_present() {
    skip_if_no_network!();

    let body = run_request(vec![user_message("u1"), assistant_message("a1")]).await;
    let messages = messages_from(&body);
    let assistant = first_assistant(&messages);

    assert_eq!(assistant["content"], Value::String("a1".into()));
    assert!(assistant.get("reasoning").is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn attaches_reasoning_to_previous_assistant() {
    skip_if_no_network!();

    let body = run_request(vec![
        user_message("u1"),
        assistant_message("a1"),
        reasoning_item("rA"),
    ])
    .await;
    let messages = messages_from(&body);
    let assistant = first_assistant(&messages);

    assert_eq!(assistant["content"], Value::String("a1".into()));
    assert_eq!(assistant["reasoning"], Value::String("rA".into()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn attaches_reasoning_to_function_call_anchor() {
    skip_if_no_network!();

    let body = run_request(vec![
        user_message("u1"),
        reasoning_item("rFunc"),
        function_call(),
    ])
    .await;
    let messages = messages_from(&body);
    let assistant = first_assistant(&messages);

    assert_eq!(assistant["reasoning"], Value::String("rFunc".into()));
    let tool_calls = assistant["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("tool call list missing"));
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["type"], Value::String("function".into()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn attaches_reasoning_to_local_shell_call() {
    skip_if_no_network!();

    let body = run_request(vec![
        user_message("u1"),
        reasoning_item("rShell"),
        local_shell_call(),
    ])
    .await;
    let messages = messages_from(&body);
    let assistant = first_assistant(&messages);

    assert_eq!(assistant["reasoning"], Value::String("rShell".into()));
    assert_eq!(
        assistant["tool_calls"][0]["type"],
        Value::String("local_shell_call".into())
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drops_reasoning_when_last_role_is_user() {
    skip_if_no_network!();

    let body = run_request(vec![
        assistant_message("aPrev"),
        reasoning_item("rHist"),
        user_message("uNew"),
    ])
    .await;
    let messages = messages_from(&body);
    assert!(messages.iter().all(|msg| msg.get("reasoning").is_none()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ignores_reasoning_before_last_user() {
    skip_if_no_network!();

    let body = run_request(vec![
        user_message("u1"),
        assistant_message("a1"),
        user_message("u2"),
        reasoning_item("rAfterU1"),
    ])
    .await;
    let messages = messages_from(&body);
    assert!(messages.iter().all(|msg| msg.get("reasoning").is_none()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn skips_empty_reasoning_segments() {
    skip_if_no_network!();

    let body = run_request(vec![
        user_message("u1"),
        assistant_message("a1"),
        reasoning_item(""),
        reasoning_item("   "),
    ])
    .await;
    let messages = messages_from(&body);
    let assistant = first_assistant(&messages);
    assert!(assistant.get("reasoning").is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn suppresses_duplicate_assistant_messages() {
    skip_if_no_network!();

    let body = run_request(vec![assistant_message("dup"), assistant_message("dup")]).await;
    let messages = messages_from(&body);
    let assistant_messages: Vec<_> = messages
        .iter()
        .filter(|msg| msg["role"] == "assistant")
        .collect();
    assert_eq!(assistant_messages.len(), 1);
    assert_eq!(
        assistant_messages[0]["content"],
        Value::String("dup".into())
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepends_system_message_with_instructions() {
    skip_if_no_network!();

    let body = run_request(vec![user_message("hi")]).await;
    let messages = messages_from(&body);
    assert_eq!(messages[0]["role"], Value::String("system".into()));
    assert!(
        messages[0]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
    );
    assert_eq!(messages[1]["role"], Value::String("user".into()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sets_stream_and_model_in_payload() {
    skip_if_no_network!();

    let body = run_request(vec![user_message("hi")]).await;
    assert_eq!(body["stream"], Value::Bool(true));
    assert!(body["model"].as_str().is_some());
}
