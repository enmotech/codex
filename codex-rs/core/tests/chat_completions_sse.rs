#![allow(clippy::expect_used)]

use std::sync::Arc;

use assert_matches::assert_matches;
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
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionSource;
use core_test_support::TestCodexResponsesRequestKind;
use core_test_support::load_default_config_for_test;
use core_test_support::responses_metadata as test_responses_metadata;
use core_test_support::skip_if_no_network;
use futures::StreamExt;
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

async fn run_stream(sse_body: &str) -> Vec<ResponseEvent> {
    run_stream_with_bytes(sse_body.as_bytes()).await
}

async fn run_stream_with_bytes(sse_body: &[u8]) -> Vec<ResponseEvent> {
    let server = MockServer::start().await;

    let template = ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_bytes(sse_body.to_vec());

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
    prompt.input = vec![ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "hello".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];

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
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(ev) => events.push(ev),
            Err(_e) => break,
        }
    }
    events
}

fn assert_message(item: &ResponseItem, expected: &str) {
    if let ResponseItem::Message { content, .. } = item {
        let text = content.iter().find_map(|part| match part {
            ContentItem::OutputText { text } | ContentItem::InputText { text } => Some(text),
            _ => None,
        });
        let Some(text) = text else {
            panic!("message missing text: {item:?}");
        };
        assert_eq!(text, expected);
    } else {
        panic!("expected message item, got: {item:?}");
    }
}

fn assert_reasoning(item: &ResponseItem, expected: &str) {
    if let ResponseItem::Reasoning {
        content: Some(parts),
        ..
    } = item
    {
        let mut combined = String::new();
        for part in parts {
            match part {
                ReasoningItemContent::ReasoningText { text }
                | ReasoningItemContent::Text { text } => combined.push_str(text),
            }
        }
        assert_eq!(combined, expected);
    } else {
        panic!("expected reasoning item, got: {item:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streams_text_without_reasoning() {
    skip_if_no_network!();

    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{}}]}\n\n",
        "data: [DONE]\n\n",
    );

    let events = run_stream(sse).await;
    assert_eq!(events.len(), 4, "unexpected events: {events:?}");

    match &events[0] {
        ResponseEvent::OutputItemAdded(ResponseItem::Message { .. }) => {}
        other => panic!("expected initial assistant item, got {other:?}"),
    }

    match &events[1] {
        ResponseEvent::OutputTextDelta(text) => assert_eq!(text, "hi"),
        other => panic!("expected text delta, got {other:?}"),
    }

    match &events[2] {
        ResponseEvent::OutputItemDone(item) => assert_message(item, "hi"),
        other => panic!("expected terminal message, got {other:?}"),
    }

    assert_matches!(events[3], ResponseEvent::Completed { .. });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streams_reasoning_from_string_delta() {
    skip_if_no_network!();

    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think1\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{} ,\"finish_reason\":\"stop\"}]}\n\n",
    );

    let events = run_stream(sse).await;
    assert_eq!(events.len(), 7, "unexpected events: {events:?}");

    match &events[0] {
        ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { .. }) => {}
        other => panic!("expected initial reasoning item, got {other:?}"),
    }

    match &events[1] {
        ResponseEvent::ReasoningContentDelta {
            delta,
            content_index,
        } => {
            assert_eq!(delta, "think1");
            assert_eq!(content_index, &0);
        }
        other => panic!("expected reasoning delta, got {other:?}"),
    }

    match &events[2] {
        ResponseEvent::OutputItemAdded(ResponseItem::Message { .. }) => {}
        other => panic!("expected initial message item, got {other:?}"),
    }

    match &events[3] {
        ResponseEvent::OutputTextDelta(text) => assert_eq!(text, "ok"),
        other => panic!("expected text delta, got {other:?}"),
    }

    match &events[4] {
        ResponseEvent::OutputItemDone(item) => assert_reasoning(item, "think1"),
        other => panic!("expected terminal reasoning, got {other:?}"),
    }

    match &events[5] {
        ResponseEvent::OutputItemDone(item) => assert_message(item, "ok"),
        other => panic!("expected terminal message, got {other:?}"),
    }

    assert_matches!(events[6], ResponseEvent::Completed { .. });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streams_reasoning_from_object_delta() {
    skip_if_no_network!();

    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning\":{\"text\":\"partA\"}}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning\":{\"content\":\"partB\"}}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{} ,\"finish_reason\":\"stop\"}]}\n\n",
    );

    let events = run_stream(sse).await;
    assert_eq!(events.len(), 8, "unexpected events: {events:?}");

    match &events[0] {
        ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { .. }) => {}
        other => panic!("expected initial reasoning item, got {other:?}"),
    }

    match &events[1] {
        ResponseEvent::ReasoningContentDelta {
            delta,
            content_index,
        } => {
            assert_eq!(delta, "partA");
            assert_eq!(content_index, &0);
        }
        other => panic!("expected reasoning delta, got {other:?}"),
    }

    match &events[2] {
        ResponseEvent::ReasoningContentDelta {
            delta,
            content_index,
        } => {
            assert_eq!(delta, "partB");
            assert_eq!(content_index, &1);
        }
        other => panic!("expected reasoning delta, got {other:?}"),
    }

    match &events[3] {
        ResponseEvent::OutputItemAdded(ResponseItem::Message { .. }) => {}
        other => panic!("expected initial message item, got {other:?}"),
    }

    match &events[4] {
        ResponseEvent::OutputTextDelta(text) => assert_eq!(text, "answer"),
        other => panic!("expected text delta, got {other:?}"),
    }

    match &events[5] {
        ResponseEvent::OutputItemDone(item) => assert_reasoning(item, "partApartB"),
        other => panic!("expected terminal reasoning, got {other:?}"),
    }

    match &events[6] {
        ResponseEvent::OutputItemDone(item) => assert_message(item, "answer"),
        other => panic!("expected terminal message, got {other:?}"),
    }

    assert_matches!(events[7], ResponseEvent::Completed { .. });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streams_reasoning_from_final_message() {
    skip_if_no_network!();

    let sse = "data: {\"choices\":[{\"message\":{\"reasoning\":\"final-cot\"},\"finish_reason\":\"stop\"}]}\n\n";

    let events = run_stream(sse).await;
    assert_eq!(events.len(), 4, "unexpected events: {events:?}");

    match &events[0] {
        ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { .. }) => {}
        other => panic!("expected initial reasoning item, got {other:?}"),
    }

    match &events[1] {
        ResponseEvent::ReasoningContentDelta {
            delta,
            content_index,
        } => {
            assert_eq!(delta, "final-cot");
            assert_eq!(content_index, &0);
        }
        other => panic!("expected reasoning delta, got {other:?}"),
    }

    match &events[2] {
        ResponseEvent::OutputItemDone(item) => assert_reasoning(item, "final-cot"),
        other => panic!("expected reasoning item, got {other:?}"),
    }

    assert_matches!(events[3], ResponseEvent::Completed { .. });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streams_reasoning_before_tool_call() {
    skip_if_no_network!();

    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning\":\"pre-tool\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"run\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
    );

    let events = run_stream(sse).await;
    assert_eq!(events.len(), 5, "unexpected events: {events:?}");

    match &events[0] {
        ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { .. }) => {}
        other => panic!("expected initial reasoning item, got {other:?}"),
    }

    match &events[1] {
        ResponseEvent::ReasoningContentDelta {
            delta,
            content_index,
        } => {
            assert_eq!(delta, "pre-tool");
            assert_eq!(content_index, &0);
        }
        other => panic!("expected reasoning delta, got {other:?}"),
    }

    match &events[2] {
        ResponseEvent::OutputItemDone(item) => assert_reasoning(item, "pre-tool"),
        other => panic!("expected reasoning item, got {other:?}"),
    }

    match &events[3] {
        ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        }) => {
            assert_eq!(name, "run");
            assert_eq!(arguments, "{}");
            assert_eq!(call_id, "call_1");
        }
        other => panic!("expected function call, got {other:?}"),
    }

    assert_matches!(events[4], ResponseEvent::Completed { .. });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streams_tool_call_with_streamed_arguments() {
    skip_if_no_network!();

    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"run\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
    );

    let events = run_stream(sse).await;
    assert_eq!(events.len(), 2, "unexpected events: {events:?}");

    match &events[0] {
        ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        }) => {
            assert_eq!(name, "run");
            assert_eq!(arguments, "{\"a\":1}");
            assert_eq!(call_id, "call_1");
        }
        other => panic!("expected function call, got {other:?}"),
    }

    assert_matches!(events[1], ResponseEvent::Completed { .. });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completes_on_done_sentinel_without_json() {
    skip_if_no_network!();

    let sse = "data: [DONE]\n\n";
    let events = run_stream(sse).await;
    assert_matches!(events.last(), Some(ResponseEvent::Completed { .. }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completes_on_stream_end_without_done_sentinel() {
    skip_if_no_network!();

    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
    let events = run_stream(sse).await;
    assert_matches!(events.last(), Some(ResponseEvent::Completed { .. }));
}
