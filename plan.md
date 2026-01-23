# GitHub Copilot Provider - Implementation Plan

## Overview

Implement GitHub Copilot as a provider for LLM-API-Key-Proxy following the official opencode reference implementation.

**Reference:** `PRD.md`

---

## Task List

```json
[
  {
    "category": "setup",
    "description": "Create GitHub Copilot Auth Base class",
    "steps": [
      "Create src/rotator_library/providers/github_copilot_auth_base.py",
      "Define CLIENT_ID and OAuth URLs as constants",
      "Implement _perform_device_flow_oauth() method",
      "Implement _load_credentials() and _save_credentials()",
      "Implement get_auth_header() method",
      "Implement setup_credential() for credential tool integration",
      "Implement list_credentials() method"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Create GitHub Copilot Provider class",
    "steps": [
      "Create src/rotator_library/providers/github_copilot_provider.py",
      "Inherit from GitHubCopilotAuthBase and ProviderInterface",
      "Define provider_env_name and tier configuration",
      "Define AVAILABLE_MODELS list",
      "Implement has_custom_logic() returning True",
      "Implement get_models() method"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement chat completions endpoint",
    "steps": [
      "Implement _build_copilot_headers() helper",
      "Implement _detect_vision_content() helper",
      "Implement acompletion() for non-streaming",
      "Route to /chat/completions endpoint",
      "Handle response translation to litellm format"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement streaming support",
    "steps": [
      "Implement _stream_response() async generator",
      "Parse SSE chunks correctly",
      "Yield litellm.ModelResponse chunks",
      "Handle stream completion properly"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement Responses API for GPT-5/o-series",
    "steps": [
      "Define RESPONSES_API_MODELS set",
      "Implement _responses_api_completion() method",
      "Route /responses endpoint for qualifying models",
      "Handle different response format"
    ],
    "passes": false
  },
  {
    "category": "integration",
    "description": "Integrate with provider factory",
    "steps": [
      "Add GitHubCopilotAuthBase to PROVIDER_MAP in provider_factory.py",
      "Verify auto-discovery of provider class works"
    ],
    "passes": false
  },
  {
    "category": "integration",
    "description": "Integrate with credential tool",
    "steps": [
      "Add github_copilot to OAUTH_FRIENDLY_NAMES",
      "Add github_copilot to oauth_providers list",
      "Test credential setup flow"
    ],
    "passes": false
  },
  {
    "category": "testing",
    "description": "Test authentication flow",
    "steps": [
      "Run credential tool and authenticate",
      "Verify credential file is created in oauth_creds/",
      "Verify credential is loadable"
    ],
    "passes": false
  },
  {
    "category": "testing",
    "description": "Test API endpoints",
    "steps": [
      "Test /v1/models lists github_copilot models",
      "Test non-streaming chat completion",
      "Test streaming chat completion",
      "Verify response format is correct"
    ],
    "passes": false
  },
  {
    "category": "polish",
    "description": "Add enterprise support",
    "steps": [
      "Support GITHUB_COPILOT_ENTERPRISE_DOMAIN env var",
      "Detect enterprise from credential metadata",
      "Use correct API base URL for enterprise"
    ],
    "passes": false
  }
]
```

---

## Agent Instructions

1. Read `activity.md` first to understand current state
2. Find next task with `"passes": false`
3. Complete all steps for that task
4. Verify implementation works
5. Update task to `"passes": true`
6. Log completion in `activity.md`
7. Repeat until all tasks pass

**Important:** Only modify the `passes` field. Do not remove or rewrite tasks.

---

## Completion Criteria

All tasks marked with `"passes": true`
