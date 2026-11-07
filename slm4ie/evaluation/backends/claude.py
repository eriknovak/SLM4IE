"""Chat backend for Claude through its headless mode (`claude -p`), on the user's subscription."""

import json
import os
import subprocess
import tempfile
from typing import Any

from slm4ie.evaluation.backends.base import ChatBackend, ChatResult, register_backend

# ===============================
# Defaults
# ===============================

# bare prompt: no tools, no session file, no skills, no MCP servers, no settings files.
# Without them the CLI adds about 66,000 tokens of its own system prompt to every call.
CLAUDE_COMMAND = [
    "claude",
    "-p",
    "--output-format",
    "json",
    "--tools",
    "",
    "--no-session-persistence",
    "--disable-slash-commands",
    "--strict-mcp-config",
    "--setting-sources",
    "",
]

# the CLI retries an answer off the schema; no retries, as for the local models (D3)
NO_RETRIES_ENV = {"MAX_STRUCTURED_OUTPUT_RETRIES": "0"}

# an exported API key would bill every window to the API, not the subscription (D8)
DROPPED_ENV_KEYS = frozenset({"ANTHROPIC_API_KEY"})

# one call is one window; a stuck CLI must not hang the run
CALL_TIMEOUT_SECONDS = 600

# the CLI's result subtype when its structured-output retries ran out: an answer off the schema
OFF_SCHEMA_SUBTYPE = "error_max_structured_output_retries"

# ===============================
# Backend class
# ===============================


@register_backend("claude")
class ClaudeBackend(ChatBackend):
    """Chat through `claude -p`, with the answer schema passed as `--json-schema`.

    The CLI has no temperature option, so decoding is the service's default, not greedy.
    The CLI has no seed option, so `seed` is ignored. It is told not to retry an off-schema answer.
    The call runs in a temporary directory so no project `CLAUDE.md` joins the prompt.
    Fresh, cache-written and cache-read input tokens are kept apart, as is the CLI's list-price cost.
    An answer the CLI could not bring onto the schema returns empty text, which `parse_answer` rejects.

    Args:
        model: A Claude alias (`opus`, `sonnet`, `haiku`) or a full model name.
    """

    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> ChatResult:
        """Run the CLI with the system turn as its system prompt and the user turn on stdin.

        Raises:
            ValueError: If `messages` is not one system turn then one user turn.
            RuntimeError: If the CLI exits non-zero or reports an error other than an off-schema answer.
        """
        roles = [message["role"] for message in messages]
        if roles != ["system", "user"]:
            raise ValueError(f"claude backend takes one system then one user message, got roles {roles}")
        system, user = messages[0]["content"], messages[1]["content"]

        # run the CLI
        command = [*CLAUDE_COMMAND, "--model", self.model, "--system-prompt", system]
        command += ["--json-schema", json.dumps(schema)]
        with tempfile.TemporaryDirectory() as workdir:
            process = subprocess.run(
                command,
                input=user,
                capture_output=True,
                text=True,
                cwd=workdir,
                timeout=CALL_TIMEOUT_SECONDS,
                env={**{k: v for k, v in os.environ.items() if k not in DROPPED_ENV_KEYS}, **NO_RETRIES_ENV},
            )
        # read the result before the exit code: an off-schema answer may exit non-zero
        reply = _result_json(process.stdout)
        if reply is None:
            raise RuntimeError(f"claude exited {process.returncode} for model {self.model!r}: {process.stderr.strip()}")
        off_schema = reply.get("subtype") == OFF_SCHEMA_SUBTYPE
        if process.returncode != 0 and not off_schema:
            raise RuntimeError(f"claude exited {process.returncode} for model {self.model!r}: {process.stderr.strip()}")

        # read the answer and the usage
        usage = reply["usage"]
        if reply.get("is_error") and not off_schema:
            raise RuntimeError(f"claude reported an error for model {self.model!r}: {reply.get('result')}")
        structured = reply.get("structured_output")
        if off_schema:
            text = ""
        elif structured is not None:
            text = json.dumps(structured, ensure_ascii=False)
        else:
            text = reply["result"]
        return ChatResult(
            text,
            usage["input_tokens"],
            usage["output_tokens"],
            0.0,
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cost_usd=reply.get("total_cost_usd"),
        )


# ===============================
# Helpers
# ===============================


def _result_json(stdout: str) -> dict | None:
    """Return the CLI's result object, or None when stdout holds no JSON object."""
    try:
        reply = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return reply if isinstance(reply, dict) else None
