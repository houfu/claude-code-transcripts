"""Tests for Claude Cowork session discovery."""

import json
from pathlib import Path

import pytest

from claude_code_transcripts import find_cowork_sessions, parse_session_file


def make_cowork_session(
    base_dir,
    org_uuid="org-123",
    workspace_uuid="ws-456",
    session_uuid="sess-789",
    process_name="quirky-eager-fermat",
    cli_session_id="cli-abc",
    title="Review VIMA model",
    last_activity_at=1700000000000,
    folders=None,
    jsonl_content=None,
):
    """Helper to create a mock Cowork session directory structure."""
    if folders is None:
        folders = ["/Users/test/Documents"]

    # Create metadata JSON
    org_dir = base_dir / org_uuid / workspace_uuid
    org_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "sessionId": f"local_{session_uuid}",
        "processName": process_name,
        "cliSessionId": cli_session_id,
        "title": title,
        "initialMessage": f"Initial: {title}",
        "userSelectedFolders": folders,
        "createdAt": last_activity_at - 60000,
        "lastActivityAt": last_activity_at,
        "model": "claude-opus-4-5",
    }
    metadata_file = org_dir / f"local_{session_uuid}.json"
    metadata_file.write_text(json.dumps(metadata))

    # Create JSONL file at expected path
    jsonl_dir = (
        org_dir
        / f"local_{session_uuid}"
        / ".claude"
        / "projects"
        / f"-sessions-{process_name}"
    )
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = jsonl_dir / f"{cli_session_id}.jsonl"

    if jsonl_content is None:
        jsonl_content = (
            '{"type":"queue-operation","data":{}}\n'
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi!"}]}}\n'
        )
    jsonl_file.write_text(jsonl_content)

    return metadata_file, jsonl_file


def test_find_cowork_sessions_empty(tmp_path):
    """Returns [] when base dir doesn't exist."""
    non_existent = tmp_path / "does-not-exist"
    result = find_cowork_sessions(base_dir=non_existent)
    assert result == []


def test_find_cowork_sessions_finds_sessions(tmp_path):
    """Finds sessions and returns correct metadata."""
    make_cowork_session(tmp_path, title="My Test Session", folders=["/Users/test/Work"])

    result = find_cowork_sessions(base_dir=tmp_path)

    assert len(result) == 1
    session = result[0]
    assert session["title"] == "My Test Session"
    assert isinstance(session["jsonl_path"], Path)
    assert session["jsonl_path"].exists()
    assert session["folders"] == ["/Users/test/Work"]
    assert isinstance(session["mtime"], float)


def test_find_cowork_sessions_skips_missing_jsonl(tmp_path):
    """Skips sessions where the JSONL file doesn't exist."""
    org_dir = tmp_path / "org-123" / "ws-456"
    org_dir.mkdir(parents=True)

    # Create metadata without the JSONL file
    metadata = {
        "sessionId": "local_sess-999",
        "processName": "test-process",
        "cliSessionId": "missing-cli",
        "title": "Missing JSONL",
        "initialMessage": "test",
        "userSelectedFolders": [],
        "createdAt": 1700000000000,
        "lastActivityAt": 1700000000000,
        "model": "claude-opus-4-5",
    }
    (org_dir / "local_sess-999.json").write_text(json.dumps(metadata))
    # No JSONL file created

    result = find_cowork_sessions(base_dir=tmp_path)
    assert result == []


def test_find_cowork_sessions_sorted_by_mtime(tmp_path):
    """Returns sessions sorted by lastActivityAt, most recent first."""
    make_cowork_session(
        tmp_path,
        session_uuid="old-session",
        process_name="old-process",
        cli_session_id="old-cli",
        title="Older Session",
        last_activity_at=1700000000000,
    )
    make_cowork_session(
        tmp_path,
        session_uuid="new-session",
        process_name="new-process",
        cli_session_id="new-cli",
        title="Newer Session",
        last_activity_at=1700001000000,
    )

    result = find_cowork_sessions(base_dir=tmp_path)

    assert len(result) == 2
    assert result[0]["title"] == "Newer Session"
    assert result[1]["title"] == "Older Session"


def test_find_cowork_sessions_limit(tmp_path):
    """Respects the limit parameter."""
    for i in range(5):
        make_cowork_session(
            tmp_path,
            session_uuid=f"session-{i}",
            process_name=f"process-{i}",
            cli_session_id=f"cli-{i}",
            title=f"Session {i}",
            last_activity_at=1700000000000 + i * 1000,
        )

    result = find_cowork_sessions(base_dir=tmp_path, limit=3)
    assert len(result) == 3


def test_cowork_jsonl_parses_with_queue_operation(tmp_path):
    """Verifies parse_session_file handles Cowork JSONL with queue-operation first line."""
    jsonl_content = (
        '{"type":"queue-operation","data":{"some":"data"}}\n'
        '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello cowork"}}\n'
        '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi from cowork!"}]}}\n'
    )
    jsonl_file = tmp_path / "session.jsonl"
    jsonl_file.write_text(jsonl_content)

    data = parse_session_file(jsonl_file)
    loglines = data["loglines"]

    # queue-operation should be filtered out
    assert len(loglines) == 2
    assert loglines[0]["type"] == "user"
    assert loglines[1]["type"] == "assistant"
    # Content should be correct
    assert loglines[0]["message"]["content"] == "Hello cowork"
