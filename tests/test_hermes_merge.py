"""
Unit and integration tests for merged Hermes Agent capabilities in J.A.R.V.I.S.:
1. SQLite WAL + FTS5 Full-Text Search Conversational Memory Store
2. Closed-Loop Procedural Skill Learner (agentskills.io format)
3. Omnichannel Messaging Gateway (Telegram Bot)
4. Hands-Free Ambient Wake-Word Listener
"""

from pathlib import Path

from src.assistant.skill_learner import SkillLearner
from src.core.sqlite_session import SQLiteMemoryStore, get_sqlite_memory
from src.modules.gateway.telegram_bridge import TelegramGateway, get_telegram_gateway
from src.modules.voice.wake_word import WakeWordListener, get_wake_listener

# ==============================================================================
# 1. SQLite WAL + FTS5 Full-Text Search Tests
# ==============================================================================


def test_sqlite_memory_store_lifecycle(tmp_path: Path):
    """Test creating SQLite memory, writing turns, and executing FTS5 queries."""
    db_file = tmp_path / "test_memory.db"
    store = SQLiteMemoryStore(db_path=db_file)

    session_id = "test-session-001"
    store.record_turn(
        session_id=session_id,
        role="user",
        content="Deploy the deep learning model to production Kubernetes cluster.",
        persona="DevOps Engineer",
        tools_called=["kubectl", "docker"],
    )
    store.record_turn(
        session_id=session_id,
        role="assistant",
        content="Deployment successful. Service exposed on ingress port 443 with TLS.",
        persona="DevOps Engineer",
    )

    # Verify recent turns retrieval
    turns = store.get_recent_turns(session_id, limit=10)
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert "Kubernetes" in turns[0]["content"]
    assert turns[1]["role"] == "assistant"
    assert "TLS" in turns[1]["content"]

    # Verify FTS5 keyword matching
    matches = store.search_memory("Kubernetes")
    assert len(matches) >= 1
    assert matches[0]["session_id"] == session_id
    assert "Kubernetes" in matches[0]["content"]

    # Verify another FTS5 search
    matches_tls = store.search_memory("ingress")
    assert len(matches_tls) >= 1
    assert matches_tls[0]["role"] == "assistant"

    # Search non-existent token
    empty_matches = store.search_memory("xyznonexistentterm123")
    assert len(empty_matches) == 0


def test_sqlite_memory_singleton():
    """Verify get_sqlite_memory() singleton returns an active store instance."""
    store = get_sqlite_memory()
    assert isinstance(store, SQLiteMemoryStore)
    assert store.db_path.name.endswith(".db")


# ==============================================================================
# 2. Closed-Loop Skill Learner Tests
# ==============================================================================


def test_skill_learner_workflow(tmp_path: Path):
    """Test distilling an autonomous DAG execution into an agentskills.io bundle."""
    skills_dir = tmp_path / "skills"
    learner = SkillLearner(skills_dir=skills_dir)

    goal = "Competitor Pricing Audit"
    plan = {
        "goal_summary": "Scrape competitor sites and produce financial analysis spreadsheet",
        "tasks": [
            {
                "id": "task_1",
                "title": "Fetch competitor catalog",
                "instruction": "Scrape product prices using adaptive scraper",
                "tool_hint": "scrapling_adaptive_scraper",
                "expected_deliverable": "Raw catalog JSON",
            },
            {
                "id": "task_2",
                "title": "Generate pricing matrix",
                "instruction": "Build Excel spreadsheet comparing margins",
                "tool_hint": "audit_outreach_campaign",
                "expected_deliverable": "workspace/pricing_matrix.xlsx",
            },
        ],
    }

    result = learner.learn_from_completed_plan(
        goal=goal,
        plan=plan,
        final_synthesis="Competitor matrix generated with 15 price points.",
    )

    assert result is not None
    slug = result["name"]
    assert "competitor-pricing" in slug

    # Check that SKILL.md was generated
    skill_file = skills_dir / slug / "SKILL.md"
    assert skill_file.exists()
    content = skill_file.read_text(encoding="utf-8")
    assert "name: competitor-pricing-audit" in content
    assert "tools: ['audit_outreach_campaign', 'scrapling_adaptive_scraper']" in content
    assert "## Execution Procedure" in content
    assert "Fetch competitor catalog" in content
    assert "Generate pricing matrix" in content

    # Check registry update
    registered = learner.get_registered_skills()
    assert slug in registered
    assert registered[slug]["steps_count"] == 2

    # Check match lookup
    match = learner.find_matching_skill("I want to run a competitor pricing audit on our rivals")
    assert match is not None
    assert match["name"] == slug


def test_skill_learner_single_step_skipped(tmp_path: Path):
    """Verify that trivial 1-step tasks are not saved as complex skills."""
    skills_dir = tmp_path / "skills"
    learner = SkillLearner(skills_dir=skills_dir)
    plan = {"tasks": [{"id": "1", "title": "Check time"}]}
    res = learner.learn_from_completed_plan("What time is it?", plan)
    assert res is None


# ==============================================================================
# 3. Omnichannel Gateway (Telegram) Tests
# ==============================================================================


def test_telegram_gateway_configuration():
    """Verify gateway credential detection and base URL configuration."""
    # Unconfigured state
    gw_empty = TelegramGateway(bot_token="")
    assert not gw_empty.is_configured
    info_fail = gw_empty.get_me()
    assert not info_fail["ok"]

    # Configured state
    dummy_token = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz123456"
    gw = TelegramGateway(bot_token=dummy_token)
    assert gw.is_configured
    assert gw.base_url == f"https://api.telegram.org/bot{dummy_token}"
    assert not gw.is_active()


def test_telegram_gateway_singleton():
    """Verify global gateway singleton accessor."""
    dummy_token = "987654321:XYZabc123"
    gw = get_telegram_gateway(bot_token=dummy_token)
    assert isinstance(gw, TelegramGateway)
    assert gw.bot_token == dummy_token


# ==============================================================================
# 4. Hands-Free Ambient Wake-Word Engine Tests
# ==============================================================================


def test_wake_word_listener_properties():
    """Verify wake-word listener properties and callback registration."""
    listener = WakeWordListener(wake_phrase="jarvis")
    assert listener.wake_phrase == "jarvis"
    assert not listener.is_listening()

    triggered = []

    def _dummy_callback(directive: str):
        triggered.append(directive)

    listener.on_directive_captured = _dummy_callback
    assert listener.on_directive_captured is _dummy_callback


def test_wake_word_singleton():
    """Verify global wake-word singleton accessor."""
    listener = get_wake_listener()
    assert isinstance(listener, WakeWordListener)
    assert listener.wake_phrase == "jarvis"
