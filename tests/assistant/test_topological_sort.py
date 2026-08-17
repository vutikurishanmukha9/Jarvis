"""
Tests for Kahn's topological sort algorithm, handling linear chains, fork-join DAGs,
disconnected subgraphs, deterministic tie-breaking, and cycle detection.
"""

from src.assistant.goal_planner import topological_sort


def test_topological_sort_empty_and_single():
    """Verify empty list and single task list sorting."""
    assert topological_sort([]) == []

    single = [{"id": "task_1", "depends_on": []}]
    assert topological_sort(single) == single


def test_topological_sort_linear_chain():
    """Verify strict sequence T1 -> T2 -> T3 -> T4 is sorted correctly."""
    tasks = [
        {"id": "task_4", "depends_on": ["task_3"]},
        {"id": "task_2", "depends_on": ["task_1"]},
        {"id": "task_1", "depends_on": []},
        {"id": "task_3", "depends_on": ["task_2"]},
    ]
    sorted_tasks = topological_sort(tasks)
    ids = [t["id"] for t in sorted_tasks]
    assert ids == ["task_1", "task_2", "task_3", "task_4"]


def test_topological_sort_fork_join_diamond():
    """Verify fork-join diamond graph: T1 -> [T2, T3] -> T4."""
    tasks = [
        {"id": "task_4", "depends_on": ["task_2", "task_3"]},
        {"id": "task_3", "depends_on": ["task_1"]},
        {"id": "task_2", "depends_on": ["task_1"]},
        {"id": "task_1", "depends_on": []},
    ]
    sorted_tasks = topological_sort(tasks)
    ids = [t["id"] for t in sorted_tasks]
    assert ids[0] == "task_1"
    assert ids[-1] == "task_4"
    assert set(ids[1:3]) == {"task_2", "task_3"}


def test_topological_sort_disconnected_trees():
    """Verify two independent dependency trees: [A -> B] and [C -> D]."""
    tasks = [
        {"id": "task_d", "depends_on": ["task_c"]},
        {"id": "task_b", "depends_on": ["task_a"]},
        {"id": "task_c", "depends_on": []},
        {"id": "task_a", "depends_on": []},
    ]
    sorted_tasks = topological_sort(tasks)
    ids = [t["id"] for t in sorted_tasks]
    assert ids.index("task_a") < ids.index("task_b")
    assert ids.index("task_c") < ids.index("task_d")


def test_topological_sort_direct_cycle_fallback():
    """Verify direct 2-node cycle A -> B -> A triggers graceful fallback."""
    tasks = [
        {"id": "task_1", "depends_on": ["task_2"]},
        {"id": "task_2", "depends_on": ["task_1"]},
    ]
    # Cycle detected: should return original order fallback
    result = topological_sort(tasks)
    assert len(result) == 2
    assert result[0]["id"] == "task_1"
    assert result[1]["id"] == "task_2"


def test_topological_sort_deep_cycle_fallback():
    """Verify 3-node cycle A -> B -> C -> A triggers graceful fallback."""
    tasks = [
        {"id": "task_1", "depends_on": ["task_3"]},
        {"id": "task_2", "depends_on": ["task_1"]},
        {"id": "task_3", "depends_on": ["task_2"]},
    ]
    result = topological_sort(tasks)
    assert len(result) == 3


def test_topological_sort_self_dependency():
    """Verify self-dependency A -> A triggers fallback."""
    tasks = [{"id": "task_1", "depends_on": ["task_1"]}, {"id": "task_2", "depends_on": []}]
    result = topological_sort(tasks)
    assert len(result) == 2


def test_topological_sort_deterministic_alphabetical():
    """Verify tie-breaking between independent zero-in-degree tasks is deterministic."""
    tasks = [
        {"id": "task_z", "depends_on": []},
        {"id": "task_a", "depends_on": []},
        {"id": "task_m", "depends_on": []},
    ]
    sorted_tasks = topological_sort(tasks)
    ids = [t["id"] for t in sorted_tasks]
    assert ids == ["task_a", "task_m", "task_z"]
