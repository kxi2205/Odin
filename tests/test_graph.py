import pytest
from unittest.mock import patch, MagicMock

from odin.graph import build_graph
from odin.state import RunHistory, GraphState

@pytest.fixture
def mock_graph_deps():
    with patch("odin.nodes.generate_code") as mock_gen, \
         patch("odin.nodes.repair_code") as mock_rep, \
         patch("odin.nodes.execute_code") as mock_exec, \
         patch("odin.nodes.classify_error") as mock_class:
        yield mock_gen, mock_rep, mock_exec, mock_class


@pytest.mark.asyncio
async def test_graph_sequence_fail_then_pass(mock_graph_deps):
    mock_gen, mock_rep, mock_exec, mock_class = mock_graph_deps
    
    # generate returns some files
    mock_gen.return_value = ([], MagicMock())
    
    # exec fails first time, passes second time
    fail_exec = MagicMock(success=False, traceback="error1", stderr="error1")
    pass_exec = MagicMock(success=True)
    mock_exec.side_effect = [fail_exec, pass_exec]
    
    # classify gives logical error first time
    mock_class.return_value = "logical"
    
    # repair returns files
    mock_rep.return_value = ([], MagicMock())
    
    run_history = RunHistory(requirement="test")
    state: GraphState = {
        "requirement": "test",
        "test_input": "",
        "is_continuation": False,
        "user_feedback": "",
        "past_run_context": None,
        "iteration": 0,
        "current_files": [],
        "last_execution": None,
        "last_error_analysis": None,
        "last_retrieval": None,
        "max_iterations": 5,
        "execution_timeout": 5,
        "run_history": run_history,
        "success": False,
        "should_end": False,
    }
    
    graph = build_graph()
    
    # run graph
    final_state = await graph.ainvoke(state)
    
    assert final_state["success"] is True
    assert final_state["should_end"] is True
    assert final_state["iteration"] == 2
    assert mock_gen.call_count == 1
    assert mock_exec.call_count == 2
    assert mock_class.call_count == 1
    assert mock_rep.call_count == 1


@pytest.mark.asyncio
async def test_graph_max_iterations(mock_graph_deps):
    mock_gen, mock_rep, mock_exec, mock_class = mock_graph_deps
    
    mock_gen.return_value = ([], MagicMock())
    mock_rep.return_value = ([], MagicMock())
    mock_class.return_value = "logical"
    
    # always fail
    fail_exec = MagicMock(success=False, traceback="error1", stderr="error1")
    mock_exec.return_value = fail_exec
    
    run_history = RunHistory(requirement="test")
    state: GraphState = {
        "requirement": "test",
        "test_input": "",
        "is_continuation": False,
        "user_feedback": "",
        "past_run_context": None,
        "iteration": 0,
        "current_files": [],
        "last_execution": None,
        "last_error_analysis": None,
        "last_retrieval": None,
        "max_iterations": 3,
        "execution_timeout": 5,
        "run_history": run_history,
        "success": False,
        "should_end": False,
    }
    
    graph = build_graph()
    final_state = await graph.ainvoke(state)
    
    assert final_state["success"] is False
    assert final_state["should_end"] is True
    assert run_history.final_status == "max_iterations_reached"
    assert final_state["iteration"] == 3


@pytest.mark.asyncio
async def test_retrieval_routing_reevaluated(mock_graph_deps):
    mock_gen, mock_rep, mock_exec, mock_class = mock_graph_deps
    
    mock_gen.return_value = ([], MagicMock())
    mock_rep.return_value = ([], MagicMock())
    
    fail_exec = MagicMock(success=False, traceback="error", stderr="error")
    pass_exec = MagicMock(success=True)
    # Fail 2 times, pass 3rd
    mock_exec.side_effect = [fail_exec, fail_exec, pass_exec]
    
    # First time syntax (should not retrieve), second time attribute (should retrieve)
    mock_class.side_effect = ["syntax", "attribute"]
    
    run_history = RunHistory(requirement="test")
    state: GraphState = {
        "requirement": "test",
        "test_input": "",
        "is_continuation": False,
        "user_feedback": "",
        "past_run_context": None,
        "iteration": 0,
        "current_files": [],
        "last_execution": None,
        "last_error_analysis": None,
        "last_retrieval": None,
        "max_iterations": 5,
        "execution_timeout": 5,
        "run_history": run_history,
        "success": False,
        "should_end": False,
    }
    
    graph = build_graph()
    
    with patch("odin.nodes.RAGRetriever") as mock_rag_cls:
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = []
        mock_rag_cls.return_value = mock_rag
        
        final_state = await graph.ainvoke(state)
        
        assert final_state["success"] is True
        # Check retrievals
        rets = run_history.retrievals
        assert len(rets) == 2
        assert rets[0].triggered is False # syntax
        assert rets[1].triggered is True # attribute
