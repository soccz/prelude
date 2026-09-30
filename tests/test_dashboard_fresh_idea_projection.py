"""Builder wiring: one freshly recomputed idea report, no operational overwrite."""
import ast
from pathlib import Path


def test_fresh_validated_idea_is_embedded_before_the_only_summary_write():
    source = Path("scripts/build_dashboard.py").read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    fresh = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "build_idea_validation_report")
    validated = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "validate_idea_validation_payload")
    writes = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "_write_json"
              and "summary.json" in ast.unparse(node.args[0])]
    assert len(writes) == 1
    assignment = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                      and any(ast.unparse(target) == "summary['idea_validation']" for target in node.targets))
    assert ast.unparse(assignment.value) == "idea_payload"
    assert fresh.lineno < validated.lineno < assignment.lineno < writes[0].lineno
    assert "write_idea_outputs(" not in ast.unparse(main)
