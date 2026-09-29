"""The Worldline-on-G1 contract: tools, results, executions, events, state machine,
observations, services, skills and common types (PLAN 5, 6).

Stdlib only. Imported by the py3.11 runtime, the py3.12 body server and the py3.11
Isaac process, so nothing here may import numpy, zmq, msgpack or any project package.

    from api.tools import TOOL_SPECS, SchemaContext, json_schemas, ToolCall
    from api.results import ToolResult, finish
    from api.execution import Execution, ExecutionManager, ResultHandle, Rejected
    from api.services import RobotBridge
"""

API_VERSION = "1.0"

__all__ = ["API_VERSION"]
