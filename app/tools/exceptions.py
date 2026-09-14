class ToolError(Exception):
    """Base for tool-related failures. A future agent can catch this
    alone to handle "the tool call didn't work" generically, or catch
    one of the subtypes below to react to why specifically.
    """


class ToolInputError(ToolError):
    """Raised when a tool's raw_input fails validation against its own
    input schema - the caller (eventually an LLM's function-call
    arguments) supplied something the tool cannot execute.
    """


class BusinessServiceUnavailableError(ToolError):
    """Raised when the business service (the future NestJS Business API)
    cannot be reached or fails. Distinct from ToolExecutionError so a
    caller can tell "the business system is down, maybe retry" from
    "something else in the tool went wrong."
    """


class ToolExecutionError(ToolError):
    """Raised when a tool's own operation fails for a reason that is
    neither invalid input nor business-service unavailability.
    """
