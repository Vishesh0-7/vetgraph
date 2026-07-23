class VetGraphError(Exception):
    """Base exception for the VetGraph extraction service."""
    def __init__(self, message: str, provider_name: str, original_exception: Exception = None):
        super().__init__(message)
        self.provider_name = provider_name
        self.original_exception = original_exception

    @property
    def actionable_context(self) -> str:
        """Provides a summary of the failure context for developers."""
        if self.original_exception:
            return f"Original cause (Type: {type(self.original_exception).__name__}): {str(self.original_exception)}"
        return "No original exception captured."

    def __str__(self):
        context = "\n[DEBUG CONTEXT]\n  Actionable Context: " + self.actionable_context + "\n"
        return f"{super().__str__()} ({self.provider_name}) {context}"