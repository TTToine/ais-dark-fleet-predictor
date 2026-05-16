from typing import Optional, Dict, Any
from datetime import datetime

class AISPipelineError(Exception):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, recoverable: bool = False):
        self.timestamp = datetime.utcnow()
        self.context = context or {}
        self.recoverable = recoverable
        super().__init__(message)
    def to_dict(self) -> Dict[str, Any]:
        return {"error_type": self.__class__.__name__, "message": str(self), "timestamp": self.timestamp.isoformat(), "context": self.context, "recoverable": self.recoverable}

class DataValidationError(AISPipelineError): pass
class CausalLeakageError(AISPipelineError): pass
class ModelInferenceError(AISPipelineError): pass
class ConfigurationError(AISPipelineError): pass
class ResourceExhaustionError(AISPipelineError): pass
