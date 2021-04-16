from ...protocol import InferenceProtocol


class MLInference(InferenceProtocol):
    pass


class DLInference(InferenceProtocol):
    pass


__all__ = [
    "MLInference",
    "DLInference",
]
