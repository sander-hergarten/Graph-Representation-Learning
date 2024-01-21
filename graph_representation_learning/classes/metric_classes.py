from typing import Protocol, Optional

class MetricProtocol(Protocol):
    ...

class Evaluation:
    def __init__(self, samples) -> None:
        ...

class Train:
    def __init__(self) -> None:
        ...

    def mark_episode(self, episode:Optional[int]=None):
        ...
