from typing import Protocol, Optional


class MetricProtocol(Protocol):
    episode: int
    kl_divergence: float
    quotient_graph_size: int
    graph_density: float


class Evaluation:
    episode: int = -1
    kl_divergence: float
    quotient_graph_size: int
    graph_density: float

    def __init__(self, kl_divergence, quotient_graph_size, graph_density) -> None:
        self.kl_divergence = kl_divergence
        self.quotient_graph_size = quotient_graph_size
        self.graph_density = graph_density


class Train:
    epsiode: int
    kl_divergence: float
    quotient_graph_size: int
    graph_density: float

    def __init__(self, kl_divergence, quotient_graph_size, graph_density) -> None:
        self.kl_divergence = kl_divergence
        self.quotient_graph_size = quotient_graph_size
        self.graph_density = graph_density

    def mark_episode(self, episode: int):
        self.episode = episode
