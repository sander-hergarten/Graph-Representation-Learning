from .bisimulation import find_bisimilar_graph
from .utils import config
from .train import train
from .evaluation import create_evaluation
from .io.logging import log
from .io import Environment 
from . import embedding as Embedding
from . import grapher as Grapher


def main():
    # TRAINING LOOP
    for episode in config["num_episodes"]:
        # generate observations
        observations = Environment.generate(config["max_steps_per_episode"])
        # generate embeddings
        embeddings = Embedding.from_observations(observations)
        # generate graph
        graph = Grapher.encode(embeddings)
        # find bisimilar graph
        bisimilar_graph = find_bisimilar_graph(graph)
        # decode graph
        reconstructed_embeddings = Grapher.decode(bisimilar_graph)
        # train
        training_metrics = train(embeddings, reconstructed_embeddings)
        training_metrics.mark_episode(episode)

        log(training_metrics)

    # evaluate
    evaluation_samples = Environment.evaluation_samples
    evaluation_metrics = create_evaluation(evaluation_samples)
    log(evaluation_metrics)
    

if __name__ == "__main__":
    main()
