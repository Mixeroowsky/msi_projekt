from strlearn.ensembles import WAE,AWE,AUE, SEA
from strlearn.evaluators import TestThenTrain
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from strlearn.metrics import precision
import numpy as np
from sklearn.metrics import accuracy_score
import strlearn as sl
from sea import SEA as my_sea
from tabulate import tabulate
from scipy.stats import ranksums

scores = []
drifts_names = dict([
    (0, "sudden drift"),
    (1, "gradual drift"),
    (2, "incremetal drift")
])
list = [[] for i in range (3)]


for rst in range(10, 110, 10):
    stream = sl.streams.StreamGenerator(n_chunks=200,
                                        chunk_size=500,
                                        random_state=rst,
                                        n_classes=2,
                                        n_drifts=1,
                                        n_features=10)
    list[0].append(stream)
    stream = sl.streams.StreamGenerator(n_chunks=200,
                                        chunk_size=500,
                                        random_state=rst,
                                        n_classes=2,
                                        n_drifts=1,
                                        n_features=10,
                                        concept_sigmoid_spacing=5)
    list[1].append(stream)
    stream = sl.streams.StreamGenerator(n_chunks=200,
                                        chunk_size=500,
                                        random_state=rst,
                                        n_classes=2,
                                        n_drifts=1,
                                        n_features=10,
                                        concept_sigmoid_spacing=5,
                                        incremental=True)
    list[2].append(stream)

metrics = [accuracy_score, precision]
# chosen metrics
metrics = [accuracy_score,
           precision]


# metrics name
metrics_names = ["accuracy_score",
                 "precision"]
results = []
for index, stream in enumerate(list):
    for st in stream:
        clfs = [
            WAE(GaussianNB(), n_estimators=10),
            AUE(GaussianNB(), n_estimators=10),
            AWE(GaussianNB(), n_estimators=10),
            SEA(GaussianNB(), n_estimators=10),
            my_sea(GaussianNB(), n_estimators=10),
        ]
        clf_names = [
            "WAE",
            "AUE",
            "AWE",
            "SEA",
            "my_sea",
        ]
        evaluator = TestThenTrain(metrics)
        evaluator.process(st, clfs)
        results.append(evaluator.scores)

    np.save("results", results)

    scores = np.load('results.npy')
    print("\nScores:\n", scores.shape)

    mean_scores = np.mean(scores, axis=2)
    print("\nMean scores:\n", mean_scores)

    from scipy.stats import rankdata
    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    print("\nRanks:\n", ranks)

    mean_ranks = np.mean(ranks, axis=0)
    print("\nMean ranks:\n", mean_ranks)

    fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
    for m, metric in enumerate(metrics):
        ax[m].set_title(metrics_names[m])
        ax[m].set_ylim(0, 1)
        for i, clf in enumerate(clfs):
            ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
        plt.ylabel("Metric")
        plt.xlabel("Chunk")
        ax[m].legend()
    plt.show()


    clfs = {
        'WAE': WAE(base_estimator=GaussianNB()),
        'AUE': AUE(base_estimator=GaussianNB()),
        'AWE': AWE(base_estimator=GaussianNB()),
        'SEA': SEA(base_estimator=GaussianNB()),
        'my_sea': my_sea(base_estimator=GaussianNB)
    }


    alfa = .05
    w_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])


    headers = list(clfs.keys())
    names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)
