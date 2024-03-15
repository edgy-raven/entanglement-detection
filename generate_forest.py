import os

import argparse
import numpy as np
import pandas as pd
from sklearn import ensemble

# Since the generated data is mostly noise, the forest tries to use as many trees as possible.
# Cap it at some reasonable number to keep disk space down.

parser = argparse.ArgumentParser()
parser.add_argument("--input_root")
parser.add_argument("--output_root")

classifier_kwargs = {
    3: {"n_estimators": 200, "min_weight_fraction_leaf": 0.0001},
}

if __name__ == "__main__":
    args = parser.parse_args()

    for qubits in range(2, 7):
        kwargs = classifier_kwargs.get(qubits, {"n_estimators": 200, "min_weight_fraction_leaf": 0.0005})
        kwargs['n_jobs'] = 8
        kwargs['random_state'] = 0
        kwargs['criterion'] = "entropy"
        for ix in range(0, 3 ** qubits):
            dfs = []
            for shard in range(1, 7):
                dfs.append(pd.read_csv(f"{args.input_root}/{qubits}_{ix}_{shard}.csv", header=None))
            df = pd.concat(dfs)
            X, y = df.iloc[:, :-1], df.iloc[:, -1]

            clf = ensemble.RandomForestClassifier(**kwargs)
            clf.fit(X, y)
            if qubits == 2 and ix == 0:
                importances = clf.feature_importances_
                print(importances)

            with open(os.path.join(args.output_root, f"{qubits}_{ix}.txt"), "w") as f:
                for estimator in clf.estimators_:
                    tree = estimator.tree_
                    p = (tree.children_left == -1) * np.array(list(_v[0][1] for _v in tree.value))
                    n = (tree.children_left == -1) * np.array(list(_v[0][0] for _v in tree.value))
                    f.write("\n".join([
                        str(len(tree.feature)),
                        ",".join(map(str, tree.feature)),
                        ",".join([f"{t:.2g}" for t in tree.threshold]),
                        ",".join(map(str, tree.children_left)),
                        ",".join(map(str, tree.children_right)),
                        ",".join([f"{int(t):d}" for t in p]),
                        ",".join([f"{int(t):d}" for t in n]),
                        ""
                    ]))
