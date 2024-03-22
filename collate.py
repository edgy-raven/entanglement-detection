import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="")

TOTAL_STATES = 1500

if __name__ == "__main__":
    args = parser.parse_args()

    results = {}
    for shard in range(1, 7):
        for qubits in range(2, 7):
            try:
                df = pd.read_csv(f'{args.prefix}{qubits}_{shard}.csv', names=['random', 'optimal', 'tree', 'forest'])
            except FileNotFoundError:
                break
            if shard == 1:
                results[qubits] = df
            else:
                results[qubits] += df

    for qubits, qubit_results in results.items():
        avg = qubit_results.diff().mul(range(len(qubit_results)), axis=0).sum(axis=0, skipna=True) / TOTAL_STATES
        col_values = [avg['forest'], avg['tree'], avg['tree'] - avg['forest'], avg['optimal'], avg['random']]
        print(" & ".join([str(qubits)] + [f"{v:.2f}" for v in col_values]) + " \\\\")
        qubit_results = qubit_results[:qubit_results[(qubit_results == TOTAL_STATES).all(axis=1)].index[0] + 1]
        qubit_results.to_csv(f"access_{qubits}.csv")
