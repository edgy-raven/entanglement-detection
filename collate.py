import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="")

if __name__ == "__main__":
    args = parser.parse_args()
    all_results = None

    for shard in range(1, 7):
        shard_results = {}
        with open(f'{args.prefix}{shard}.txt') as f:
            current = []
            qubits = None
            for line in f.readlines():
                if line.startswith("-"):
                    shard_results[qubits] = np.array(current)
                    qubits = None
                    current = []
                elif qubits is not None:
                    current.append(list(map(int, line.strip().split(","))))
                else:
                    qubits = int(line.strip().partition("=")[2])
        if all_results is not None:
            for qubit in all_results:
                all_results[qubit] += shard_results[qubit]
        else:
            all_results = shard_results

    for qubit in all_results:
        avg = np.sum(np.diff(np.transpose(all_results[qubit])) * np.arange(1, 3**qubit), axis=1) / 2100
        print(f"{qubit} & {avg[3]:.2f} & {avg[2]:.2f} & {avg[2]-avg[3]:.2f} & {avg[0]:.2f} & {avg[1]:.2f} \\\\")

    for qubit in all_results:
        with open(f'access{qubit}.txt', 'w') as f:
            f.write("steps,random,optimal,laskowski,forest\n")
            m = 1
            for row in all_results[qubit]:
                f.write(f"{m},{','.join(map(str,row))}\n")
                if all(s == 2100 for s in row):
                    break
                m += 1
