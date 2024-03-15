# What's included
- `cliques.cpp` : Generates cliques for Laskowski tree algorithm (requires C++ Boost for Bron-Kerbosch algorithm).
- `cliques/<qubits>.csv` : Output from `cliques.cpp`.
- `datagen.cpp` : Generates the dataset for training.
- `detect.cpp` : Runs the detection algorithms and outputs results.
- `operators.cpp` : Shared code between `datagen.cpp` and `operators.cpp`.
- `generate_forest.py` : Trains the random forests (requires sklearn).
- `collate.py` : Collects the outputs from `detect.cpp`.

# Reproducing the data from the paper
You don't need to run `cliques.cpp`, it is provided for reference only.

On my system, I used `/drv1` as the path to store training and model data, you'll probably need to change this in the code and recompile. It's a separate external SSD -- there is not a LOT of data, but it's significant.

In the following commands that `Eigen` is available on your path and `sklearn` (and dependent libraries like `numpy` and `pandas` are available in your python environment). The steps to reproduce the results in the paper are:
- `g++ datagen.cpp -o run_datagen -O9 -I.`
- `for i in {1..6}; do ./run_datagen $i & done;`
    - Ubuntu bash syntax, you might need to change for Windows
    - I'm running this on my personal computer, more shards meant that I couldn't do my personal stuff
    - This generates about 200G of data and runs in about ~10 hours on my 2018 desktop.
- `python generate_forest.py --input_root /drv1/training_data --output_root /drv1/models`
    - sklearn already has an `n_jobs` parameter for parallelism.
    - Haven't timed this, but it seems like it takes ~5 hours.
- `g++ detect.cpp -o run_detect -O9 -I.`
- `for i in {1..6}; do ./run_detect $i > $i.txt & done;`
    - Takes about 20 hours.
- `python collate.py`
    - Just generates the data used in the figures.
