# aoop-lab6-pybind11

## Environment Setup

### Option1: Docker environment

Make sure you have installed Docker in your computer. Check announcement on e3 for docker installation tutorial.

```bash
source docker_run.sh
```
for the second or more terminals
```bash
source docker_join.sh
```
In docker container, install pybind11
```bash
pip install pybind11
```

### Option2: conda environment

For those have problem installing docker. Conda environment is an alternative. But also need installation.
Check this site for instruction. [Installing Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2)

```bash
conda env create -f environment.yml
conda activate pybind11_lab # or source activate pybind11_lab
```

### Last option: direct install

If you failed to use docker and conda. Create virtual environment here (aoop-lab6-pybind).

```bash
python3 -m venv pybind11_env
source pybind11_env/bin/activate
pip install -r requirements.txt
sudo apt-get install build-essential python3-dev
```

## Compilation

### Task1
```bash
~/aoop-lab6-pybind/Task1$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) math_ops.cpp -o math_ops$(python3-config --extension-suffix) 
```

### Task2
```bash
~/aoop-lab6-pybind/Task2$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) geometry.cpp -o geometry$(python3-config --extension-suffix)
```

### Task3
```bash
~/aoop-lab6-pybind/Task3$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) statistics.cpp -o statistics$(python3-config --extension-suffix)
```