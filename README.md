# TPS shape registration

> Thin-plate spline

[OUTPUT](https://i.imgur.com/ectHCgn.png)

**Requirements:**

- matplotlib
- numpy
- torch
- python3
- pyvista
- csv
- argparse

---

### Setup

> Install all the requirements first

```shell
pip install matplotlib numpy pyvista csv argparse torch torchvision torchaudio
```

> Clone the repository

```shell
git clone https://github.com/Fix-it-Felix-jr/TPS.git
```

> Go to the right folder

```shell
cd TPS/
```

> Load your figures in /data/input.txt

---

## Usage

```shell
python3 main.py -s STEPS -d DIMENSIONS -l NUMBER_OF_LANDMARKS_PER_SHAPE -m MU
```
> Where STEPS is the number of steps, DIMENSIONS is the number of dimensions of your shapes (2D/3D), NUMBER_OF_LANDMARKS_PER_SHAPE is the numeber of points stored in each shape, and MU is the value of mu

> A file output.csv will be saved in the same directory where main.py is stored. This file contains all the output points plotted by this program

---

## Known issues

- The 3D plotter doesn't work, but the output.csv file will still be saved

