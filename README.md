<p align="center">
    <img src="docs/images/oraqle_logo_cropped.svg" alt="Oraqle logo" style="width: 70%;"><br>
    <b>A secure computation compiler</b>
</p>

The oraqle compiler lets you generate arithmetic circuits from high-level Python code. It also lets you generate code using HElib.

This repository uses a fork of fhegen as a dependency and adapts some of the code from [fhegen](https://github.com/Crypto-TII/fhegen), which was written by Johannes Mono, Chiara Marcolla, Georg Land, Tim Güneysu, and Najwa Aaraj. You can read their theoretical work at: https://eprint.iacr.org/2022/706.

See [our documentation](https://jelle-vos.nl/oraqle) for more details.

## Setting up
The best way to get things up and running is using a virtual environment:
- Set up a virtualenv using `python3 -m venv venv` in the directory.
- Enter the virtual environment using `source venv/bin/activate`.
- Install the requirements using `pip install requirements.txt`.
- *To overcome import problems*, run `pip install -e .`, which will create links to your files (so you do not need to re-install after every change).

We are currently setting up documentation to be rendered using GitHub Actions.
