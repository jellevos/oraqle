This folder contains all the experiments for the depth-aware arithmetization paper.
The three experiments from section 9 are directly in this folder.
The other experiments are in the `execution` subfolder.
The experiment where we compare against TFHE is in the `tfhe` subfolder.
Some of these experiments require executing with HElib, which requires you to call them from the `helib_template` directory.
After building them using any C++ compiler (we tested g++ and clang), you can call them with command line arguments like `x=3` to indicate that the `x` variable should have value `3`. Any unspecified variables will have value `-1`, which HElib still handles as a valid input. Since the actual value should not matter for the final run time, it is possible to execute the commands without the CLI arguments.
