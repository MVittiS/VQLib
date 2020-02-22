# VQToolkit

**VQToolkit** is a series of open-source MatLAB implementations of the Vector Quantization algorithms, that are GPU-accelerated, fault-tolerant, and basically free for non-commercial use. C++ implementations may follow later on to allow them to be used elsewhere.

See a usage example [here](ExampleVQ.m).


## Modes of Operation

### Dictionary

- *Accurate*, where the generated dictionary will be close to optimal at the cost of slow convergence time.
- *Fast*, where dictionary generation is considerably faster at the cost of optimality.

## Benchmarks

CPU Intel 1: Core i7 7567U @ 3.5GHz, ~3.8GHz turbo
CPU Intel 2: Core i5 3240M @ ???
CPU ARM: Raspberry Pi 3B+ @ 1.2 GHz
GPU: Nvidia GeForce 640M
MatLAB Version: R2018a
Octave Version: 4.2.0

### MatLAB/Octave CPU only

TODO: Put benchmark results in tables here

### MatLAB CPU + GPU

TODO: Put benchmark results in tables here