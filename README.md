# Time- and Frequency-Domain Dynamic Spectrum Access: Learning Cyclic Medium Access Patterns in Partially Observable Environments
Open source code release for the publication at NetSys 2021, September 2021 in Lübeck, Germany.

## Instructions
Each Python `.py` file performs a simulation and saves results into a corresponding `_data/<generated_name>.json` file. These are then parsed by the same Python script to create a graph in `_imgs/<generated_name>.pdf`.  
You can check each file for its command line parameters regarding the simulation, and you can also disable simulation or plotting.

The `Makefile` contains targets to re-create the graphs presented and discussed in the publication.
Please note that for the paper, a relatively large number of repetitions are performed to obtain statistically meaningful results.
However, this requires substantital simulation time. For a quicker first check, consider simulation just once or for shorter times.

You can run `make all` to replicate all graphs from the paper in one command.
