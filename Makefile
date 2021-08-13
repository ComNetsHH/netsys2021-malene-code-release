#   Code to replicate results of the publication 
#   'Time- and Frequency-Domain Dynamic Spectrum Access: Learning Cyclic Medium Access Patterns in Partially Observable Environments' 
#   published at the Conference on Networked Systems (NetSys) 2021, Lübeck, Germany.
#   https://github.com/ComNetsHH/netsys2021-malene-code-release
#
#     Copyright (C) 2021  Institute of Communication Networks, 
# 			  	          Hamburg University of Technology,
# 			  	          https://www.tuhh.de/comnets
# 			  	(C) 2021  Sebastian Lindner, 
# 			  	          sebastian.lindner@tuhh.de
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Remove the # in the next line to make TensorFlow use your GPU.
ARGUMENTS = #--use_gpu --gpu_mem 1024

ALLRUNS = tests comparison_loss_over_time comparison_convergence_over_channels fully_observable_comparison_loss_over_time_matrix_input partially_observable_comparison_loss_over_time_matrix_input comparison_convergence_over_sample_length

.PHONY: $(ALLRUNS)

all: $(ALLRUNS)

tests:
	python test_env.py
	python test_confidence_intervals.py

# Note that for all further targets, 50 repetitions and 5 splits may require substantial simulation time.
# For a quick check, consider reducing these numbers. n_reps must be divisible by split, however, to be able to compute confidence intervals on batch means, where batches are grouped through the 'split' argument.
# Check the individual Python files for more command-line parameters: for example, simulations are saved to .json files, and plotting reads these files. Therefore, you can simulate once, and then plot in a number of ways.
comparison_loss_over_time:
	python comparison_loss_over_time.py $(ARGUMENTS) --n_channels 5 --n_neurons 128 --max_t 30000 --n_reps 50 --split 5

comparison_convergence_over_channels:
	python comparison_convergence_over_channels.py $(ARGUMENTS) --n_channels 2 3 4 5 6 7 --n_neurons 128 --max_t 250000 --n_reps 50 --split 5

fully_observable_comparison_loss_over_time_matrix_input:
	python fully_observable_comparison_loss_over_time_matrix_input.py $(ARGUMENTS) --n_channels 5 --n_neurons 128 --n_sample_length 16 --max_t 2500 --n_reps 50 --split 5

partially_observable_comparison_loss_over_time_matrix_input:
	python partially_observable_comparison_loss_over_time_matrix_input.py $(ARGUMENTS) --n_channels 5 --n_neurons 128 --n_sample_length 64 --max_t 50000 --n_reps 50 --split 5	

comparison_convergence_over_sample_length:
	python comparison_convergence_over_sample_length.py $(ARGUMENTS) --n_channels 5 --n_neurons 128 --n_sample_length 1 6 16 32 64 96 --max_t 250000 --n_reps 50 --split 5