# Histogram utilities

This package contains several python classes and scripts for building histograms:
* `boltzmann_constant.py`: The Boltzmann and some other commonly used constants.
* `histogram.py`: Main histogram classes: `Axis`, `HistogramScalar`, `HistogramVector` and `HistogramFiles`.
* `detect_boundary.py`: Boundary detection using the method in https://doi.org/10.1002/jcc.25520.
* `print_weight.py`: Reweighting using a multidimensional PMF.
* `print_weight_egabf.py`: Reweighting using multiple one-dimensional PMFs.
* `plot_colvars_traj.py`: From [Colvars](https://github.com/Colvars/colvars/) for loading colvars traj files.
* `read_colvars_traj.py`: Read colvars traj with context manager and line by line.
* `reweight.py`: Example reweighting script.
* `build_histogram_from_traj.py`: Example script of histogramming data from colvars trajs.
