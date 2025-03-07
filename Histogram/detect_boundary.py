#!/usr/bin/env python3
from .histogram import HistogramScalar
import argparse
import numpy as np


class DetectBoundary:

    def __init__(self, hist):
        import copy
        import numpy as np
        self.histogramP = copy.deepcopy(hist)
        self.histogramP.data = -1.0 * np.exp(-1.0 * np.clip(self.histogramP.get_data(), 0, None))
        self.histogramV = HistogramScalar(self.histogramP.get_axes())
        point_table_transposed = self.histogramV.pointTable.T
        for pos in point_table_transposed:
            neighbor_points = self.histogramP.all_neighbor(pos)
            ki = 0.0
            sum_p = 0.0
            for result in neighbor_points:
                if result[1] is True:
                    ki += 1.0
                    sum_p += self.histogramP[result[0]]
            if ki > 0:
                self.histogramV[pos] = self.histogramP[pos] - sum_p / ki

    def get_histogram_p(self):
        return self.histogramP

    def get_histogram_v(self):
        return self.histogramV


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect the boundary of a histogram')
    required_args = parser.add_argument_group('required named arguments')
    required_args.add_argument('--hist', help='the PMF file', required=True)
    required_args.add_argument('--output', help='the output file with weights', required=True)
    args = parser.parse_args()
    hist_input = HistogramScalar()
    with open(args.hist, 'r') as f_hist:
        hist_input.read_from_stream(f_hist)
    detect_boundary = DetectBoundary(hist_input)
    with open(args.output + '.histP', 'w') as f_output_p, \
         open(args.output + '.grid', 'w') as f_output_colvars_factor_grid:
        detect_boundary.get_histogram_p().write_to_stream(f_output_p)
        import copy
        colvars_force_factor_grid = copy.deepcopy(hist_input)
        boundary_abs = np.abs(detect_boundary.get_histogram_v().data)
        colvars_force_factor_grid.data = np.sign(colvars_force_factor_grid.data + boundary_abs)
        colvars_force_factor_grid.write_to_stream(f_output_colvars_factor_grid)
    with open(args.output + '.histV', 'w') as f_output_v:
        detect_boundary.get_histogram_v().write_to_stream(f_output_v)
