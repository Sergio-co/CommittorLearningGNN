#!/usr/bin/env python3
# from histogram import Axis
from .histogram import HistogramScalar
from .boltzmann_constant import boltzmann_constant_kcalmolk
import argparse
from .read_colvars_traj import ReadColvarsTraj
import numpy as np
import csv
from scipy.special import logsumexp
import gzip


class GetTrajWeightEGABF:

    def __init__(self, column_names, pmf_filenames, kbt=300.0*boltzmann_constant_kcalmolk):
        import logging
        import copy
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        logging_handler = logging.StreamHandler()
        logging_formatter = logging.Formatter('[%(name)s %(levelname)s]: %(message)s')
        logging_handler.setFormatter(logging_formatter)
        self.logger.addHandler(logging_handler)
        self.logger.setLevel(logging.INFO)
        self.column_names = copy.deepcopy(column_names)
        self.weight_sum = 0.0
        self.count = 0.0
        # self.maxColumn = max(self.positionColumns)
        self.pmfs = list()
        self.kbt = kbt
        self.log_weights = list()
        self.max_sum_dG = None
        self.max_CV_dG = [0] * len(column_names)
        # self.logger.warning('The weight will not be normalized!')
        # for egABF simulations, all PMFs must be 1D, and the number of PMFs should match the number of position columns
        if len(column_names) != len(pmf_filenames):
            raise RuntimeError('The number of PMFs mismatches the number of columns')
        for pmf_filename in pmf_filenames:
            with open(pmf_filename, 'r') as f_pmf:
                self.logger.info(f'Reading {pmf_filename}')
                pmf = HistogramScalar()
                pmf.read_from_stream(f_pmf)
                if pmf.get_dimension() != 1:
                    raise RuntimeError(f'PMF should be 1D in egABF reweighting (file: {pmf_filename})')
                self.pmfs.append(copy.deepcopy(pmf))

    def accumulate_weights_sum(self, f_traj):
        total_lines = 0
        valid_lines = 0
        for line in f_traj:
            total_lines = total_lines + 1
            position_in_grid = True
            sum_delta_G = 0
            tmp_positions = list()
            for i, (pmf, column_name) in enumerate(zip(self.pmfs, self.column_names)):
                if position_in_grid:
                    pos = float(line[column_name])
                    tmp_positions.append(pos)
                    tmp_position = [pos]
                    if pmf.is_in_grid(tmp_position):
                        valid_lines += 1
                        sum_delta_G += pmf[tmp_position]
                        if self.max_CV_dG[i] is None:
                            self.max_CV_dG[i] = pmf[tmp_position]
                        else:
                            if pmf[tmp_position] > self.max_CV_dG[i]:
                                self.max_CV_dG[i] = pmf[tmp_position]
                    else:
                        sum_delta_G = 0
                        position_in_grid = False
            weight = np.exp(-1.0 * sum_delta_G / self.kbt)
            if self.max_sum_dG is None:
                self.max_sum_dG = sum_delta_G
            else:
                if sum_delta_G > self.max_sum_dG:
                    self.max_sum_dG = sum_delta_G
            if position_in_grid:
                self.weight_sum += weight
                self.count += 1.0
                self.log_weights.append(-1.0 * sum_delta_G / self.kbt)
            else:
                self.logger.warning(f'position {tmp_positions} is not in the boundary.')

    def parse_traj(self, f_traj, f_output, first_time=False, csv_writer=None):
        factor = self.count * 1.0 / self.weight_sum
        log_const = np.log(self.count) - logsumexp(a=self.log_weights)
        total_lines = 0
        valid_lines = 0
        cv_weight = dict()
        for column_name in self.column_names:
            cv_weight[f'{column_name}_weight'] = 0.0
        for line in f_traj:
            total_lines += 1
            line['weight'] = 0
            line['log_weight'] = 0
            if csv_writer is None:
                csv_writer = csv.DictWriter(f_output, fieldnames=(list(line.keys()) + list(cv_weight.keys())))
            if first_time:
                csv_writer.writeheader()
                first_time = False
            sum_delta_G = 0
            position_in_grid = True
            tmp_positions = list()
            for i, (pmf, column_name) in enumerate(zip(self.pmfs, self.column_names)):
                if position_in_grid:
                    pos = float(line[column_name])
                    tmp_positions.append(pos)
                    tmp_position = [pos]
                    if pmf.is_in_grid(tmp_position):
                        cv_weight[f'{column_name}_weight'] = np.exp(-1.0 * pmf[tmp_position] / self.kbt)
                        valid_lines += 1
                        sum_delta_G += pmf[tmp_position]
                    else:
                        cv_weight[f'{column_name}_weight'] = np.exp(-1.0 * self.max_CV_dG[i] / self.kbt)
                        sum_delta_G = 0
                        position_in_grid = False
                        continue
            if position_in_grid:
                line['weight'] = factor * np.exp(-1.0 * sum_delta_G / self.kbt)
                line['log_weight'] = -1.0 * sum_delta_G / self.kbt + log_const
            else:
                # prevent the destruction of time series
                line['weight'] = factor * np.exp(-1.0 * self.max_sum_dG / self.kbt)
                line['log_weight'] = -1.0 * self.max_sum_dG / self.kbt + log_const
                self.logger.warning(f'position {tmp_positions} is not in the boundary.')
            csv_writer.writerow({**line, **cv_weight})
        self.logger.info(f'Total data lines: {total_lines}')
        self.logger.info(f'Valid data lines: {valid_lines}')
        return first_time, csv_writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print the weights of a Colvars trajectory from an egABF simulation')
    required_args = parser.add_argument_group('required named arguments')
    required_args.add_argument('--pmfs', nargs='+', help='egABF 1D PMF file(s)', required=True)
    required_args.add_argument('--traj', nargs='+', help='Colvars trajectory files', required=True)
    required_args.add_argument('--columns', type=str, nargs='+', help='the columns in the trajectory matching the CVs '
                                                                      'of the PMF', required=True)
    required_args.add_argument('--output', help='the output file with weights', required=True)
    parser.add_argument('--kbt', default=300.0*boltzmann_constant_kcalmolk, type=float, help='KbT')
    args = parser.parse_args()
    get_weight_traj = GetTrajWeightEGABF(args.columns, args.pmfs, args.kbt)
    first_time = True
    csv_writer = None
    for traj_file in args.traj:
        with ReadColvarsTraj(traj_file) as f_traj:
            get_weight_traj.accumulate_weights_sum(f_traj)
    with gzip.open(args.output, 'wt') as f_output:
        for traj_file in args.traj:
            with ReadColvarsTraj(traj_file) as f_traj:
                first_time, csv_writer = get_weight_traj.parse_traj(f_traj, f_output, first_time, csv_writer)