#!/usr/bin/env python3
# from histogram import Axis
import math
from .histogram import HistogramScalar
from .boltzmann_constant import boltzmann_constant_kcalmolk
import argparse
from .read_colvars_traj import ReadColvarsTraj
import csv
from scipy.special import logsumexp
import gzip


class GetTrajWeight:

    def __init__(self, column_names, pmf_filename, kbt=300.0*boltzmann_constant_kcalmolk):
        import numpy as np
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
        self.probability = None
        self.weight_sum = 0
        self.count = 0
        self.log_weights = []
        with open(pmf_filename, 'r') as f_pmf:
            pmf = HistogramScalar()
            pmf.read_from_stream(f_pmf)
            pmf.data = -1.0 * pmf.data / kbt
            self.log_probability = pmf
            self.probability = copy.deepcopy(pmf)
            self.probability.data = np.exp(self.probability.data)
        if self.probability.get_dimension() != len(self.column_names):
            self.logger.warning('the number of columns selected does not match the dimension of the PMF!')
        self.logger.info(f'Get weights from PMF file {pmf_filename}')
        self.logger.info(f'Will read columns: {self.column_names}')

    def accumulate_weights_sum(self, f_traj):
        total_lines = 0
        valid_lines = 0
        for line in f_traj:
            total_lines = total_lines + 1
            tmp_position = [line[i] for i in self.column_names]
            # check if the position is in boundary
            if self.probability.is_in_grid(tmp_position):
                valid_lines = valid_lines + 1
                weight = self.probability[tmp_position]
                self.log_weights.append(self.log_probability[tmp_position])
                self.weight_sum += weight
                self.count += 1.0
                # f_output.write(' '.join(tmp_fields) + f' {weight:22.15e}\n')
            else:
                self.logger.warning(f'position {tmp_position} is not in the boundary.')
        self.logger.info(f'(Accumulate weights) Total data lines: {total_lines}')
        self.logger.info(f'(Accumulate weights) Valid data lines: {valid_lines}')
        self.logger.info(f'(Accumulate weights) Total weights: {self.weight_sum}')

    def parse_traj(self, f_traj, f_output, first_time=False, csv_writer=None):
        total_lines = 0
        valid_lines = 0
        try:
            factor = self.count * 1.0 / self.weight_sum
        except ZeroDivisionError:
            print('Warning: weight sum is 0, please running accumulate_weights_sum at first!')
            print('Continue with factor = 1.0')
            factor = 1.0
            self.weight_sum = self.count
        log_const = math.log(self.count) - logsumexp(a=self.log_weights)
        for line in f_traj:
            line['weight'] = 0
            line['log_weight'] = 0
            if csv_writer is None:
                csv_writer = csv.DictWriter(f_output, fieldnames=line.keys())
            if first_time:
                csv_writer.writeheader()
                first_time = False
            total_lines = total_lines + 1
            tmp_position = [line[i] for i in self.column_names]
            if self.probability.is_in_grid(tmp_position):
                w = self.probability[tmp_position] * factor
                if math.isnan(w):
                    w = 1.0
                line['weight'] = w
                line['log_weight'] = self.log_probability[tmp_position] + log_const
                valid_lines = valid_lines + 1
                # f_output.write(f_traj.current_str().rstrip('\n') + f' {factor * weight:22.15e}\n')
            else:
                line['weight'] = min(self.probability) * factor
                line['log_weight'] = max(self.log_probability) + log_const
                self.logger.warning(f'position {tmp_position} is not in the boundary.')
            csv_writer.writerow(line)
        self.logger.info(f'(parse_traj) Total data lines: {total_lines}')
        self.logger.info(f'(parse_traj) Valid data lines: {valid_lines}')
        return first_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print the weights of a Colvars trajectory')
    required_args = parser.add_argument_group('required named arguments')
    required_args.add_argument('--pmf', help='the PMF file', required=True)
    required_args.add_argument('--traj', nargs='+', help='the Colvars trajectory file', required=True)
    required_args.add_argument('--columns', type=str, nargs='+', help='the columns in the trajectory'
                                                                      ' matching the CVs of the PMF', required=True)
    required_args.add_argument('--output', help='the output file with weights', required=True)
    parser.add_argument('--kbt', default=300.0*boltzmann_constant_kcalmolk, type=float, help='KbT')
    args = parser.parse_args()
    # all the arguments are mandatory
    get_weight_traj = GetTrajWeight(args.columns, args.pmf, args.kbt)
    for traj_file in args.traj:
        with ReadColvarsTraj(traj_file) as f_traj:
            get_weight_traj.accumulate_weights_sum(f_traj)
    first_time = True
    with gzip.open(args.output, 'wt') as f_output:
        for traj_file in args.traj:
            with ReadColvarsTraj(traj_file) as f_traj:
                first_time = get_weight_traj.parse_traj(f_traj, f_output, first_time)
