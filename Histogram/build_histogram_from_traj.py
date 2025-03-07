#!/usr/bin/env python3
from .read_colvars_traj import ReadSpaceSeparatedTraj, ReadColvarsTraj


class BuildHistogramFromTraj:

    def __init__(self, json_file, position_cols):
        import copy
        import logging
        from .histogram import HistogramScalar
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        logging_handler = logging.StreamHandler()
        logging_formatter = logging.Formatter('[%(name)s %(levelname)s]: %(message)s')
        logging_handler.setFormatter(logging_formatter)
        self.logger.addHandler(logging_handler)
        self.logger.setLevel(logging.INFO)
        self.histogram = HistogramScalar.from_json_file(json_file)
        if position_cols is None:
            self.positionColumns = list(range(0, self.histogram.get_dimension()))
        else:
            self.positionColumns = copy.deepcopy(position_cols)
        # self.maxColumn = max(self.positionColumns)

    def read_traj(self, f_traj):
        total_lines = 0
        valid_lines = 0
        for line in f_traj:
            total_lines += 1
            tmp_position = [line[key] for key in self.positionColumns]
            if self.histogram.is_in_grid(tmp_position):
                self.histogram[tmp_position] += 1.0
                valid_lines += 1
            else:
                self.logger.warning(f'Position {tmp_position} is not in the boundary!')
        self.logger.info(f'Total data lines: {total_lines}')
        self.logger.info(f'Valid data lines: {valid_lines}')

    def read_pandas(self, df):
        total_lines = 0
        valid_lines = 0
        np_data = df[self.positionColumns].to_numpy()
        for row in range(0, df.shape[0]):
            total_lines += 1
            tmp_position = np_data[row]
            if self.histogram.is_in_grid(tmp_position):
                self.histogram[tmp_position] += 1.0
                valid_lines += 1
            else:
                self.logger.warning(f'Position {tmp_position} is not in the boundary!')
        self.logger.info(f'Total data lines: {total_lines}')
        self.logger.info(f'Valid data lines: {valid_lines}')

    def get_histogram(self):
        return self.histogram


if __name__ == '__main__':
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description='Build histogram from colvars trajectories')
    required_args = parser.add_argument_group('required named arguments')
    required_args.add_argument('--axis', help='json file to setup axes')
    parser.add_argument('--traj', nargs='*', help='Colvars trajectory files')
    parser.add_argument('--csv', nargs='*', help='CSV traj files')
    required_args.add_argument('--output', help='the output file with weights', required=True)
    parser.add_argument('--columns', type=str, nargs='+', help='columns in the trajectory (default to 0...Ndim)')
    args = parser.parse_args()
    build_histogram = BuildHistogramFromTraj(json_file=args.axis, position_cols=args.columns)
    if args.columns is None:
        if args.traj is not None:
            for traj_file in args.traj:
                with ReadSpaceSeparatedTraj(traj_file) as f_traj:
                    build_histogram.read_traj(f_traj=f_traj)
    else:
        if args.traj is not None:
            for traj_file in args.traj:
                with ReadColvarsTraj(traj_file) as f_traj:
                    build_histogram.read_traj(f_traj=f_traj)
    if args.csv is not None:
        for csv_file in args.csv:
            df = pd.read_csv(csv_file)
            build_histogram.read_pandas(df)
    with open(args.output, 'w') as f_output:
        build_histogram.get_histogram().write_to_stream(stream=f_output)
