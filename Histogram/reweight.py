#!/usr/bin/env python3
from .histogram import HistogramScalar
import argparse


def convert_pmf_to_probability(hist_pmf, kbT):
    import copy
    import numpy as np
    hist_prob = copy.deepcopy(hist_pmf)
    hist_prob.data = np.exp(-1.0 / kbT * hist_prob.data)
    return hist_prob


def convert_probability_to_pmf(hist_prob, kbT):
    import copy
    import numpy as np
    from math import isclose
    hist_pmf = copy.deepcopy(hist_prob)
    norm_factor = np.sum(hist_prob.data)
    max_prob = np.max(hist_prob.data)
    min_pmf = 0.0
    if max_prob > 0:
        min_pmf = -1.0 * kbT * np.log(max_prob / norm_factor)
    if norm_factor > 0:
        for i in range(hist_pmf.get_histogram_size()):
            if hist_pmf[i] > 0:
                hist_pmf[i] = -1.0 * kbT * np.log(hist_pmf[i] / norm_factor) - min_pmf
        max_pmf = np.max(hist_pmf.data)
        for i in range(hist_pmf.get_histogram_size()):
            if isclose(hist_prob[i], 0):
                hist_pmf[i] = max_pmf
    return hist_pmf


def reweighting(f_traj, prob_origin, prob_target, from_columns, to_columns, pbar=None):
    for line in f_traj:
        if pbar is not None:
            pbar.update(len(line))
        tmp_fields = line.split()
        if len(tmp_fields) > 0 and (not tmp_fields[0].startswith('#')):
            source_position = [float(tmp_fields[i]) for i in from_columns]
            target_position = [float(tmp_fields[i]) for i in to_columns]
            if prob_origin.is_in_grid(source_position) and prob_target.is_in_grid(target_position):
                weight = prob_origin[source_position]
                prob_target[target_position] += weight
    return prob_target


if __name__ == '__main__':
    import os
    import tqdm
    from .boltzmann_constant import boltzmann_constant_kcalmolk
    parser = argparse.ArgumentParser(description='Calculate PMF along CVs from traj by reweighting PMF')
    required_args = parser.add_argument_group('required named arguments')
    required_args.add_argument('--pmf', help='the PMF file', required=True)
    required_args.add_argument('--traj', nargs='+', help='the Colvars trajectory file', required=True)
    required_args.add_argument('--from_columns', type=int, nargs='+',
                               help='the source columns in the trajectory', required=True)
    required_args.add_argument('--to_columns', type=int, nargs='+',
                               help='the target columns in the trajectory', required=True)
    required_args.add_argument('--axis', help='json file to setup axes')
    required_args.add_argument('--output', help='the output file with weights', required=True)
    parser.add_argument('--kbt', default=300.0*boltzmann_constant_kcalmolk, type=float, help='KbT in kcal/mol')
    args = parser.parse_args()
    with open(args.pmf, 'r') as f_pmf:
        hist_pmf = HistogramScalar()
        hist_pmf.read_from_stream(f_pmf)
        hist_prob_source = convert_pmf_to_probability(hist_pmf, args.kbt)
        hist_prob_target = HistogramScalar.from_json_file(args.axis)
        for traj_file in args.traj:
            with tqdm.tqdm(total=os.path.getsize(traj_file), mininterval=1.0) as pbar:
                pbar.set_description(f'Processing file {traj_file}')
                with open(traj_file, 'r') as f_traj:
                    hist_prob_target = reweighting(f_traj=f_traj, prob_origin=hist_prob_source, prob_target=hist_prob_target,
                                                   from_columns=args.from_columns, to_columns=args.to_columns,
                                                   pbar=pbar)
        with open(args.output, 'w') as f_output:
            # convert to PMF
            hist_pmf_target = convert_probability_to_pmf(hist_prob_target, args.kbt)
            hist_pmf_target.write_to_stream(f_output, data_fmt='15.10f')
            # TODO: test reweighting
