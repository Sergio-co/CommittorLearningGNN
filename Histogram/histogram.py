#!/usr/bin/env python3

class Axis:

    def __init__(self, lower_bound=0.0, upper_bound=0.0, bins=0, periodic=False):
        self.lowerBound = lower_bound
        self.upperBound = upper_bound
        self.bins = bins
        self.periodic = periodic
        if bins != 0:
            self.width = (self.upperBound - self.lowerBound) / float(self.bins)
        else:
            self.width = 0
        self.periodicLowerBound = self.lowerBound
        self.periodicUpperBound = self.upperBound

    def set_periodicity(self, periodic, periodic_lower, periodic_upper):
        self.periodic = periodic
        self.periodicUpperBound = periodic_upper
        self.periodicLowerBound = periodic_lower

    def get_lower_bound(self):
        return self.lowerBound

    def set_lower_bound(self, new_lower_bound):
        if self.width > 0:
            self.bins = round((self.upperBound - new_lower_bound) / self.width)
        else:
            self.bins = 0
        if self.bins == 0:
            self.lowerBound = new_lower_bound
        else:
            self.lowerBound = self.upperBound - float(self.bins) * self.width

    def get_upper_bound(self):
        return self.upperBound

    def set_upper_bound(self, new_upper_bound):
        if self.width > 0:
            self.bins = round((new_upper_bound - self.lowerBound) / self.width)
        else:
            self.bins = 0
        if self.bins == 0:
            self.upperBound = new_upper_bound
        else:
            self.upperBound = self.lowerBound + float(self.bins) * self.width

    def get_width(self):
        return self.width

    def set_width(self, new_width):
        self.bins = int(round((self.upperBound - self.lowerBound) / new_width))
        if self.bins == 0:
            self.width = new_width
        else:
            self.width = (self.upperBound - self.lowerBound) / float(self.bins)

    def get_bin(self):
        return self.bins

    def set_bin(self, new_bin):
        self.bins = new_bin
        self.width = (self.upperBound - self.lowerBound) / float(self.bins)

    def in_boundary(self, x):
        x = self.wrap(x)
        if (x < self.lowerBound) or (x > self.upperBound):
            return False
        else:
            return True

    def wrap(self, x):
        from math import fabs, isclose
        if not self.periodic:
            return x
        if (x >= self.periodicLowerBound) and (x <= self.periodicUpperBound):
            return x
        periodicity = self.periodicUpperBound - self.periodicLowerBound
        if x < self.periodicLowerBound:
            dist_to_lower = self.periodicLowerBound - x
            num_period_add = int(dist_to_lower / periodicity)
            tmp = fabs(dist_to_lower / periodicity - round(dist_to_lower / periodicity))
            if isclose(tmp, 0.0):
                x += num_period_add * periodicity
            else:
                x += (num_period_add + 1) * periodicity
        if x > self.periodicUpperBound:
            dist_to_upper = x - self.periodicUpperBound
            num_period_subtract = int(dist_to_upper / periodicity)
            tmp = fabs(dist_to_upper / periodicity - round(dist_to_upper / periodicity))
            if isclose(tmp, 0.0):
                x -= num_period_subtract * periodicity
            else:
                x -= (num_period_subtract + 1) * periodicity
        return x

    def index(self, x, boundary_check=False):
        from math import floor
        x = self.wrap(x)
        check_result = True
        if boundary_check is True:
            check_result = self.in_boundary(x)
        if check_result is False:
            return 0, check_result
        idx = int(floor((x - self.lowerBound) / self.width))
        if idx == self.bins:
            idx = idx - 1
        return idx, check_result

    def dist(self, x, reference):
        from math import fabs
        if not self.periodic:
            return x - reference
        else:
            x = self.wrap(x)
            reference = self.wrap(reference)
            dist = x - reference
            p = self.period()
            if fabs(dist) > (p * 0.5):
                if reference > x:
                    return dist + p
                elif reference < x:
                    return dist - p
                else:
                    return dist
            else:
                return dist

    def period(self):
        return self.periodicUpperBound - self.periodicLowerBound

    def get_middle_points(self):
        result = [self.lowerBound + (i + 0.5) * self.width for i in range(0, self.bins)]
        return result

    def info_header(self):
        pbc = 0
        if self.periodic is True:
            pbc = 1
        return f'# {self.lowerBound:.9f} {self.width:.9f} {self.bins:d} {pbc}'

    def __str__(self) -> str:
        s = f'boundary: [{self.lowerBound}, {self.upperBound}] ; width: {self.width} ; PBC: {self.periodic}'
        return s

def create_axis_from_dict(json_dict):
    lb = json_dict['Lower bound']
    ub = json_dict['Upper bound']
    if 'Width' not in json_dict:
        if 'Bins' not in json_dict:
            bins = 0
            raise RuntimeError('Cannot find "Width" or "Bins"')
        else:
            bins = json_dict['Bins']
    else:
        width = json_dict['Width']
        bins = round(float(ub - lb) / width)
    periodic = False
    if 'Periodic' in json_dict:
        periodic = json_dict['Periodic']
    ax = Axis(lower_bound=lb, upper_bound=ub, bins=bins, periodic=periodic)
    return ax


class HistogramBase:

    def __init__(self, ax=None):
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        logging_handler = logging.StreamHandler()
        logging_formatter = logging.Formatter('[%(name)s %(levelname)s]: %(message)s')
        logging_handler.setFormatter(logging_formatter)
        self.logger.addHandler(logging_handler)
        self.logger.setLevel(logging.INFO)
        if ax is None:
            import numpy as np
            self.ndim = 0
            self.histogramSize = 0
            self.axes = list()
            self.pointTable = np.array(list(list()))
            self.accu = list()
        else:
            self.ndim = len(ax)
            self.axes = ax.copy()
            self.accu = [0] * self.ndim
            self._real_init()

    def _real_init(self):
        if self.ndim == 0:
            return
        self.histogramSize = 1
        for i in range(0, self.ndim):
            if i == 0:
                self.accu[i] = 1
            else:
                self.accu[i] = self.accu[i - 1] * self.axes[i - 1].get_bin()
            self.histogramSize *= self.axes[i].get_bin()
        self.fill_table()
        for s in str(self).split('\n'):
            self.logger.info(s)

    def __str__(self) -> str:
        s = f'histogram size: {self.histogramSize}\n'
        s += f'histogram dimension: {self.ndim}\n'
        s += f'axes:\n'
        s += '\n'.join([str(ax) for ax in self.axes])
        return s

    def fill_table(self):
        import numpy as np
        middle_points = list()
        for i in range(0, self.ndim):
            middle_points.append(self.axes[i].get_middle_points())
        # python list are reference
        self.pointTable = np.zeros((self.get_dimension(), self.get_histogram_size()))
        for i in range(0, self.ndim):
            repeat_all = 1
            repeat_one = 1
            for j in range(i + 1, self.ndim):
                repeat_one *= len(middle_points[j])
            for j in range(0, i):
                repeat_all *= len(middle_points[j])
            in_i_sz = len(middle_points[i])
            for l in range(0, in_i_sz):
                for m in range(0, repeat_one):
                    (self.pointTable[i])[m+l*repeat_one] = middle_points[i][l]
            # print(self.pointTable[i])
            # print(f'i = {i}, repeat_all = {repeat_all}, repeat_one = {repeat_one}')
            for k in range(0, repeat_all - 1):
                count = repeat_one * in_i_sz
                # if count > 0:
                for m in range(0, count):
                    self.pointTable[i][m + count * (k + 1)] = self.pointTable[i][m]
            # print(len(self.pointTable[0]))

    def read_from_stream(self, stream):
        line = stream.readline()
        tmp = line.split()
        if len(tmp) < 2:
            return False
        self.ndim = int(float(tmp[1]))
        self.accu = [0] * self.ndim
        for i in range(0, self.ndim):
            line = stream.readline()
            tmp = line.split()
            if len(tmp) < 5:
                return False
            lower_bound = float(tmp[1])
            width = float(tmp[2])
            num_bins = int(tmp[3])
            upper_bound = float(lower_bound + width * num_bins)
            pbc = True
            if int(tmp[4]) == 0:
                pbc = False
            ax = Axis(lower_bound=lower_bound, upper_bound=upper_bound,
                      bins=num_bins, periodic=pbc)
            if pbc is True:
                ax.set_periodicity(periodic=pbc, periodic_lower=lower_bound,
                                   periodic_upper=upper_bound)
            self.axes.append(ax)
        self._real_init()
        return True

    def write_to_stream(self, stream):
        stream.write(f'# {self.ndim}\n')
        for ax in self.axes:
            stream.write(ax.info_header()+'\n')

    def is_in_grid(self, pos):
        for val, ax in zip(pos, self.axes):
            if ax.in_boundary(val) is False:
                return False
        return True

    def index(self, pos, boundary_check=False):
        idx = list()
        check = list()
        for val, ax in zip(pos, self.axes):
            i, c = ax.index(val, boundary_check)
            idx.append(i)
            check.append(c)
            if c is False:
                self.logger.warning(f'position {pos} is not in boundary!')
        return idx, check

    def address(self, pos, boundary_check=False):
        addr = 0
        for val, ax, ac in zip(pos, self.axes, self.accu):
            i, c = ax.index(val, boundary_check)
            addr += ac * i
            if c is False and boundary_check is True:
                self.logger.warning(f'position {pos} is not in boundary!')
        return addr

    def reverse_address(self, address):
        center_pos = list()
        if address < 0 or address >= self.histogramSize:
            return [0] * self.ndim, False
        else:
            for ax_r, ac_r in zip(reversed(self.axes), reversed(self.accu)):
                id_i = int(float(address) / ac_r)
                val = ax_r.get_lower_bound() + (0.5 + id_i) * ax_r.get_width()
                center_pos.append(val)
                address = address - id_i * ac_r
        return list(reversed(center_pos)), True

    def get_histogram_size(self):
        return self.histogramSize

    def get_dimension(self):
        return self.ndim

    def get_axes(self):
        return self.axes.copy()

    def get_point_table(self):
        return self.pointTable.copy()

    def get_bin_size(self):
        sz = 1.0
        for ax in self.axes:
            sz *= ax.width()
        return sz

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, key):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def neighbor(self, pos, axis_index, previous=False):
        import copy
        bin_width_i = self.axes[axis_index].get_width()
        pos_next = list(copy.deepcopy(pos))
        if previous is True:
            pos_next[axis_index] = pos_next[axis_index] - bin_width_i
        else:
            pos_next[axis_index] = pos_next[axis_index] + bin_width_i
        in_boundary = self.is_in_grid(pos_next)
        return self.address(pos_next, False), in_boundary

    def neighbor_by_address(self, addr, axis_index, previous=False):
        if addr >= self.histogramSize:
            return 0, False
        pos = self.reverse_address(addr)
        return self.neighbor(pos, axis_index, previous)

    def all_neighbor(self, pos):
        results = list()
        for i in range(self.ndim):
            results.append(self.neighbor(pos, i, True))
            results.append(self.neighbor(pos, i, False))
        return results

    def all_neighbor_by_address(self, addr):
        results = list()
        for i in range(self.ndim):
            results.append(self.neighbor_by_address(addr, i, True))
            results.append(self.neighbor_by_address(addr, i, False))
        return results

    def write_to_dx(self, stream):
        axes_bins = [str(ax.get_bin()) for ax in self.axes]
        axes_origins = [str(ax.get_lower_bound() + 0.5 * ax.get_width()) for ax in self.axes]
        stream.write(f'object 1 class gridpositions counts {" ".join(axes_bins)}\n')
        stream.write(f'origin {" ".join(axes_origins)}\n')
        ax_widths = [ax.get_width() for ax in self.axes]
        for i in range(0, len(ax_widths)):
            stream.write('delta ')
            for j in range(0, len(ax_widths)):
                if i == j:
                    stream.write(f' {ax_widths[i]}')
                else:
                    stream.write(' 0')
            stream.write('\n')
        stream.write(f'object 2 class gridconnections counts {" ".join(axes_bins)}\n')
        stream.write(f'object 3 class array type double rank 0 items {len(self)} data follows')
        pos = [0] * self.get_dimension()
        write_count = 0
        for i in range(0, self.get_histogram_size()):
            for j in range(0, self.get_dimension()):
                pos[j] = self.pointTable[j][i]
            if write_count % 3 == 0:
                stream.write('\n')
            write_count += 1
            addr = self.address(pos, False)
            stream.write(f' {self[addr]}')
        stream.write('\nobject "collective variables scalar field" class field\n')


class HistogramScalar(HistogramBase):

    def __init__(self, ax=None):
        import numpy as np
        super().__init__(ax)
        self.data = np.zeros(self.get_histogram_size())

    @staticmethod
    def from_json_file(json_file):
        import json
        with open(json_file, 'r') as f_json:
            all_data = f_json.read()
            json_dict = json.loads(all_data)
            ax_list = list()
            for ax in json_dict['Axes']:
                ax_list.append(create_axis_from_dict(ax))
            return HistogramScalar(ax=ax_list)

    def __len__(self):
        return self.get_histogram_size()

    def read_from_stream(self, stream):
        import numpy as np
        read_status = HistogramBase.read_from_stream(self, stream)
        if not read_status:
            return False
        self.data = np.zeros(self.get_histogram_size())
        data_line = 0
        for line in stream:
            tmp_fields = line.split()
            if len(tmp_fields) == self.get_dimension() + 1:
                if tmp_fields[0].startswith('#'):
                    continue
                data_line += 1
                pos = list(map(float, tmp_fields[0:self.get_dimension()]))
                addr = self.address(pos, True)
                self.data[addr] = float(tmp_fields[self.get_dimension()])
        self.logger.info(f'Expect {self.get_histogram_size()} lines, read {data_line} lines.')
        return True

    def __getitem__(self, key):
        try:
            iter(key)
            if self.is_in_grid(key):
                addr = self.address(key, False)
                return self.data[addr]
            else:
                raise IndexError(f'Index {key} is out of bound.')
        except TypeError:
            return self.data[key]

    def __setitem__(self, key, value):
        try:
            iter(key)
            if self.is_in_grid(key):
                addr = self.address(key, False)
                self.data[addr] = value
                return self.data[addr]
            else:
                raise IndexError(f'Index {key} is out of bound.')
        except TypeError:
            self.data[key] = value
            return self.data[key]

    def get_data(self, copy=False):
        if copy is True:
            import numpy as np
            return np.copy(self.data)
        else:
            return self.data

    def write_to_stream(self, stream, pos_fmt='12.7f', data_fmt='15.7f'):
        write_ok = HistogramBase.write_to_stream(self, stream)
        if write_ok is False:
            return False
        pos = [0] * self.get_dimension()
        for i in range(0, self.get_histogram_size()):
            for j in range(0, self.get_dimension()):
                pos[j] = self.pointTable[j][i]
                stream.write(f' {pos[j]:{pos_fmt}}')
            addr = self.address(pos, False)
            stream.write(f' {self.data[addr]:{data_fmt}}\n')
        return True


class HistogramVector(HistogramBase):

    def __init__(self, ax=None, multiplicity=0):
        import numpy as np
        super().__init__(ax)
        self.multiplicity = multiplicity
        self.data = np.zeros(self.get_histogram_size() * self.get_multiplicity())

    def get_multiplicity(self):
        return self.multiplicity

    def __getitem__(self, key):
        try:
            iter(key)
            if self.is_in_grid(key):
                addr = self.address(key, False)
                return self.data[addr:addr+self.get_multiplicity()]
            else:
                raise IndexError(f'Index {key} is out of bound.')
        except TypeError:
            return self.data[key]

    def __setitem__(self, key, value):
        try:
            iter(key)
            if self.is_in_grid(key):
                addr = self.address(key, False)
                for i in range(self.get_multiplicity()):
                    self.data[addr+i] = value[i]
            else:
                raise IndexError(f'Index {key} is out of bound.')
        except TypeError:
            self.data[key] = value

    def get_data(self, copy=False):
        if copy is True:
            import numpy as np
            return np.copy(self.data)
        else:
            return self.data

    def write_to_stream(self, stream, pos_fmt='12.7f', data_fmt='15.7f'):
        write_ok = HistogramBase.write_to_stream(self, stream)
        if write_ok is False:
            return False
        pos = [0] * self.get_dimension()
        for i in range(0, self.get_histogram_size()):
            for j in range(0, self.get_dimension()):
                pos[j] = self.pointTable[j][i]
                stream.write(f' {pos[j]:{pos_fmt}}')
            addr = self.address(pos, False)
            for j in range(0, self.get_multiplicity()):
                stream.write(f' {self.data[addr+j]:{data_fmt}}')
            stream.write('\n')
        return True

    def read_from_stream(self, stream, multiplicity=0):
        import numpy as np
        read_status = HistogramBase.read_from_stream(self, stream)
        if not read_status:
            return False
        if multiplicity > 0:
            self.multiplicity = multiplicity
        else:
            self.multiplicity = self.ndim
        self.data = np.zeros(self.get_histogram_size() * self.get_multiplicity())
        data_line = 0
        for line in stream:
            tmp_fields = line.split()
            if len(tmp_fields) == self.get_dimension() + 1:
                if tmp_fields[0].startswith('#'):
                    continue
                data_line += 1
                pos = list(map(float, tmp_fields[0:self.get_dimension()]))
                addr = self.address(pos, True)
                # self.data[addr] = float(tmp_fields[self.get_dimension()])
                for j in range(0, self.get_multiplicity()):
                    self.data[addr * self.get_multiplicity() + j] = float(tmp_fields[self.get_dimension() + j])
        self.logger.info(f'Expect {self.get_histogram_size()} lines, read {data_line} lines.')
        return True

    def __len__(self):
        return len(self.data)


class HistogramFiles(HistogramBase):

    # Possible bug: opening too many files
    def __init__(self, prefix, ax=None):
        super().__init__(ax)
        N = self.get_histogram_size()
        num_digits = len(str(N - 1))
        self.files = [open(f'{prefix}_{i:0{num_digits}d}.dat', 'w') for i in range(N)]

    def __exit__(self, exc_type, exc_value, exc_tb):
        for file in self.files:
            file.close()

    def __len__(self):
        return self.get_histogram_size()

    def __getitem__(self, key):
        try:
            iter(key)
            if self.is_in_grid(key):
                addr = self.address(key, False)
                return self.files[addr]
            else:
                raise IndexError(f'Index {key} is out of bound.')
        except TypeError:
            return self.files[key]

    def __setitem__(self, key, value):
        raise NotImplementedError


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Histogram test')
    parser.add_argument('--test1', action='store_true', help='run test1')
    parser.add_argument('--test2', action='store_true', help='run test2')
    parser.add_argument('--test3', action='store_true', help='run test3')
    args = parser.parse_args()
    def test1():
        ax1 = Axis(-180.0, 180.0, 10, True)
        ax2 = Axis(-180.0, 180.0, 10, True)
        hist = HistogramScalar([ax1, ax2])

        for x, y in zip(hist.pointTable[0], hist.pointTable[1]):
            print(f'{x} {y}')

        pos = (182, -90)
        idx = hist.index(pos, True)
        print(idx)
        print(hist.address(pos))
        print(hist.reverse_address(hist.address(pos)))
        print(hist.all_neighbor(pos))
        import sys
        hist.write_to_stream(sys.stdout)


    def test2():
        with open('qt_par_test_7.pmf', 'r') as f_test_pmf:
            hist = HistogramScalar()
            hist.read_from_stream(f_test_pmf)
            with open('test_py.pmf', 'w') as f_out:
                hist.write_to_stream(f_out)
        import pandas as pd
        import numpy as np
        # compute error
        origin = pd.read_csv('qt_par_test_7.pmf', delimiter=r'\s+', comment='#', header=None)
        output = pd.read_csv('test_py.pmf', delimiter=r'\s+', comment='#', header=None)
        diff = (origin[2] - output[2]).to_numpy()
        total_error = np.sqrt(np.sum(diff * diff))
        print(f'Total error = {total_error}')
        return hist


    def test3():
        hist = HistogramScalar.from_json_file('test_files/axis_encoded.json')
        from sys import stdout
        hist.write_to_stream(stdout)

    if args.test1 is True:
        test1()
    if args.test2 is True:
        test2()
    if args.test3 is True:
        test3()
    pass
