#!/usr/bin/env python3

class ReadSpaceSeparatedTraj:

    def __init__(self, filename):
        # from collections import OrderedDict
        self.f_traj = open(filename, 'r')
        self.parsed_line = dict()
        self._fields = list()
        self._line = ''
        self._start = {}
        self._end = {}
        self._eof = False

    def _parse_comment_line(self):
        pass

    def _parse_data_line(self):
        data = map(float, self._line.split())
        for i, x in enumerate(data):
            self.parsed_line[i] = x

    def _read_line(self):
        self._line = self.f_traj.readline()
        if self._line == '':
            self._eof = True
            return
        while self._line.strip() == '':
            self._line = self.f_traj.readline()
        while self._line[0] == '#':
            self._parse_comment_line()
            self._line = self.f_traj.readline()
        self._parse_data_line()

    def current_str(self):
        return self._line

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __next__(self):
        if self._eof is False:
            self._read_line()
            if self._eof is False:
                return self.parsed_line
            else:
                raise StopIteration

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.f_traj.close()


class ReadColvarsTraj:

    def __init__(self, filename):
        self.f_traj = open(filename, 'r')
        self.parsed_line = dict()
        self._fields = list()
        self._line = ''
        self._start = {}
        self._end = {}
        self._eof = False

    def _parse_comment_line(self):
        self.parsed_line = dict()
        self._fields = self._line[1:].split()
        if self._fields[0] != 'step':
            raise KeyError("Error: file format incompatible with colvars.traj")
        for i in range(1, len(self._fields)):
            if i == 1:
                pos = self._line.find(' ' + self._fields[i] + ' ')
            else:
                pos = self._line.find(' ' + self._fields[i],
                                      self._start[self._fields[i-1]] +
                                      len(self._fields[i-1]))
            self._start[self._fields[i]] = pos
            self._end[self._fields[i-1]] = pos
        self._end[self._fields[-1]] = -1

    def _parse_data_line(self):
        step = int(self._line[0:self._end['step']])
        self.parsed_line[self._fields[0]] = step
        for v in self._fields[1:]:
            text = self._line[self._start[v]:self._end[v]].strip()
            if text[0] == '(':
                v_v = list(map(float, text[1:-1].split(',')))
            else:
                v_v = float(text)
            self.parsed_line[v] = v_v

    def _read_line(self):
        self._line = self.f_traj.readline()
        if self._line == '':
            self._eof = True
            return
        while self._line.strip() == '':
            self._line = self.f_traj.readline()
        while self._line[0] == '#':
            self._parse_comment_line()
            self._line = self.f_traj.readline()
        self._parse_data_line()

    def current_str(self):
        return self._line

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __next__(self):
        if self._eof is False:
            self._read_line()
            if self._eof is False:
                return self.parsed_line
            else:
                raise StopIteration

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.f_traj.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reading a colvars traj file line by line test')
    parser.add_argument('--test1', action='store_true', help='run test1')
    args = parser.parse_args()
    if args.test1 is True:
        with ReadColvarsTraj('test.colvars.traj') as f_traj:
            for line in f_traj:
                if 'E_harmonic1' in line:
                    print(line['E_harmonic1'])
