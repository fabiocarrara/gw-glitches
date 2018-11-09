import os
import shutil
from io import StringIO

import pandas as pd
import torch


class RunManager:

    def __init__(self, args, root='runs/', ignore=None, main=None):

        self.args = args
        self.filenames = {
            'params': 'params.csv',
            'log': 'log.csv',
            'results': 'results.csv',
            'ckpt': 'ckpt',
            'last': 'last.pth',
            'best': 'best.pth',
        }
        self.root = root
        self.ignore = ignore
        self.params = {k: v for k, v in vars(self.args).items() if k not in ignore}

        assert main is None or main in self.params, "'main' should be one of: ({}), got {}".format(
            ','.join(self.params.keys()), main)
        self.main = main

        self._init_paths()  # initializes self.run_dir, self.abs_run_dir, self.abs_ckpt_dir

        self.existing = os.path.exists(self.abs_run_dir)

        log_fname = self.pathTo('log')
        param_fname = self.pathTo('params')
        results_fname = self.pathTo('results')
        self.log_df = pd.read_csv(log_fname) if os.path.exists(log_fname) else pd.DataFrame()
        self.param_df = pd.read_csv(param_fname) if os.path.exists(param_fname) else pd.DataFrame(self.params, index=[0])  # an index is mandatory for a single line
        self.results_df = pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()

        if not self.existing:
            os.makedirs(self.abs_ckpt_dir)
            self.param_df.to_csv(self.pathTo('params'), index=False)

    def _init_paths(self):
        abbrev = {k: '{}{}{}'.format(*self._abbr(k, v)) for k, v in self.params.items()}

        if self.main:
            run_dir = []
            main_param = abbrev[self.main]
            secondary_params = sorted(v for k, v in abbrev.items() if k != self.main)
            run_dir.append(main_param)
            run_dir.extend(secondary_params)
        else:
            run_dir = abbrev.values()

        run_dir = '_'.join(run_dir)
        self.run_dir = run_dir
        self.abs_run_dir = os.path.join(self.root, self.run_dir)
        self.abs_ckpt_dir = self.pathTo('ckpt')

    def _abbr(self, name, value):

        is_main_param = self.main == name

        def prefixLen(a, b):
            i = 0
            while a[i] == b[i]:
                i += 1
            return i

        if is_main_param:
            prefix = ''
        else:
            prefix = [name[:prefixLen(p, name) + 1] for p in self.params.keys() if p != name]
            prefix = max(prefix, key=lambda x: len(x))

        sep = '-' if (type(value) == str and not is_main_param) else ''

        return prefix, sep, str(value)

    def __str__(self):
        s = StringIO()
        print('Run Dir: {}'.format(self.abs_run_dir), file=s)
        print('Params:', file=s)
        with pd.option_context('display.width', None), pd.option_context('max_columns', None):
            self.param_df.to_string(s, index=False)

        if not self.results_df.empty:
            print('Results:', file=s)
            with pd.option_context('display.width', None), pd.option_context('max_columns', None):
                self.results_df.to_string(s, index=False)

        return s.getvalue()

    def pathTo(self, what):
        assert what in self.filenames, "Unknown run resource: '{}'".format(what)
        path = os.path.join(self.root, self.run_dir, self.filenames[what])
        return path

    def ckpt(self, which='best'):
        ckpt_path = os.path.join(self.pathTo('ckpt'), self.filenames[which])
        return ckpt_path

    def save_checkpoint(self, state, is_best):
        filename = self.ckpt('last')
        torch.save(state, filename)
        if is_best:
            best_filename = self.ckpt('best')
            shutil.copyfile(filename, best_filename)

    def pushLog(self, metrics):
        ts = pd.to_datetime('now')
        metrics = pd.DataFrame(metrics, index=(ts,))
        self.log_df = pd.concat((self.log_df, metrics))
        self.log_df.to_csv(self.pathTo('log'))

    def writeResults(self, metrics):
        ts = pd.to_datetime('now')
        self.results_df = pd.DataFrame(metrics, index=(ts,))
        self.results_df.to_csv(self.pathTo('results'))
        with pd.option_context('display.width', None), pd.option_context('max_columns', None):
            print(self.results_df)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GW Glitches Classification')
    parser.add_argument('-e', '--epochs', type=int, default=70)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-m', '--model', choices=('1d-conv', 'paper'), default='1d-conv')
    parser.add_argument('-s', '--seed', type=int, default=23)
    parser.add_argument('--no-cuda', action='store_true')
    parser.set_defaults(no_cuda=False)
    args = parser.parse_args()

    run = RunManager(args, root='prova', ignore=['no_cuda'], main='model')
    print(run)
    print(run.ckpt('best'))
