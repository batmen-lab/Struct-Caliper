"""Check wrapper output retention with simulated search binaries and real annotation."""
import contextlib
import io
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class WrapperOutputTests(unittest.TestCase):
    def setup_run(self, tool):
        temp = tempfile.TemporaryDirectory(prefix=f'{tool}-outputs-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / 'results/nested'
        self.output = self.out / 'result.tsv'
        self.lookup = self.root / 'lookup.txt'
        self.lookup.write_text('-1 0.9\n0 0.5\n1 0.1\n')
        self.query = self.root / ('query.pdb' if tool == 'foldseek' else 'query.bca')
        self.query.write_text('input must be preserved')
        self.tool = tool
        self.calls = []
        self.fail_search = False

    def simulate_search(self, cmd, check):
        self.calls.append(cmd)
        if self.tool == 'foldseek':
            if cmd[1] in ['createdb', 'structureshuffle']:
                db = Path(cmd[3])
                db.write_text('generated database')
                if cmd[1] == 'createdb':
                    Path(str(db) + '.index').write_text('0\t0\t1\n')
                    Path(str(db) + '.dbtype').write_text('type')
                return
            path = Path(cmd[4])
            shuffled = cmd[2].endswith('shuffledDB')
            Path(cmd[5]).mkdir(exist_ok=True)
            (Path(cmd[5]) / 'search-cache').write_text('cache')
            text = ('query1_markovtog1\ttarget1\t1\t1\nquery1_markovtog2\ttarget1\t3\t1\n'
                    if shuffled else 'query1\ttarget1\t5\t0.1\n')
        else:
            path = Path(cmd[cmd.index('-output') + 1])
            shuffled = '-num_shuffles' in cmd
            text = ('1\tquery1_shuffle0\ttarget1\t1\t1\n3\tquery1_shuffle1\ttarget1\t1\t1\n'
                    if shuffled else '5\tquery1\ttarget1\t0.1\t0.1\n')
        if self.fail_search and not shuffled:
            raise subprocess.CalledProcessError(1, cmd)
        path.write_text(text)

    def execute(self, keep=False, before=False, labels=False):
        script = ROOT / f'src/{self.tool}.py'
        argv = [str(script), '--lookup_url', str(self.lookup), '--num_shuffle', '2']
        if keep and before:
            argv += ['--keep-search-results']
        if self.tool == 'foldseek':
            argv += ['easy-search', '-query', str(self.query), '-db', 'target', '-out', str(self.output)]
        else:
            argv += ['search', '--search', str(self.query), '--db', 'target', '--output', str(self.output)]
        if keep and not before:
            argv += ['--keep-search-results']
        if labels:
            fasta = self.root / 'labels.fa'
            fasta.write_text('>query1 a.1.1.1\nAAA\n>target1 a.1.1.1\nAAA\n')
            argv += ['--fasta_url', str(fasta)]
        with patch.object(sys, 'argv', argv), patch('subprocess.run', side_effect=self.simulate_search), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(script), run_name='__main__')

    def test_default_retains_only_sgpval(self):
        for tool in ['foldseek', 'reseek']:
            with self.subTest(tool=tool):
                self.setup_run(tool)
                self.execute()
                self.assertEqual([p.name for p in self.out.iterdir()], ['result_sgpval.tsv'])
                text = self.output.with_name(self.output.stem + '_sgpval.tsv').read_text()
                self.assertIn('sgpval\n', text)
                self.assertIn('1.000000e-01', text)
                self.assertEqual(self.query.read_text(), 'input must be preserved')

    def test_keep_flag_positions_and_cache_reuse(self):
        for tool in ['foldseek', 'reseek']:
            for before in [False, True]:
                with self.subTest(tool=tool, before=before):
                    self.setup_run(tool)
                    self.execute(keep=True, before=before)
                    self.assertTrue(self.output.is_file())
                    self.assertTrue(self.output.with_name(self.output.stem + '_sgpval.tsv').is_file())
                    self.assertEqual(len(list(self.out.rglob('shuf_result.tsv'))), 1)
                    self.calls.clear()
                    self.execute(keep=True, before=before)
                    self.assertEqual(self.calls, [], 'Retained search files should be reused')

    def test_default_preserves_existing_files(self):
        for tool in ['foldseek', 'reseek']:
            with self.subTest(tool=tool):
                self.setup_run(tool)
                self.out.mkdir(parents=True)
                self.output.write_text('old raw hits')
                cache = self.out / ('result' if tool == 'foldseek' else 'reseek_output_old')
                cache.mkdir()
                (cache / 'shuf_result.tsv').write_text('old shuffled hits')
                self.execute()
                self.assertEqual(self.output.read_text(), 'old raw hits')
                self.assertEqual((cache / 'shuf_result.tsv').read_text(), 'old shuffled hits')
                self.assertTrue(self.calls, 'Default mode should perform fresh searches')
                self.assertFalse(list(self.out.glob('.*-*/')))

    def test_default_cleans_temporary_files_on_search_failure(self):
        for tool in ['foldseek', 'reseek']:
            with self.subTest(tool=tool):
                self.setup_run(tool)
                self.fail_search = True
                with self.assertRaises(subprocess.CalledProcessError):
                    self.execute()
                self.assertEqual(list(self.out.iterdir()), [])
                self.assertTrue(self.query.is_file())

    def test_optional_labels_are_retained(self):
        for tool in ['foldseek', 'reseek']:
            with self.subTest(tool=tool):
                self.setup_run(tool)
                self.execute(labels=True)
                labeled = self.out / 'analyze_label/result_sgpval.tsv'
                self.assertTrue(labeled.is_file())
                self.assertIn('category', labeled.read_text())
                self.assertFalse(self.output.exists())
                self.assertFalse(list(self.out.rglob('shuf_result.tsv')))


if __name__ == '__main__':
    unittest.main()
