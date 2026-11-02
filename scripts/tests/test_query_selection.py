"""Verify one query entry per source structure without changing input databases."""
import contextlib
import io
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'src/foldseek.py'))


class QuerySelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='first-query-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = str(self.root / 'input')
        self.work = self.root / 'work'
        self.work.mkdir()
        for suffix in ['', '_ss', '_h', '_ca']:
            Path(self.raw + suffix).write_text('database contents')
            Path(self.raw + suffix + '.dbtype').write_text('type')
            Path(self.raw + suffix + '.index').write_text('20\t0\t1\n5\t1\t1\n9\t2\t1\n2\t3\t1\n')
        Path(self.raw + '.source').write_text('0\tquery1_MODEL_2\n1\tquery1_MODEL_1\n2\tquery2_MODEL_1\n')
        Path(self.raw + '.lookup').write_text('20\tquery1_MODEL_2_A\t0\n5\tquery1_MODEL_1_A\t1\n9\tquery1_MODEL_1_B\t1\n2\tquery2_A\t2\n')

    def fake_subdb(self, cmd, check):
        _, action, keys, source, dest, *_ = cmd
        self.assertEqual(action, 'createsubdb')
        selected = set(Path(keys).read_text().splitlines())
        Path(dest).write_bytes(Path(source).read_bytes())
        Path(dest + '.dbtype').write_bytes(Path(source + '.dbtype').read_bytes())
        Path(dest + '.index').write_text(''.join(line for line in Path(source + '.index').read_text().splitlines(keepends=True)
                                                  if line.split()[0] in selected))
        # Real Foldseek links lookup/source metadata to its input.
        if source == self.raw:
            for suffix in ['.lookup', '.source']:
                path = Path(dest + suffix)
                if path.is_symlink() or path.exists():
                    path.unlink()
                path.symlink_to(source + suffix)

    def select(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return MODULE['select_first_query_entries']('foldseek', self.raw, str(self.work), **kwargs)

    def test_lowest_key_per_source_includes_model_and_chain_selection(self):
        chosen, _ = MODULE['first_query_keys'](self.raw)
        self.assertEqual(chosen, [2, 5])

    def test_all_components_and_original_metadata_preserved(self):
        original = {p: p.read_bytes() for p in self.root.glob('input*')}
        with patch('subprocess.run', side_effect=self.fake_subdb) as run:
            first, changed = self.select()
        self.assertTrue(changed)
        self.assertEqual(run.call_count, 4)
        for suffix in ['', '_ss', '_h', '_ca']:
            self.assertEqual(len(Path(first + suffix + '.index').read_text().splitlines()), 2)
        self.assertEqual(len(Path(first + '.lookup').read_text().splitlines()), 2)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)
        with patch('subprocess.run') as run:
            self.assertEqual(self.select(), (first, False))
            run.assert_not_called()

    def test_unique_source_entries_need_no_subdatabase(self):
        Path(self.raw + '.source').write_text('0\tq1\n1\tq2\n2\tq3\n3\tq4\n')
        Path(self.raw + '.lookup').write_text('20\tq1\t0\n5\tq2\t1\n9\tq3\t2\n2\tq4\t3\n')
        with patch('subprocess.run') as run:
            self.assertEqual(self.select(), (self.raw, False))
            run.assert_not_called()

    def test_single_entry_requires_no_metadata(self):
        Path(self.raw + '.index').write_text('20\t0\t1\n')
        Path(self.raw + '.source').unlink()
        Path(self.raw + '.lookup').unlink()
        self.assertEqual(self.select(), (self.raw, False))

    def test_missing_multi_entry_metadata_fails_clearly(self):
        Path(self.raw + '.lookup').unlink()
        with self.assertRaisesRegex(ValueError, 'requires .source and .lookup'):
            self.select()

    def test_failed_subset_does_not_mark_selection_complete(self):
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'foldseek')):
            with self.assertRaises(subprocess.CalledProcessError):
                self.select()
        self.assertFalse((self.work / 'queryDB_first.selection').exists())

    def test_overwrite_rebuilds_selected_database(self):
        with patch('subprocess.run', side_effect=self.fake_subdb):
            first, _ = self.select()
        with patch('subprocess.run', side_effect=self.fake_subdb) as run:
            self.assertEqual(self.select(overwrite=True), (first, True))
            self.assertEqual(run.call_count, 4)


if __name__ == '__main__':
    unittest.main()
