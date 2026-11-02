"""Exercise build selection with simulated hosts/toolchains; no foreign builds occur.

Run: python3 -m unittest discover -s scripts/tests -v
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'build.sh'


class BuildSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='caliper build ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'scripts').mkdir()
        shutil.copy2(SCRIPT, self.root / 'scripts/build.sh')
        self.fake = self.root / 'fake-bin'
        self.fake.mkdir()
        self.log = self.root / 'calls.log'
        self.env = dict(os.environ, PATH=f'{self.fake}:{os.environ["PATH"]}',
                        TEST_HOST='Linux', TEST_ARCH='x86_64', TEST_RELEASE='generic',
                        TEST_CALL_LOG=str(self.log), JOBS='2')
        for key in ['SC_BUILD_DIR', 'SC_BIN_DIR', 'WSL_DISTRO_NAME']:
            self.env.pop(key, None)
        self.stub('uname', '''case "$1" in
-s) echo "$TEST_HOST";; -m) echo "$TEST_ARCH";; -r) echo "$TEST_RELEASE";; esac''')
        self.stub('nproc', 'echo 2')
        self.stub('sysctl', 'echo 2')
        self.stub('arch', 'exit "${TEST_ROSETTA_STATUS:-0}"')
        self.stub('brew', 'printf "%s\\n" "/mock brew/zstd"')
        self.stub('cmake', '''printf '%s\\n' "$*" >> "$TEST_CALL_LOG"
if [[ "${TEST_CMAKE_FAIL:-0}" == 1 ]]; then exit 9; fi
if [[ "$1" == --build ]]; then
    mkdir -p "$2/src"
    printf '#!/bin/sh\\nexit 0\\n' > "$2/src/foldseek"
    chmod +x "$2/src/foldseek"
fi''')
        for name in ['make', 'cargo', 'cc', 'c++', 'gcc', 'g++', 'gcc-11', 'g++-11',
                     'clang', 'clang++', 'curl', 'python3']:
            self.stub(name, 'exit 0')
        source = self.root / 'libs/reseek/src'
        source.mkdir(parents=True)
        for name in ['linux_x86', 'osx_x86', 'osx_arm']:
            script = source / f'build_{name}.bash'
            script.write_text('''#!/bin/bash
printf '%s %s\\n' "$0" "$MAKEFLAGS" >> "$TEST_CALL_LOG"
# Simulate the upstream helper masking a compiler failure.
if [[ "${TEST_RESEEK_NO_BINARY:-0}" == 1 ]]; then exit 0; fi
mkdir -p ../bin
printf '#!/bin/sh\\nexit 0\\n' > ../bin/reseek
chmod +x ../bin/reseek
''')

    def stub(self, name, source):
        path = self.fake / name
        path.write_text('#!/bin/bash\nset -eu\n' + source + '\n')
        path.chmod(0o755)

    def run_build(self, *args, **env):
        return subprocess.run(['bash', str(self.root / 'scripts/build.sh'), *args],
                              env={**self.env, **env}, text=True, capture_output=True)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_platform_dispatch(self):
        cases = [
            ('linux', 'Linux', 'x86_64', 'generic', 'build_linux_x86.bash'),
            ('macos', 'Darwin', 'x86_64', 'darwin', 'build_osx_x86.bash'),
            ('macOS', 'Darwin', 'arm64', 'darwin', 'build_osx_arm.bash'),
            ('windows', 'Linux', 'x86_64', 'microsoft-standard-WSL2', 'build_linux_x86.bash'),
        ]
        for system, host, arch, release, selected in cases:
            with self.subTest(system=system, arch=arch):
                result = self.run_build('--system', system, TEST_HOST=host,
                                        TEST_ARCH=arch, TEST_RELEASE=release)
                self.assert_success(result)
                self.assertIn(selected, self.log.read_text())
                self.assertTrue((self.root / 'bin/foldseek').is_file())
                self.assertTrue((self.root / 'bin/reseek').is_file())
                if host == 'Darwin':
                    self.assertIn('-DREQUIRE_OPENMP=0', self.log.read_text())
                    self.assertIn('-DZSTD_LIBRARIES=/mock brew/zstd/lib/libzstd.a', self.log.read_text())
                if arch == 'arm64':
                    self.assertIn('-DHAVE_ARM8=1', self.log.read_text())
                self.log.unlink()

    def test_dry_run_writes_nothing(self):
        result = self.run_build('--system=linux', '--dry-run')
        self.assert_success(result)
        self.assertFalse((self.root / 'build').exists())
        self.assertFalse((self.root / 'bin').exists())
        self.assertFalse(self.log.exists())

    def test_invalid_arguments(self):
        for args in [('--system',), ('--system', 'plan9'), ('--system=',),
                     ('--typo',), ('foldseek', 'reseek'), ('--tool',),
                     ('--tool', 'unknown'), ('--tool=',), ('--tool', 'both', 'reseek')]:
            with self.subTest(args=args):
                self.assertNotEqual(self.run_build(*args).returncode, 0)
        self.assertFalse((self.root / 'build').exists())

    def test_wrong_host_rejected_before_writes(self):
        for system in ['macos', 'windows']:
            result = self.run_build('--system', system)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'build').exists())

    def test_native_windows_explains_wsl(self):
        result = self.run_build('--system', 'windows', TEST_HOST='MINGW64_NT-10.0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Open WSL2', result.stderr)

    def test_wsl_auto_detection(self):
        result = self.run_build('--dry-run', TEST_RELEASE='microsoft-standard-WSL2')
        self.assert_success(result)
        self.assertIn('system=windows', result.stdout)

    def test_linux_auto_detection(self):
        result = self.run_build('--dry-run')
        self.assert_success(result)
        self.assertIn('system=linux', result.stdout)

    def test_tool_selection(self):
        result = self.run_build('reseek', '--system', 'linux')
        self.assert_success(result)
        self.assertFalse((self.root / 'bin/foldseek').exists())
        self.assertTrue((self.root / 'bin/reseek').exists())

    def test_named_tool_selection(self):
        for tool in ['foldseek', 'reseek', 'both']:
            with self.subTest(tool=tool):
                result = self.run_build('--system', 'linux', '--tool', tool, '--dry-run')
                self.assert_success(result)
                self.assertEqual('cmake -S' in result.stdout, tool in ['foldseek', 'both'])
                self.assertEqual('build_linux_x86.bash' in result.stdout, tool in ['reseek', 'both'])
        self.assert_success(self.run_build('--tool=foldseek', '--dry-run'))

    def test_arm_linux_reseek_is_rejected(self):
        result = self.run_build('--dry-run', TEST_ARCH='aarch64')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('select foldseek only', result.stderr)
        self.assert_success(self.run_build('foldseek', '--dry-run', TEST_ARCH='aarch64'))

    def test_jobs_validation(self):
        self.assertNotEqual(self.run_build('--dry-run', JOBS='0').returncode, 0)
        self.assertNotEqual(self.run_build('--dry-run', JOBS='wrong').returncode, 0)

    def test_missing_rosetta_fails_before_build(self):
        result = self.run_build('--system', 'macos', TEST_HOST='Darwin',
                                TEST_ARCH='arm64', TEST_ROSETTA_STATUS='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Rosetta', result.stderr)
        self.assertFalse((self.root / 'build').exists())

    def test_compilation_failure_propagates(self):
        result = self.run_build('--system', 'linux', TEST_CMAKE_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'bin/foldseek').exists())

    def test_failed_reseek_cannot_reuse_stale_binary(self):
        stale = self.root / 'build/linux-x86_64/reseek/bin/reseek'
        stale.parent.mkdir(parents=True)
        stale.write_text('stale binary')
        result = self.run_build('reseek', TEST_RESEEK_NO_BINARY='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('produced no executable', result.stderr)
        self.assertFalse((self.root / 'bin/reseek').exists())


if __name__ == '__main__':
    unittest.main()
