#!/usr/bin/env bash
# Build the bundled modified tools. Windows builds run inside WSL2 (Linux binaries).
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/build.sh [--system linux|macos|windows] [--tool foldseek|reseek|both] [--dry-run]

  --system NAME  Host platform (default: auto-detect; macOS is also accepted).
                 windows requires an x86-64 WSL2 Ubuntu/Linux shell.
  --tool NAME    Build foldseek, reseek, or both (default: both).
                 Positional foldseek/reseek/all are also accepted.
  --dry-run      Print build commands without downloading, building, or writing files.
  -h, --help     Show this help.

Examples:
  bash scripts/build.sh --system linux
  bash scripts/build.sh --system macos --tool foldseek
  bash scripts/build.sh --system windows       # inside WSL2

JOBS sets parallel build jobs. SC_BUILD_DIR and SC_BIN_DIR override the build
and binary directories. This script builds for the current host, not a foreign OS.
HELP
}
fail() { echo "[build] ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"; }
run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

SYSTEM=auto
WHAT=all
TOOL_SELECTED=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system)
            [[ $# -ge 2 && "$2" != -* ]] || fail '--system requires linux, macos, or windows'
            SYSTEM="$2"; shift 2 ;;
        --system=*) SYSTEM="${1#*=}"; shift ;;
        --tool|--tool=*)
            [[ "$TOOL_SELECTED" == 0 ]] || fail 'Select only one tool (foldseek, reseek, or both)'
            if [[ "$1" == --tool ]]; then
                [[ $# -ge 2 && "$2" != -* ]] || fail '--tool requires foldseek, reseek, or both'
                WHAT="$2"; shift 2
            else
                WHAT="${1#*=}"; shift
            fi
            case "$WHAT" in
                foldseek|reseek) ;;
                both|all) WHAT=all ;;
                *) fail "Unknown tool: $WHAT. Choose foldseek, reseek, or both." ;;
            esac
            TOOL_SELECTED=1 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        all|both|foldseek|reseek)
            [[ "$TOOL_SELECTED" == 0 ]] || fail 'Select only one tool (foldseek, reseek, or both)'
            WHAT="$1"; TOOL_SELECTED=1; shift
            [[ "$WHAT" != both ]] || WHAT=all ;;
        *) fail "Unknown argument: $1 (use --help)" ;;
    esac
done
case "$SYSTEM" in
    auto|linux|macos|windows) ;;
    macOS|Darwin|darwin) SYSTEM=macos ;;
    *) fail "Unknown system: $SYSTEM. Choose linux, macos, or windows." ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="$(uname -s)"
ARCH="$(uname -m)"
WSL=0
case "$HOST" in
    Linux)
        HOST_SYSTEM=linux
        case "$(uname -r)" in *[Mm]icrosoft*|*WSL*) WSL=1 ;; esac
        [[ -z "${WSL_DISTRO_NAME:-}" ]] || WSL=1 ;;
    Darwin) HOST_SYSTEM=macos ;;
    MINGW*|MSYS*|CYGWIN*)
        fail 'Native Windows Bash is not supported. Open WSL2 Ubuntu and run: bash scripts/build.sh --system windows' ;;
    *) fail "Unsupported host: $HOST" ;;
esac
if [[ "$SYSTEM" == auto ]]; then
    SYSTEM="$HOST_SYSTEM"
    [[ "$WSL" == 0 ]] || SYSTEM=windows
fi
if [[ "$SYSTEM" == windows ]]; then
    [[ "$HOST_SYSTEM" == linux && "$WSL" == 1 ]] || fail '--system windows must run inside WSL2; this is not a Windows cross-compiler'
else
    [[ "$SYSTEM" == "$HOST_SYSTEM" ]] || fail "--system $SYSTEM does not match host $HOST_SYSTEM; cross-compilation is not supported"
fi
# Bundled Reseek depends on x86 SIMD; macOS ARM uses its Rosetta build script.
if [[ "$WHAT" != foldseek ]]; then
    case "$HOST_SYSTEM/$ARCH" in
        linux/x86_64|linux/amd64|macos/x86_64|macos/arm64) ;;
        *) fail "Bundled Reseek is not supported on $HOST_SYSTEM/$ARCH; select foldseek only" ;;
    esac
fi

if [[ -z "${JOBS:-}" ]]; then
    if command -v nproc >/dev/null 2>&1; then JOBS="$(nproc)"
    elif [[ "$HOST_SYSTEM" == macos ]]; then JOBS="$(sysctl -n hw.ncpu)"
    else JOBS=1
    fi
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail 'JOBS must be a positive integer'
# Resolve overrides before entering Reseek's source directory.
BUILD_DIR="${SC_BUILD_DIR:-$ROOT/build/$SYSTEM-$ARCH}"
BIN_DIR="${SC_BIN_DIR:-$ROOT/bin}"
[[ "$BUILD_DIR" == /* && "$BIN_DIR" == /* ]] || fail 'SC_BUILD_DIR and SC_BIN_DIR must be absolute paths'

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
RESEEK_SCRIPT=build_linux_x86.bash
if [[ "$HOST_SYSTEM" == macos ]]; then
    CMAKE_ARGS+=(-DREQUIRE_OPENMP=0 -DUSE_SYSTEM_ZSTD=1)
    if [[ "$ARCH" == arm64 ]]; then
        RESEEK_SCRIPT=build_osx_arm.bash
        CMAKE_ARGS+=(-DHAVE_ARM8=1)
    else
        RESEEK_SCRIPT=build_osx_x86.bash
    fi
    # Homebrew's zstd may be outside CMake's default search paths.
    if [[ "$WHAT" != reseek ]] && command -v brew >/dev/null 2>&1; then
        if ZSTD_PREFIX="$(brew --prefix zstd 2>/dev/null)"; then
            CMAKE_ARGS+=("-DZSTD_INCLUDE_DIRS=$ZSTD_PREFIX/include" "-DZSTD_LIBRARIES=$ZSTD_PREFIX/lib/libzstd.a")
        elif [[ "$DRY_RUN" == 0 ]]; then
            fail 'Install zstd with: brew install zstd'
        fi
    fi
fi

if [[ "$DRY_RUN" == 0 ]]; then
    need make
    if [[ "$WHAT" != reseek ]]; then
        need cmake; need cargo
        if [[ "$HOST_SYSTEM" == macos ]]; then need clang; need clang++
        else need cc; need c++
        fi
    fi
    if [[ "$WHAT" != foldseek ]]; then
        need curl; need python3
        if [[ "$HOST_SYSTEM" == linux ]]; then need gcc; need g++
        elif [[ "$ARCH" == arm64 ]]; then
            need clang; need clang++; need arch
            arch -x86_64 /usr/bin/true || fail 'Reseek requires Rosetta 2 on Apple Silicon'
        else
            need gcc-11; need g++-11
        fi
    fi
fi

echo "[build] system=$SYSTEM host=$HOST/$ARCH tools=$WHAT jobs=$JOBS"
[[ "$SYSTEM" != windows ]] || echo '[build] Windows via WSL: producing Linux binaries; run wrappers inside WSL too.'
run mkdir -p "$BIN_DIR" "$BUILD_DIR"

if [[ "$WHAT" == all || "$WHAT" == foldseek ]]; then
    echo '[build] foldseek'
    run cmake -S "$ROOT/libs/foldseek" -B "$BUILD_DIR/foldseek" "${CMAKE_ARGS[@]}"
    run cmake --build "$BUILD_DIR/foldseek" --parallel "$JOBS"
    run install -m 755 "$BUILD_DIR/foldseek/src/foldseek" "$BIN_DIR/foldseek"
fi

if [[ "$WHAT" == all || "$WHAT" == reseek ]]; then
    echo "[build] reseek ($RESEEK_SCRIPT)"
    # Fresh copy prevents a stale binary from masking a failed upstream build.
    run rm -rf "$BUILD_DIR/reseek"
    run cp -R "$ROOT/libs/reseek" "$BUILD_DIR/reseek"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "[dry-run] cd $BUILD_DIR/reseek/src"
        run env "MAKEFLAGS=-j$JOBS" bash -e "$RESEEK_SCRIPT"
    else
        (cd "$BUILD_DIR/reseek/src" && MAKEFLAGS="-j$JOBS" bash -e "$RESEEK_SCRIPT")
        [[ -x "$BUILD_DIR/reseek/bin/reseek" ]] || fail "Reseek build produced no executable; inspect $BUILD_DIR/reseek/src/make.stderr"
    fi
    run install -m 755 "$BUILD_DIR/reseek/bin/reseek" "$BIN_DIR/reseek"
fi

echo "[build] done: $BIN_DIR"
