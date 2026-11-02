commit=`cat vcxproj_make_commit.txt`

curl -fsSL https://raw.githubusercontent.com/rcedgar/vcxproj_make/$commit/vcxproj_make.py \
  > vcxproj_make.py

mkdir -p ../bin

# reseek uses x86 SSE intrinsics and has no native ARM support.
# Build as x86_64 under Rosetta 2, which runs transparently on Apple Silicon.
# We create thin wrapper scripts so vcxproj_make can invoke "arch -x86_64 ..."
# via a plain compiler name (no spaces).
cat > _clangxx_x86.sh << 'WRAPPER'
#!/usr/bin/env bash
exec arch -x86_64 clang++ -march=haswell --std=c++17 "$@"
WRAPPER
cat > _clang_x86.sh << 'WRAPPER'
#!/usr/bin/env bash
exec arch -x86_64 clang -march=haswell "$@"
WRAPPER
chmod +x _clangxx_x86.sh _clang_x86.sh

python3 ./vcxproj_make.py --nonative --nostatic \
  --cppcompiler "$(pwd)/_clangxx_x86.sh" \
  --ccompiler   "$(pwd)/_clang_x86.sh"

rm -f _clangxx_x86.sh _clang_x86.sh
