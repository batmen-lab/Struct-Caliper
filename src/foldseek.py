import os, argparse, subprocess, re, shutil
from contextlib import nullcontext
from tempfile import TemporaryDirectory
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd

def main(args):
    check_args(args)
    query_input, target_db, output_tsv, real_kwargs, shuf_kwargs = parse_search_args(args)
    print('query_input={}\ttarget_db={}\toutput_tsv={}'.format(query_input, target_db, output_tsv))

    output_dir = os.path.dirname(os.path.abspath(output_tsv))
    os.makedirs(output_dir, exist_ok=True)
    out_stem = os.path.splitext(os.path.basename(output_tsv))[0]
    retained_work_dir = os.path.join(output_dir, out_stem)
    # Only this run's temporary directory is cleaned; existing caches are untouched.
    workspace = (nullcontext(retained_work_dir) if args.keep_search_results else
                 TemporaryDirectory(prefix='.foldseek-', dir=output_dir))
    with workspace as work_dir:
        real_tsv = output_tsv if args.keep_search_results else os.path.join(work_dir, 'real_result.tsv')
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Create query DB (skip if input is already a foldseek DB)
        if os.path.exists(query_input + '.dbtype'):
            query_db = query_input
            print('Using pre-built query DB: {}'.format(query_db))
        else:
            query_db = os.path.join(work_dir, 'queryDB_original')
            if not os.path.exists(query_db) or args.overwrite > 0:
                cmd = [args.foldseek_url, 'createdb', query_input, query_db]
                print('Creating query DB: {}'.format(cmd))
                subprocess.run(cmd, check=True)

        # Keep one model-and-chain entry per input structure, including pre-built queries.
        query_db, selection_changed = select_first_query_entries(
            args.foldseek_url, query_db, work_dir, args.overwrite > 0)
        selection_signature = None
        search_selection_path = os.path.join(work_dir, 'search_selection.txt')
        if os.path.isfile(query_db + '.selection'):
            with open(query_db + '.selection') as f:
                selection_signature = f.read()
            previous_selection = None
            if os.path.isfile(search_selection_path):
                with open(search_selection_path) as f:
                    previous_selection = f.read()
            selection_changed = selection_changed or previous_selection != selection_signature

        # Step 2: Create shuffled DB (num_shuffle decoys for every entry of the query DB)
        shuf_db = os.path.join(work_dir, 'shuffledDB')
        if not os.path.exists(shuf_db) or args.overwrite > 0 or selection_changed:
            cmd = [args.foldseek_url, 'structureshuffle', query_db, shuf_db,
                   '--num-shuffles', str(args.num_shuffle)]
            # Use structureshuffle's default first-order Markov chains (order 1).
            if args.seed is not None:
                cmd += ['--seed', str(args.seed)]
            print('Creating shuffled DB: {}'.format(cmd))
            subprocess.run(cmd, check=True)

        # Shared tmp dir for both searches so the target directory is only indexed once
        # (foldseek caches the target DB inside tmp when the target is a raw directory)
        shared_tmp = os.path.join(work_dir, 'tmp')

        # Step 3: Search shuffled queries against target (uses shuf_kwargs)
        shuf_tsv = os.path.join(work_dir, 'shuf_result.tsv')
        if not os.path.exists(shuf_tsv) or args.overwrite > 0 or selection_changed:
            cmd = build_easy_search_cmd(args.foldseek_url, shuf_db, target_db, shuf_tsv, shared_tmp, shuf_kwargs)
            print('Searching shuffled queries: {}'.format(cmd))
            subprocess.run(cmd, check=True)

        # Step 4: Build null distribution from shuffled output
        print('Building null distribution from: {}'.format(shuf_tsv))
        null_dist = build_null_dist(shuf_tsv)
        print('null_dist: {} queries'.format(len(null_dist)))

        # Step 5: Search real queries against target (uses real_kwargs)
        if not os.path.exists(real_tsv) or args.overwrite > 0 or selection_changed:
            cmd = build_easy_search_cmd(args.foldseek_url, query_db, target_db, real_tsv, shared_tmp, real_kwargs)
            print('Searching real queries: {}'.format(cmd))
            subprocess.run(cmd, check=True)
            ensure_header(real_tsv, 'query\ttarget\tbits\tevalue')

        # Step 6: Load SG lookup table
        print('Loading lookup table: {}'.format(args.lookup_url))
        lookup_mat = np.loadtxt(args.lookup_url)
        lookup_keys = lookup_mat[:, 0].flatten()
        print('lookup_mat={}'.format(lookup_mat.shape))

        # Step 7: Annotate real hits with SG-pvalue
        new_tsv = sgpval_output_path(output_tsv)
        add_sgpval(real_tsv, new_tsv, null_dist, lookup_keys, lookup_mat)
        print('Output with SG-pvalue written to: {}'.format(new_tsv))

        # Step 8: Annotate with SCOP labels (optional)
        if args.fasta_url is not None:
            labeled_tsv = add_scop_labels(new_tsv, args.fasta_url, output_dir)
            print('Output with SCOP labels written to: {}'.format(labeled_tsv))
        if selection_signature is not None:
            with open(search_selection_path, 'w') as f:
                f.write(selection_signature)


def first_query_keys(query_db):
    """Choose the smallest database key per source structure, ignoring MODEL suffixes."""
    with open(query_db + '.index') as f:
        keys = {int(line.split()[0]) for line in f if line.strip()}
    if len(keys) <= 1:
        return sorted(keys), keys
    if not all(os.path.isfile(query_db + suffix) for suffix in ('.source', '.lookup')):
        raise ValueError('Multi-entry query DB requires .source and .lookup metadata to select '
                         'one model/chain per structure. Supply the original structure files '
                         'or rebuild the DB with foldseek createdb.')
    sources = {}
    with open(query_db + '.source') as f:
        for line in f:
            source_id, name = line.rstrip('\n').split('\t', 1)
            sources[source_id] = re.sub(r'_MODEL_\d+$', '', name)
    first = {}
    found = set()
    with open(query_db + '.lookup') as f:
        for line in f:
            key, name, source_id = line.rstrip('\n').split('\t')[:3]
            key = int(key)
            if key not in keys:
                continue
            if source_id not in sources:
                raise ValueError('Query DB source metadata is incomplete: ' + source_id)
            found.add(key)
            source = sources[source_id]
            first[source] = min(first.get(source, key), key)
    if found != keys:
        raise ValueError('Query DB lookup metadata does not cover all query entries')
    return sorted(first.values()), keys


def select_first_query_entries(foldseek_url, query_db, work_dir, overwrite=False):
    """Make a separate query subdatabase; never modify a supplied database or target."""
    selected, all_keys = first_query_keys(query_db)
    if not selected:
        raise ValueError('Query database has no entries: ' + query_db)
    if len(selected) == len(all_keys):
        return query_db, False
    first_db = os.path.join(work_dir, 'queryDB_first')
    key_path = os.path.join(work_dir, 'first_query_keys.txt')
    selection_path = first_db + '.selection'
    # Include the input identity and index version to invalidate retained search caches.
    stat = os.stat(query_db + '.index')
    selection = '{}\n{}:{}\n{}'.format(os.path.abspath(query_db), stat.st_size,
                                          stat.st_mtime_ns, ','.join(map(str, selected)))
    suffixes = [s for s in ('', '_ss', '_h', '_ca') if os.path.isfile(query_db + s + '.dbtype')]
    reusable = (not overwrite and os.path.isfile(selection_path) and
                all(os.path.isfile(first_db + s + ext) for s in suffixes for ext in ('', '.index', '.dbtype')))
    if reusable:
        with open(selection_path) as f:
            reusable = f.read() == selection
    if reusable:
        print('Using selected query DB: {} ({} structures)'.format(first_db, len(selected)))
        return first_db, False
    # A marker is written only after all subdatabases and metadata are complete.
    if os.path.exists(selection_path):
        os.remove(selection_path)
    with open(key_path, 'w') as f:
        f.writelines('{}\n'.format(key) for key in selected)
    for suffix in suffixes:
        subprocess.run([foldseek_url, 'createsubdb', key_path, query_db + suffix,
                        first_db + suffix, '--subdb-mode', '0'], check=True)
    # createsubdb may link metadata back to the input. Replace those links before
    # writing selected metadata, so the original query database remains unchanged.
    for suffix in ('.lookup', '.source'):
        if os.path.lexists(first_db + suffix):
            os.unlink(first_db + suffix)
    selected_set = set(selected)
    with open(query_db + '.lookup') as fin, open(first_db + '.lookup', 'w') as fout:
        for line in fin:
            if int(line.split('\t', 1)[0]) in selected_set:
                fout.write(line)
    shutil.copyfile(query_db + '.source', first_db + '.source')
    with open(selection_path, 'w') as f:
        f.write(selection)
    print('Selected {} of {} query entries (one model/chain per structure)'.format(len(selected), len(all_keys)))
    return first_db, True


def sgpval_output_path(output_tsv):
    """Final result next to -out/--output: result.tsv -> result_sgpval.tsv (other names get _sgpval.tsv appended)."""
    base, ext = os.path.splitext(output_tsv)
    return (base if ext == '.tsv' else output_tsv) + '_sgpval.tsv'


def check_args(args):
    assert os.path.isfile(args.lookup_url), 'lookup_url not found: {}'.format(args.lookup_url)
    assert args.foldseek_url is not None
    assert args.num_shuffle > 0


def parse_extra_argv(extra_argv):
    """Convert a list like ['--flag', 'val', '--bool-flag'] into a dict."""
    kwargs = {}
    i = 0
    while i < len(extra_argv):
        token = extra_argv[i]
        if token.startswith('-'):
            if i + 1 < len(extra_argv) and not extra_argv[i + 1].startswith('-'):
                kwargs[token] = extra_argv[i + 1]
                i += 2
            else:
                kwargs[token] = None
                i += 1
        else:
            i += 1
    return kwargs


def parse_search_args(args):
    # Build real search kwargs
    real_kwargs = {}
    real_kwargs['--format-output'] = 'query,target,bits,evalue'
    if args.threads is not None:           real_kwargs['--threads'] = args.threads
    if args.s is not None:                 real_kwargs['-s'] = args.s
    if args.max_seqs is not None:          real_kwargs['--max-seqs'] = args.max_seqs
    if args.e is not None:                 real_kwargs['-e'] = args.e
    if args.alignment_type is not None:    real_kwargs['--alignment-type'] = args.alignment_type
    if args.c is not None:                 real_kwargs['-c'] = args.c
    if args.cov_mode is not None:          real_kwargs['--cov-mode'] = args.cov_mode
    if args.exhaustive_search is not None: real_kwargs['--exhaustive-search'] = args.exhaustive_search
    if args.comp_bias_corr is not None:    real_kwargs['--comp-bias-corr'] = args.comp_bias_corr

    # Build shuffled search kwargs from the known flags only (before merging
    # pass-through extras), then override alignment-type if requested.
    # Pass-through flags are intentionally excluded from the shuffled search
    # because they may be incompatible with a different alignment-type.
    shuf_kwargs = dict(real_kwargs)
    shuf_kwargs['--no-rev-align'] = 1
    if args.shuf_alignment_type is not None:
        shuf_kwargs['--alignment-type'] = args.shuf_alignment_type
    if args.shuf_comp_bias_corr is not None:
        shuf_kwargs['--comp-bias-corr'] = args.shuf_comp_bias_corr
    if args.shuf_gap_open is not None:
        # Lower the gap-open cost for the shuffled null search ONLY, to nudge the
        # shuffled (null) bit scores a little higher without touching the real search.
        shuf_kwargs['--gap-open'] = args.shuf_gap_open

    # Merge any unrecognized pass-through flags into the real search only.
    real_kwargs.update(args.extra_kwargs)

    assert args.query is not None, '-query is required'
    assert args.db is not None, '-db is required'
    assert args.out is not None, '-out is required'

    return args.query, args.db, args.out, real_kwargs, shuf_kwargs


def build_easy_search_cmd(foldseek_url, query_db, target_db, output_tsv, tmp_dir, kwargs):
    cmd = [foldseek_url, 'easy-search', query_db, target_db, output_tsv, tmp_dir]
    for k, v in kwargs.items():
        cmd += [k] if v is None else [k, str(v)]
    return cmd



def ensure_header(tsv_path, header):
    """Prepend a header line to a TSV file if it doesn't already have one."""
    with open(tsv_path, 'r') as f:
        first_line = f.readline()
    if first_line.startswith('query\t'):
        return
    with open(tsv_path, 'r') as f:
        content = f.read()
    with open(tsv_path, 'w') as f:
        f.write(header + '\n' + content)


def build_null_dist(shuf_tsv):
    """
    Parse foldseek shuffled-query output TSV (columns: query, target, bits, evalue).
    Shuffled query names use the suffix '_shuffled{N}' (1-indexed).
    Returns: {original_id: np.array of best-bits-per-shuffle}
    """
    # best_bits[shuffled_name] = max bits across all targets for that shuffled query
    best_bits = {}
    with open(shuf_tsv) as f:
        for line in f:
            row = line.strip().split('\t')
            if len(row) < 3:
                continue
            try:
                query_name = row[0]
                bits = float(row[2])
            except ValueError:
                continue
            if query_name not in best_bits or bits > best_bits[query_name]:
                best_bits[query_name] = bits

    # Group by original ID by stripping suffix
    _SHUF_SUFFIX = re.compile(r'_(markovtog)\d+$')
    grouped = {}
    for shuf_name, b in best_bits.items():
        m = _SHUF_SUFFIX.search(shuf_name)
        if m:
            original_id = shuf_name[:m.start()]
        else:
            original_id = shuf_name
        if original_id not in grouped:
            grouped[original_id] = []
        grouped[original_id].append(b)

    return {k: np.array(v) for k, v in grouped.items()}


def compute_sgpval(observed_score, null_scores, lookup_keys, lookup_mat):
    if len(null_scores) < 2:
        return float('nan')
    null_std = np.std(null_scores)
    if null_std == 0.0:
        return float('nan')
    s_hat = (observed_score - np.mean(null_scores)) / null_std
    s_hat_key = np.floor(s_hat * 1000.0) / 1000.0
    hit_index = np.argmin(np.fabs(lookup_keys - s_hat_key))
    return float(lookup_mat[hit_index, 1])


def add_sgpval(input_tsv, output_tsv, null_dist, lookup_keys, lookup_mat):
    """
    Read foldseek output TSV (query, target, bits, evalue) and append sg_pvalue column.
    """
    missing_queries = set()
    with open(input_tsv) as fin, open(output_tsv, 'w') as fout:
        fout.write('query\ttarget\tbits\tevalue\tsgpval\n')
        for line in fin:
            if line.startswith('query\t'):
                continue
            row = line.rstrip('\n').split('\t')
            if len(row) < 3:
                fout.write(line)
                continue
            try:
                query_name = row[0]
                bits = float(row[2])
            except ValueError:
                fout.write(line)
                continue

            if query_name not in null_dist:
                missing_queries.add(query_name)
                fout.write('\t'.join(row) + '\tNA\n')   # keep the hit; no decoy scores for this query
                continue

            sg_pv = compute_sgpval(bits, null_dist[query_name], lookup_keys, lookup_mat)
            fout.write('\t'.join(row) + '\t{:.6e}\n'.format(sg_pv))

    if missing_queries:
        print('WARNING: {} queries had no decoy scores, so their hits have sgpval NA: {}'.format(
            len(missing_queries), sorted(missing_queries)[:5]))



def get_seq_label_map(fasta_url):
    records = list(SeqIO.parse(fasta_url, "fasta"))
    fa_id_label_map = {}
    for record in tqdm(records):
        fa_id = record.id
        desc = record.description
        scop_label = desc.split(' ')[1]
        arr = scop_label.split('.')
        assert len(arr) == 4
        scop_fold = '.'.join(arr[:2])
        scop_supf = '.'.join(arr[:3])
        scop_fam  = '.'.join(arr[:4])
        assert fa_id not in fa_id_label_map
        fa_id_label_map[fa_id] = (scop_fold, scop_supf, scop_fam)
    return fa_id_label_map


def check_multi_level_homolog(src_labels, tgt_labels):
    '''
        -1: non-homolog
        0: same fold, different superfamily
        1: same superfamily, different family
        2: same family
    '''
    src_scop_fold, src_scop_supf, src_scop_fam = src_labels
    tgt_scop_fold, tgt_scop_supf, tgt_scop_fam = tgt_labels
    if src_scop_fold != tgt_scop_fold:
        return -1
    elif src_scop_supf != tgt_scop_supf:
        return 0
    elif src_scop_fam != tgt_scop_fam:
        return 1
    else:
        return 2


def get_original_id(fa_id):
    if '_MODEL_' in fa_id:
        fa_id = fa_id[:fa_id.find('_MODEL_')]
    # Strip foldseek chain suffix (_A, _B, _C, etc.) if the base ID is in the map
    # ASTRAL IDs end with a single char after underscore (e.g. d1abc_), so the
    # foldseek chain suffix adds another _X making it d1abc__A — strip the last two chars.
    if len(fa_id) > 2 and fa_id[-2] == '_':
        fa_id = fa_id[:-2]
    return fa_id


def add_scop_labels(sgpval_tsv, fasta_url, output_dir):
    out_dir = os.path.join(output_dir, 'analyze_label')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(sgpval_tsv))

    df = pd.read_csv(sgpval_tsv, sep='\t')

    # Label only queries with a usable null distribution.
    scored_queries = df.loc[df['sgpval'].notna(), 'query'].unique()
    df = df[df['query'].isin(scored_queries)].copy()

    fa_id_label_map = get_seq_label_map(fasta_url)

    categories = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_fa_id = row['query']
        tgt_fa_id = row['target']
        # Normalize the query id too: multi-model (NMR) structures are split by
        # foldseek createdb into per-model entries (e.g. 'd1wgxa1_MODEL_20_A'),
        # which are not keys in the ASTRAL fasta map. get_original_id() reduces
        # them back to the base ASTRAL id (e.g. 'd1wgxa1').
        origin_src_fa_id = get_original_id(src_fa_id)
        origin_tgt_fa_id = get_original_id(tgt_fa_id)
        if '.' in tgt_fa_id:  # strange ids, e.g., 'd1pvc.1_4', 'd1apy.1_B'
            categories.append('-2')
        else:
            src_labels = fa_id_label_map[origin_src_fa_id]
            tgt_labels = fa_id_label_map[origin_tgt_fa_id]
            categories.append(check_multi_level_homolog(src_labels, tgt_labels))

    df['category'] = categories
    df['sgpval'] = df['sgpval'].map(lambda x: '{:.2e}'.format(x) if isinstance(x, float) else x)
    df.to_csv(out_path, sep='\t', index=False)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Foldseek wrapper that appends SG-pvalue to search output.'
    )
    parser.add_argument('--foldseek_url', type=str, default='foldseek',
                        help='Path to foldseek executable (default: foldseek in PATH)')
    parser.add_argument('--lookup_url', type=str, required=True,
                        help='SG lookup table (space-delimited, 2 cols: s_hat, p-value)')
    parser.add_argument('--num_shuffle', type=int, required=True,
                        help='Number of shuffled structures for null distribution (passed to structureshuffle --num-shuffles)')
    parser.add_argument('--keep-search-results', action='store_true',
                        help='Keep raw original/shuffled hits and intermediate files (default: only the _sgpval.tsv result)')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='Overwrite existing intermediate files (default: 0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for shuffling (passed as --seed to structureshuffle). Omit for random.')

    subparsers = parser.add_subparsers(dest='subcommand', help='sub-command help')
    fs_parser = subparsers.add_parser('easy-search', help='foldseek easy-search')
    fs_parser.add_argument('--keep-search-results', action='store_true', default=argparse.SUPPRESS,
                           help='Keep raw original/shuffled hits and intermediate files (default: only the _sgpval.tsv result)')

    # Required positional-style args (passed as flags for clarity)
    fs_parser.add_argument('-query', type=str, required=True,
                           help='Query PDB/mmCIF file, directory, or pre-built foldseek DB')
    fs_parser.add_argument('-db', type=str, required=True,
                           help='Target: directory of PDB/mmCIF files OR pre-built foldseek DB. '
                                'A directory is indexed automatically by easy-search (cached in tmp).')
    fs_parser.add_argument('-out', type=str, required=True,
                           help='Output basename; final result is <stem>_sgpval.tsv, e.g. result.tsv -> result_sgpval.tsv (raw TSV kept with --keep-search-results)')

    # Optional foldseek search parameters
    fs_parser.add_argument('--threads', type=int, help='Number of threads')
    fs_parser.add_argument('-s', type=float, help='Sensitivity (default foldseek: 9.5)')
    fs_parser.add_argument('--max-seqs', dest='max_seqs', type=int,
                           help='Maximum number of target sequences to report per query')
    fs_parser.add_argument('-e', type=float, help='E-value threshold')
    fs_parser.add_argument('--alignment-type', dest='alignment_type', type=int,
                           help='Alignment type for real search: 0=3Di, 1=TM-align, 2=3Di+AA (default)')
    fs_parser.add_argument('--shuf-alignment-type', dest='shuf_alignment_type', type=int,
                           help='Alignment type for shuffled null search (default: same as --alignment-type). '
                                'E.g. use 0 (3Di only) for null while real search uses 2 (3Di+AA).')
    fs_parser.add_argument('-c', type=float, help='Minimum coverage')
    fs_parser.add_argument('--cov-mode', dest='cov_mode', type=int, help='Coverage mode')
    fs_parser.add_argument('--exhaustive-search', dest='exhaustive_search', type=int,
                           help='Skip prefilter, exhaustive alignment (0/1)')
    fs_parser.add_argument('--comp-bias-corr', dest='comp_bias_corr', type=int,
                           help='Compositional bias correction for real search (0/1)')
    fs_parser.add_argument('--shuf-comp-bias-corr', dest='shuf_comp_bias_corr', type=int,
                           help='Compositional bias correction for shuffled null search (default: same as --comp-bias-corr)')
    fs_parser.add_argument('--shuf-gap-open', dest='shuf_gap_open', type=int, default=None,
                           help='Gap-open cost for the shuffled NULL search ONLY (foldseek default: 10). '
                                'Lower it (e.g. 6) to raise the shuffled-null bit scores a little; the real '
                                'search keeps the default gap-open. Applies to the aa/3Di slot.')
    fs_parser.add_argument('--fasta_url', type=str, default=None,
                           help='FASTA file with SCOP labels in description (optional). '
                                'If provided, annotates each hit with SCOP homology category.')

    args, extra_argv = parser.parse_known_args()
    if any(token.split('=', 1)[0] == '--markov-order' for token in extra_argv):
        parser.error('--markov-order is not configurable; the wrapper uses the default order of 1')
    args.extra_kwargs = parse_extra_argv(extra_argv)
    if args.extra_kwargs:
        print('Pass-through kwargs: {}'.format(args.extra_kwargs))
    if args.subcommand is None:
        parser.print_help()
    else:
        main(args)

# Example command
# --fasta_url is optional and enables SCOP label annotation):
# python src/foldseek.py \
#   --foldseek_url foldseek \
#   --lookup_url data/lookup/m50.txt \
#   --num_shuffle 50 \
#   easy-search \
#   -query query.pdb \
#   -db ~/scop40_unique \
#   -out result.tsv \
#   --threads 32 -s 10.5 --max-seqs 1000 -e 1e9 \
#   --alignment-type 2 \
#   --shuf-gap-open 2
