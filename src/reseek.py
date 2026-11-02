import os, argparse, subprocess, hashlib
from contextlib import nullcontext
from tempfile import TemporaryDirectory
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd

def main(args):
    check_args(args)
    query_bca, target_bca, output_tsv, search_kwargs = parse_search_args(args)
    print('query_bca={}\ttarget_bca={}\toutput_tsv={}'.format(query_bca, target_bca, output_tsv))

    output_dir = os.path.dirname(os.path.abspath(output_tsv))
    os.makedirs(output_dir, exist_ok=True)
    sig = arg_signature(query_bca, target_bca, search_kwargs)
    retained_work_dir = os.path.join(output_dir, 'reseek_output_{}'.format(sig))
    # Only this run's temporary directory is cleaned; existing caches are untouched.
    workspace = (nullcontext(retained_work_dir) if args.keep_search_results else
                 TemporaryDirectory(prefix='.reseek-', dir=output_dir))
    with workspace as work_dir:
        real_tsv = output_tsv if args.keep_search_results else os.path.join(work_dir, 'real_result.tsv')
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Run shuffled queries to build null distribution
        # Reseek handles shuffling internally via -num_shuffles N.
        # Shuffled query names: {original}_shuffle{i} (0-indexed).
        shuf_tsv = os.path.join(work_dir, 'shuf_result.tsv')
        if not os.path.exists(shuf_tsv) or args.overwrite > 0:
            cmd = build_search_cmd(args.reseek_url, query_bca, target_bca, shuf_tsv, search_kwargs,
                                   num_shuffles=args.num_shuffle)
            print('Searching shuffled queries: {}'.format(cmd))
            subprocess.run(cmd, check=True)

        # Step 2: Build null distribution from shuffled output
        print('Building null distribution from: {}'.format(shuf_tsv))
        null_dist = build_null_dist(shuf_tsv)
        print('null_dist: {} queries'.format(len(null_dist)))

        # Step 3: Run real search (no -num_shuffles)
        if not os.path.exists(real_tsv) or args.overwrite > 0:
            cmd = build_search_cmd(args.reseek_url, query_bca, target_bca, real_tsv, search_kwargs,
                                   num_shuffles=None)
            print('Searching real queries: {}'.format(cmd))
            subprocess.run(cmd, check=True)
            ensure_header(real_tsv, 'aq\tquery\ttarget\tevalue\tpvalue')

        # Step 4: Load SG lookup table
        print('Loading lookup table: {}'.format(args.lookup_url))
        lookup_mat = np.loadtxt(args.lookup_url)
        lookup_keys = lookup_mat[:, 0].flatten()
        print('lookup_mat={}'.format(lookup_mat.shape))

        # Step 5: Annotate real hits with SG-pvalue
        new_tsv = sgpval_output_path(output_tsv)
        add_sgpval(real_tsv, new_tsv, null_dist, lookup_keys, lookup_mat)
        print('Output with SG-pvalue written to: {}'.format(new_tsv))

        # Step 6: Annotate with SCOP labels (optional)
        if args.fasta_url is not None:
            labeled_tsv = add_scop_labels(new_tsv, args.fasta_url, output_dir)
            print('Output with SCOP labels written to: {}'.format(labeled_tsv))


def sgpval_output_path(output_tsv):
    """Final result next to -out/--output: result.tsv -> result_sgpval.tsv (other names get _sgpval.tsv appended)."""
    base, ext = os.path.splitext(output_tsv)
    return (base if ext == '.tsv' else output_tsv) + '_sgpval.tsv'


def check_args(args):
    assert os.path.isfile(args.lookup_url), 'lookup_url not found: {}'.format(args.lookup_url)
    assert args.reseek_url is not None
    assert args.num_shuffle > 0


def parse_search_args(args):
    # kwargs keys are the actual reseek flags (single-dash)
    kwargs = {}
    # Enforce required column output for SG-pvalue computation (AQ, query, target, evalue, pvalue)
    kwargs['-columns'] = 'aq+query+target+evalue+pvalue'

    if args.evalue is not None:         kwargs['-evalue'] = args.evalue
    if args.verysensitive:              kwargs['-verysensitive'] = ''   # flag only
    if args.sensitive:                  kwargs['-sensitive'] = ''       # flag only
    if args.threads is not None:        kwargs['-threads'] = args.threads
    if args.minlongdim is not None:     kwargs['-minlongdim'] = args.minlongdim
    if args.maxlongdim is not None:     kwargs['-maxlongdim'] = args.maxlongdim

    assert args.search is not None, '--search is required'
    assert args.db is not None, '--db is required'
    assert args.output is not None, '--output is required'

    return args.search, args.db, args.output, kwargs


def build_search_cmd(reseek_url, query_bca, target_bca, output_tsv, kwargs, num_shuffles=None):
    cmd = [reseek_url, '-search', query_bca, '-db', target_bca, '-output', output_tsv]
    for k, v in kwargs.items():
        if v == '':
            # Flag-only argument (e.g. -verysensitive)
            cmd.append(k)
        else:
            cmd += [k, str(v)]
    if num_shuffles is not None:
        cmd += ['-num_shuffles', str(num_shuffles)]
    return cmd


def arg_signature(query_bca, target_bca, kwargs):
    parts = ['query', os.path.abspath(query_bca), 'target', os.path.abspath(target_bca)]
    for k in sorted(kwargs.keys()):
        parts += [k, str(kwargs[k])]
    arg_str = ''.join(parts)
    print('arg_str={}'.format(arg_str))
    return hashlib.sha256(arg_str.encode()).hexdigest()[:12]


def build_null_dist(shuf_tsv):
    """
    Parse reseek shuffled-query output TSV (columns: aq, query, target, evalue, pvalue).
    Shuffled query names use the suffix '_shuffle{N}' (0-indexed).
    Returns: {original_id: np.array of best-AQ-per-shuffle}
    """
    # best_aq[shuffled_name] = max AQ across all targets for that shuffled query
    best_aq = {}
    with open(shuf_tsv) as f:
        for line in f:
            row = line.strip().split('\t')
            if len(row) < 2:
                continue
            try:
                aq = float(row[0])
                query_name = row[1]
            except ValueError:
                continue
            if query_name not in best_aq or aq > best_aq[query_name]:
                best_aq[query_name] = aq

    # Group by original ID by stripping '_shuffle{N}' suffix
    grouped = {}
    for shuf_name, a in best_aq.items():
        if '_shuffle' in shuf_name:
            original_id = shuf_name[:shuf_name.rfind('_shuffle')]
        else:
            original_id = shuf_name
        if original_id not in grouped:
            grouped[original_id] = []
        grouped[original_id].append(a)

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


def ensure_header(tsv_path, header):
    """Prepend a header line to a TSV file if it doesn't already have one."""
    with open(tsv_path, 'r') as f:
        first_line = f.readline()
    if first_line.startswith('aq\t'):
        return
    with open(tsv_path, 'r') as f:
        content = f.read()
    with open(tsv_path, 'w') as f:
        f.write(header + '\n' + content)


def add_sgpval(input_tsv, output_tsv, null_dist, lookup_keys, lookup_mat):
    """
    Read reseek output TSV (aq, query, target, evalue, pvalue) and append sg_pvalue column.
    Output is sorted by AQ descending.
    """
    missing_queries = set()
    rows_out = []
    with open(input_tsv) as fin:
        for line in fin:
            if line.startswith('aq\t'):
                continue
            row = line.rstrip('\n').split('\t')
            if len(row) < 2:
                continue
            try:
                aq = float(row[0])
                query_name = row[1]
            except ValueError:
                continue

            if query_name not in null_dist:
                missing_queries.add(query_name)
                rows_out.append((aq, row, 'NA'))
                continue

            sg_pv = compute_sgpval(aq, null_dist[query_name], lookup_keys, lookup_mat)
            rows_out.append((aq, row, '{:.6e}'.format(sg_pv)))

    rows_out.sort(key=lambda x: x[0], reverse=True)
    with open(output_tsv, 'w') as fout:
        fout.write('aq\tquery\ttarget\tevalue\tpvalue\tsgpval\n')
        for aq, row, sgpval in rows_out:
            fout.write('\t'.join(row) + '\t' + sgpval + '\n')

    if missing_queries:
        print('WARNING: {} queries had no null distribution (not in shuffled output): {}'.format(
            len(missing_queries), list(missing_queries)[:5]))


# --- SCOP label annotation (from analyze_label.py) ---

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
    # Reseek appends chain suffix (_A, _B, etc.) to BCA entry names.
    # Strip the last two chars when the second-to-last char is '_'.
    if len(fa_id) > 2 and fa_id[-2] == '_':
        fa_id = fa_id[:-2]
    return fa_id


def add_scop_labels(sgpval_tsv, fasta_url, output_dir):
    out_dir = os.path.join(output_dir, 'analyze_label')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(sgpval_tsv))

    df = pd.read_csv(sgpval_tsv, sep='\t')
    fa_id_label_map = get_seq_label_map(fasta_url)

    categories = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_fa_id = row['query']
        tgt_fa_id = row['target']
        origin_src_fa_id = get_original_id(src_fa_id)
        origin_tgt_fa_id = get_original_id(tgt_fa_id)
        if '.' in tgt_fa_id:  # strange ids, e.g., 'd1pvc.1_4', 'd1apy.1_B'
            categories.append('-2')
        elif origin_src_fa_id not in fa_id_label_map or origin_tgt_fa_id not in fa_id_label_map:
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
        description='Reseek wrapper that appends SG-pvalue to search output.'
    )
    parser.add_argument('--reseek_url', type=str, default='reseek',
                        help='Path to reseek executable (default: reseek on PATH)')
    parser.add_argument('--lookup_url', type=str, required=True,
                        help='SG lookup table (space-delimited, 2 cols: s_hat, p-value)')
    parser.add_argument('--num_shuffle', type=int, required=True,
                        help='Number of shuffled queries for null distribution')
    parser.add_argument('--keep-search-results', action='store_true',
                        help='Keep raw original/shuffled hits and intermediate files (default: only the _sgpval.tsv result)')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='Overwrite existing intermediate files (default: 0)')

    subparsers = parser.add_subparsers(dest='subcommand', help='sub-command help')
    rs_parser = subparsers.add_parser('search', help='reseek search')
    rs_parser.add_argument('--keep-search-results', action='store_true', default=argparse.SUPPRESS,
                           help='Keep raw original/shuffled hits and intermediate files (default: only the _sgpval.tsv result)')

    # Required args (double-dash in Python; mapped to single-dash reseek flags internally)
    rs_parser.add_argument('--search', type=str, required=True,
                           help='Query BCA file (converted with: reseek -convert input.pdb -bca input.bca)')
    rs_parser.add_argument('--db', type=str, required=True,
                           help='Target BCA file')
    rs_parser.add_argument('--output', type=str, required=True,
                           help='Output basename; final result is <stem>_sgpval.tsv, e.g. result.tsv -> result_sgpval.tsv (raw TSV kept with --keep-search-results)')

    # Optional reseek search parameters.
    # Double-dash in Python argparse; these are mapped to single-dash reseek flags internally.
    rs_parser.add_argument('--evalue', type=float, help='E-value threshold (reseek: -evalue)')
    rs_parser.add_argument('--verysensitive', action='store_true',
                           help='Enable very sensitive search mode (reseek: -verysensitive)')
    rs_parser.add_argument('--sensitive', action='store_true',
                           help='Enable sensitive search mode (reseek: -sensitive)')
    rs_parser.add_argument('--threads', type=int, help='Number of threads (reseek: -threads)')
    rs_parser.add_argument('--minlongdim', type=int, help='Minimum long dimension filter')
    rs_parser.add_argument('--maxlongdim', type=int, help='Maximum long dimension filter')
    rs_parser.add_argument('--fasta_url', type=str, default=None,
                           help='FASTA file with SCOP labels in description (optional). '
                                'If provided, annotates each hit with SCOP homology category.')

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
    else:
        main(args)

# Example command (--fasta_url is optional and enables SCOP label annotation):
# python src/reseek.py \
#   --reseek_url reseek \
#   --lookup_url data/pvs_f0_t34_b0.001_N1e10_m50_ls33_ss15_l0_s3.txt \
#   --num_shuffle 50 \
#   search \
#   --search data/d2wv.bca \
#   --db data/simulation/scop_target.bca \
#   --output result/bestAQ_d2wv.tsv \
#   --evalue 100 \
#   --verysensitive \
#   --fasta_url data/scop40_run/astral.fa

