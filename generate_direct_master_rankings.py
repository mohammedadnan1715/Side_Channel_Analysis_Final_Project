#!/usr/bin/env python3
"""Generate approximate K0 rankings directly from the timing objectives.

This script intentionally does not construct candidates by fixing 15 bytes of a
recovered master key. It searches the timing objective in K10 space, reverses a
large timing-derived ensemble of complete K10 candidates to K0, and computes a
max-profile score for every K0 byte value.

Plaintexts and candidate verification are not used to order the rankings. An
optional reference key is accepted only after ranking generation so the rank of
an already verified key can be reported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import aes_timing_attack as attack


RCONS = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54], dtype=np.uint8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expand_master_batch(masters: np.ndarray) -> np.ndarray:
    """Vectorized AES-128 K0 -> K10 expansion for shape (N,16)."""
    masters = np.asarray(masters, dtype=np.uint8)
    if masters.ndim != 2 or masters.shape[1] != 16:
        raise ValueError('Expected master keys with shape (N,16)')
    words = np.empty((len(masters), 44, 4), dtype=np.uint8)
    words[:, :4] = masters.reshape(-1, 4, 4)
    for i in range(4, 44):
        temp = words[:, i - 1].copy()
        if i % 4 == 0:
            temp = attack.SBOX[temp[:, [1, 2, 3, 0]]]
            temp[:, 0] ^= RCONS[i // 4]
        words[:, i] = words[:, i - 4] ^ temp
    return words[:, 40:44].reshape(-1, 16)


def invert_round10_batch(round10: np.ndarray) -> np.ndarray:
    """Vectorized AES-128 K10 -> K0 inversion for shape (N,16)."""
    round10 = np.asarray(round10, dtype=np.uint8)
    if round10.ndim != 2 or round10.shape[1] != 16:
        raise ValueError('Expected round-10 keys with shape (N,16)')
    words = np.empty((len(round10), 44, 4), dtype=np.uint8)
    words[:, 40:44] = round10.reshape(-1, 4, 4)
    for i in range(43, 3, -1):
        temp = words[:, i - 1].copy()
        if i % 4 == 0:
            temp = attack.SBOX[temp[:, [1, 2, 3, 0]]]
            temp[:, 0] ^= RCONS[i // 4]
        words[:, i - 4] = words[:, i] ^ temp
    return words[:, :4].reshape(-1, 16)


def self_test_batch_schedule() -> None:
    rng = np.random.default_rng(9941)
    masters = rng.integers(0, 256, (32, 16), dtype=np.uint8)
    rounds = expand_master_batch(masters)
    recovered = invert_round10_batch(rounds)
    if not np.array_equal(recovered, masters):
        raise AssertionError('Vectorized key-schedule round trip failed')
    for i in range(8):
        expected = np.frombuffer(attack.expand_key(masters[i].tobytes())[10], np.uint8)
        if not np.array_equal(rounds[i], expected):
            raise AssertionError('Vectorized expansion disagrees with scalar AES schedule')
        expected_master = np.frombuffer(attack.invert_round10(rounds[i].tobytes()), np.uint8)
        if not np.array_equal(masters[i], expected_master):
            raise AssertionError('Vectorized inversion disagrees with scalar AES schedule')


def empty_profile():
    return (np.full((16, 256), -np.inf, dtype=np.float64),
            np.zeros((16, 256), dtype=np.int64))


def update_profile(profile: np.ndarray, counts: np.ndarray,
                   masters: np.ndarray, values: np.ndarray) -> None:
    masters = np.asarray(masters, dtype=np.uint8)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(masters) != len(values):
        raise ValueError('Candidate and score lengths differ')
    for i in range(16):
        np.maximum.at(profile[i], masters[:, i], values)
        np.add.at(counts[i], masters[:, i], 1)


def expanded_ensemble_profile(scores: np.ndarray, states: np.ndarray,
                              state_values: np.ndarray, contexts: int):
    """Profile K0 from exact one-K10-byte neighborhoods of timing optima."""
    profile, counts = empty_profile()
    states = states[:contexts]
    state_values = state_values[:contexts]
    all_values = np.arange(256, dtype=np.uint8)
    generated = 0
    for i in range(16):
        field = attack.local_field(scores, states, i, 'expanded')
        existing = field[np.arange(len(states)), states[:, i]]
        candidate_scores = state_values[:, None] - existing[:, None] + field
        candidates = np.repeat(states[:, None, :], 256, axis=1)
        candidates[:, :, i] = all_values
        flat_rounds = candidates.reshape(-1, 16)
        masters = invert_round10_batch(flat_rounds)
        update_profile(profile, counts, masters, candidate_scores.reshape(-1))
        generated += len(flat_rounds)
    return profile, counts, generated


def simple_ensemble_profile(scores: np.ndarray, states: np.ndarray,
                            state_values: np.ndarray, contexts: int,
                            neighbor_contexts: int, neighbor_values: int):
    """Profile K0 from relative K10 optima, anchors, and strong neighbors."""
    profile, counts = empty_profile()
    states = states[:contexts]
    state_values = state_values[:contexts]
    anchors = np.arange(256, dtype=np.uint8)
    generated = 0

    # Every absolute anchor is timing-equivalent for one relative solution.
    for start in range(0, len(states), 32):
        block = states[start:start + 32]
        values = state_values[start:start + 32]
        rounds = block[:, None, :] ^ anchors[None, :, None]
        masters = invert_round10_batch(rounds.reshape(-1, 16))
        update_profile(profile, counts, masters, np.repeat(values, 256))
        generated += len(masters)

    # Add strong one-coordinate neighborhoods without using a recovered key.
    base = states[:neighbor_contexts]
    base_values = state_values[:neighbor_contexts]
    for i in range(1, 16):
        field = attack.local_field(scores, base, i, 'simple')
        existing = field[np.arange(len(base)), base[:, i]]
        candidate_scores = base_values[:, None] - existing[:, None] + field
        take = min(neighbor_values, 256)
        best = np.argpartition(-candidate_scores, take - 1, axis=1)[:, :take]
        relatives = np.repeat(base[:, None, :], take, axis=1)
        relatives[:, :, i] = best.astype(np.uint8)
        chosen_scores = np.take_along_axis(candidate_scores, best, axis=1)
        for row in range(len(base)):
            rounds = relatives[row, :, None, :] ^ anchors[None, :, None]
            masters = invert_round10_batch(rounds.reshape(-1, 16))
            values = np.repeat(chosen_scores[row], 256)
            update_profile(profile, counts, masters, values)
            generated += len(masters)
    return profile, counts, generated


def fill_unseen_values(profile: np.ndarray, counts: np.ndarray, scores: np.ndarray,
                       method: str, rng: np.random.Generator,
                       contexts_per_value: int) -> int:
    """Give unseen K0 values direct timing support from random full contexts.

    These candidates are not built from a recovered key. The random contexts are
    only a completeness fallback; timing scores still determine their support.
    """
    generated = 0
    for position in range(16):
        missing = np.flatnonzero(counts[position] == 0)
        if not len(missing):
            continue
        masters = rng.integers(0, 256,
            (len(missing) * contexts_per_value, 16), dtype=np.uint8)
        masters[:, position] = np.repeat(missing.astype(np.uint8), contexts_per_value)
        rounds = expand_master_batch(masters)
        timing_values = attack.objective(scores, rounds, method)
        update_profile(profile, counts, masters, timing_values)
        generated += len(masters)
    return generated


def ranking_rows(profile: np.ndarray, counts: np.ndarray):
    tables = []
    for i in range(16):
        order = np.lexsort((np.arange(256), -profile[i]))
        tables.append([
            {'rank': rank, 'candidate_hex': f'{int(v):02x}',
             'timing_support': float(profile[i, v]),
             'candidate_contexts_seen': int(counts[i, v])}
            for rank, v in enumerate(order, 1)
        ])
    return tables


def write_csv(path: Path, tables) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['byte_position', 'rank', 'candidate_hex',
                         'timing_support', 'candidate_contexts_seen'])
        for i, table in enumerate(tables):
            for row in table:
                writer.writerow([i, row['rank'], row['candidate_hex'],
                                 row['timing_support'], row['candidate_contexts_seen']])


def correct_ranks(tables, reference: bytes | None):
    if reference is None:
        return None
    return [next(row['rank'] for row in tables[i]
                 if int(row['candidate_hex'], 16) == reference[i])
            for i in range(16)]


def write_pdf(report: dict, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body2', parent=styles['BodyText'],
        fontName='Helvetica', fontSize=10.2, leading=14, spaceAfter=8,
        textColor=colors.HexColor('#253047')))
    styles.add(ParagraphStyle(name='Caption2', parent=styles['BodyText'],
        fontName='Helvetica', fontSize=8.3, leading=10, spaceAfter=7))
    styles['Title'].textColor = colors.HexColor('#163454')
    doc = SimpleDocTemplate(str(path), pagesize=landscape(letter),
        leftMargin=30, rightMargin=30, topMargin=28, bottomMargin=28,
        title='Week 15 - Direct timing-leakage master-key rankings',
        author='CSEC 622 Group 15')
    story = [
        Paragraph('Direct timing-leakage master-key candidate rankings', styles['Title']),
        Paragraph('Week 15 | Simple and expanded final-round attacks', styles['Heading2']),
        Spacer(1, 6),
        Paragraph('<b>What changed.</b> These rankings do not hold 15 recovered K0 bytes fixed. '
                  'They are generated from complete K10 candidates found and scored using the '
                  'timing leakage.', styles['Body2']),
        Paragraph('<b>Candidate generation.</b> Multi-start timing search produces strong complete '
                  'K10 states. The expanded method adds every one-byte K10 neighbor around the '
                  'retained states. The simple method enumerates all absolute anchors for its '
                  'relative K10 states and adds strong one-coordinate neighbors.', styles['Body2']),
        Paragraph('<b>Mapping and ranking.</b> Every complete K10 candidate is reversed through the '
                  'AES-128 key schedule to K0. For K0 byte i and value v, support is the highest '
                  'original 120-pair timing objective among generated candidates whose resulting '
                  'K0[i] equals v.', styles['Body2']),
        Paragraph('<b>No post-recovery conditioning.</b> Recovered K0 bytes and plaintext/ciphertext '
                  'verification do not construct candidates or order the tables. If a reference key '
                  'is supplied, it is consulted only after generation to report its ranks.', styles['Body2']),
        Paragraph('<b>Interpretation.</b> This is an approximate max-profile over a finite candidate '
                  'ensemble, not an exhaustive search or calibrated probability. The simple attack '
                  'can contain large ties because timing does not resolve its shared XOR anchor.',
                  styles['Body2']),
        Paragraph(f'<b>Dataset SHA-256:</b> <font face="Courier">{report["dataset_sha256"]}</font>',
                  styles['Body2']),
    ]
    if report.get('reference_key_hex'):
        story.append(Paragraph(
            f'<b>Verified key used only for rank evaluation:</b> '
            f'<font face="Courier">{report["reference_key_hex"]}</font>', styles['Body2']))
    story.append(PageBreak())

    for method in ['simple', 'expanded']:
        info = report['methods'][method]
        for first in range(0, 256, 40):
            last = min(first + 40, 256)
            story.append(Paragraph(
                f'{method.title()} attack: direct timing-leakage K0 candidates', styles['Heading2']))
            story.append(Paragraph(
                f'Ranks {first + 1}-{last} of 256 | K0 byte positions 0-15 | '
                f'Approximate max-profile over complete timing-scored candidates', styles['Caption2']))
            data = [['Rank'] + [f'K0[{i}]' for i in range(16)]]
            for r in range(first, last):
                data.append([str(r + 1)] +
                    [info['rankings'][i][r]['candidate_hex'] for i in range(16)])
            table = Table(data, repeatRows=1,
                colWidths=[30] + [44] * 16,
                rowHeights=[18] + [11.4] * (last - first))
            head = '#173758' if method == 'simple' else '#0f766e'
            style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(head)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
                ('FONTSIZE', (0, 0), (-1, -1), 6.9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                    [colors.white, colors.HexColor('#eef2f6')]),
                ('LINEBELOW', (0, 0), (-1, 0), 0.4, colors.white),
            ]
            if first == 0:
                style.append(('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#dcfce7')))
                style.append(('FONTNAME', (0, 1), (-1, 1), 'Courier-Bold'))
            table.setStyle(TableStyle(style))
            story.extend([table, PageBreak()])

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(30, 14, 'Week 15 | Direct timing-leakage master-key rankings')
        canvas.drawRightString(762, 14, f'Page {canvas.getPageNumber()}')
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', type=Path)
    p.add_argument('--output', type=Path, default=Path('direct_master_rankings'))
    p.add_argument('--method', choices=['simple', 'expanded', 'both'], default='both')
    p.add_argument('--upper-quantile', type=float, default=1.0)
    p.add_argument('--transform', choices=['raw', 'log', 'rank'], default='rank')
    p.add_argument('--statistic', choices=['welch', 'difference'], default='welch')
    p.add_argument('--offset', type=int, choices=range(0, 32, 4), default=0)
    p.add_argument('--restarts', type=int, default=128)
    p.add_argument('--sweeps', type=int, default=40)
    p.add_argument('--contexts', type=int, default=64)
    p.add_argument('--simple-neighbor-contexts', type=int, default=8)
    p.add_argument('--simple-neighbor-values', type=int, default=32)
    p.add_argument('--supplement-contexts', type=int, default=8)
    p.add_argument('--seed', type=int, default=622)
    p.add_argument('--reference-key', help='Optional verified K0 hex; used only after rankings are generated')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.restarts, args.sweeps, args.contexts,
           args.simple_neighbor_contexts, args.simple_neighbor_values,
           args.supplement_contexts) < 1:
        raise SystemExit('Search and ensemble sizes must be positive')
    attack.self_test()
    self_test_batch_schedule()
    started = time.monotonic()
    pt, ct, timing, metadata = attack.load_dataset(args.dataset)
    del pt  # Plaintexts are intentionally unavailable to ranking generation.
    prep_args = SimpleNamespace(upper_quantile=args.upper_quantile,
        transform=args.transform, max_traces=0, seed=args.seed)
    chosen_ct, chosen_t, preprocessing = attack.preprocess(ct, timing, prep_args)
    digest = sha256_file(args.dataset)
    reference = bytes.fromhex(args.reference_key) if args.reference_key else None
    if reference is not None and len(reference) != 16:
        raise SystemExit('--reference-key must contain exactly 32 hexadecimal characters')

    report = {
        'dataset': args.dataset.name,
        'dataset_sha256': digest,
        'ranking_scope': 'direct timing-leakage approximate max-profile K0 byte rankings',
        'score_source': 'original 120-pair timing objective',
        'candidate_generation_uses_recovered_key': False,
        'plaintext_verification_used_for_ranking': False,
        'preprocessing': preprocessing,
        'search': {'restarts': args.restarts, 'sweeps': args.sweeps,
                   'contexts': args.contexts, 'seed': args.seed},
        'reference_key_hex': reference.hex() if reference else None,
        'methods': {},
    }
    methods = ['simple', 'expanded'] if args.method == 'both' else [args.method]
    args.output.mkdir(parents=True, exist_ok=True)

    for method_index, method in enumerate(methods):
        method_started = time.monotonic()
        attack.log('\nDIRECT MASTER RANKING METHOD:', method)
        pair_scores = attack.build_scores(chosen_ct, chosen_t, method,
            args.offset, args.statistic)
        states, state_values = attack.optimize(pair_scores, method,
            args.restarts, args.sweeps, args.seed)
        contexts = min(args.contexts, len(states))
        if method == 'expanded':
            profile, counts, generated = expanded_ensemble_profile(
                pair_scores, states, state_values, contexts)
        else:
            profile, counts, generated = simple_ensemble_profile(
                pair_scores, states, state_values, contexts,
                min(args.simple_neighbor_contexts, contexts),
                args.simple_neighbor_values)
        rng = np.random.default_rng(args.seed + 1000 + method_index)
        supplemented = fill_unseen_values(profile, counts, pair_scores,
            method, rng, args.supplement_contexts)
        if not np.isfinite(profile).all() or np.any(counts == 0):
            raise RuntimeError('Candidate ensemble failed to cover all 256 K0 values')

        tables = ranking_rows(profile, counts)
        ranks = correct_ranks(tables, reference)
        csv_path = args.output / f'{method}_direct_master_rankings.csv'
        write_csv(csv_path, tables)
        np.savez_compressed(args.output / f'{method}_direct_profile.npz',
            support=profile, contexts_seen=counts)
        info = {
            'rankings': tables,
            'candidate_count_before_completeness_supplement': generated,
            'random_completeness_candidates': supplemented,
            'all_values_covered': True,
            'minimum_contexts_seen_for_any_value': int(counts.min()),
            'maximum_contexts_seen_for_any_value': int(counts.max()),
            'best_timing_search_state_hex': states[0].tobytes().hex(),
            'best_timing_search_score': float(state_values[0]),
            'correct_value_ranks': ranks,
            'mean_correct_value_rank': float(np.mean(ranks)) if ranks else None,
            'elapsed_seconds': round(time.monotonic() - method_started, 3),
            'csv_file': csv_path.name,
        }
        report['methods'][method] = info
        attack.log(json.dumps({k: v for k, v in info.items() if k != 'rankings'}, indent=2))
        del pair_scores

    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    json_path = args.output / 'direct_master_rankings.json'
    json_path.write_text(json.dumps(report, indent=2))
    write_pdf(report, args.output / 'Week15_Direct_Master_Key_Rankings.pdf')
    attack.log('Wrote direct rankings to', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
