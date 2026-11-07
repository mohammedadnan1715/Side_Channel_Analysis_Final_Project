#!/usr/bin/env python3
"""Post-recovery conditional MASTER-key rankings for both final-round attacks.

Default: derives an explicitly approximate additive score from saved profile
supports. Optional --exact-dataset recomputes each original joint timing score
using a LOCAL dataset. No downloads are performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import aes_timing_attack as attack


def read_support(path, method):
    result = np.full((16, 256), np.nan)
    seen = set()
    expected = 'round10_byte' if method == 'expanded' else 'round10_byte_XOR_round10_byte_0'
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            i, v = int(row['byte_position']), int(row['candidate_hex'], 16)
            if not 0 <= i < 16 or not 0 <= v < 256 or (i, v) in seen:
                raise ValueError('Invalid or duplicate support entry')
            if row['meaning'] != expected:
                raise ValueError(f'Unexpected support representation in {path}')
            seen.add((i, v))
            result[i, v] = float(row['support'])
    if len(seen) != 4096 or np.isnan(result).any():
        raise ValueError('Expected all 256 candidates for all 16 source byte positions')
    if method == 'expanded' and not np.isfinite(result).all():
        raise ValueError('Expanded support values must be finite')
    if method == 'simple' and (not np.isfinite(result[1:]).all() or not np.isfinite(result[0, 0])):
        raise ValueError('Simple non-reference support values must be finite')
    return result


def master_variants(master):
    """Row (i*256+v) changes only byte i of the recovered master key to v."""
    candidates = np.tile(np.frombuffer(master, np.uint8), (4096, 1))
    for i in range(16):
        candidates[i * 256:(i + 1) * 256, i] = np.arange(256, dtype=np.uint8)
    rounds = np.array([list(attack.expand_key(k.tobytes())[10]) for k in candidates], np.uint8)
    return candidates, rounds


def saved_profile_scores(profile, rounds, reference_round10, method):
    """Surrogate only: not an exact reconstruction of the joint pair objective."""
    coordinates = rounds if method == 'expanded' else rounds ^ rounds[:, :1]
    reference = reference_round10 if method == 'expanded' else reference_round10 ^ reference_round10[0]
    baseline = profile[np.arange(16), reference]
    # Centering makes the verified reference key have score zero. Scores are
    # not calibrated probabilities, and correlated supports are being added.
    selected = profile[np.arange(16)[None, :], coordinates]
    if not np.isfinite(selected).all():
        raise ValueError('A queried support was not finite')
    return (selected - baseline).sum(axis=1).reshape(16, 256)


def records_for(scores, master):
    rows = []
    for i in range(16):
        order = np.lexsort((np.arange(256), -scores[i]))
        rows.append([{'rank': int(rank), 'candidate_hex': f'{int(v):02x}',
                      'score_delta': float(scores[i, v])}
                     for rank, v in enumerate(order, 1)])
        if sorted(int(r['candidate_hex'], 16) for r in rows[-1]) != list(range(256)):
            raise AssertionError('Candidate completeness check failed')
    correct_ranks = [next(r['rank'] for r in rows[i] if int(r['candidate_hex'], 16) == master[i])
                     for i in range(16)]
    return rows, correct_ranks


def build_report(inputs, exact_dataset=None):
    attack.self_test()
    sources = {method: json.loads((inputs / f'{method}_result.json').read_text())
               for method in ['simple', 'expanded']}
    simple, expanded = sources['simple'], sources['expanded']
    for field in ['master_key_hex', 'round10_key_hex', 'dataset_sha256']:
        if simple[field] != expanded[field]:
            raise ValueError(f'The source reports disagree on {field}')
    if any(d['status'] != 'recovered' for d in sources.values()):
        raise ValueError('This conditional analysis requires completed key recovery')
    master = bytes.fromhex(simple['master_key_hex'])
    round10 = bytes.fromhex(simple['round10_key_hex'])
    if attack.expand_key(master)[10] != round10 or attack.invert_round10(round10) != master:
        raise ValueError('Recovered master and round-10 keys disagree')
    candidates, rounds = master_variants(master)
    reference = np.frombuffer(round10, np.uint8)
    dataset_arrays = None
    if exact_dataset is not None:
        digest = hashlib.sha256()
        with Path(exact_dataset).open('rb') as f:
            while chunk := f.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != simple['dataset_sha256']:
            raise ValueError('Local dataset SHA256 does not match the supplied recovery reports')
        pt, ct, timing, _ = attack.load_dataset(exact_dataset)
        for start in range(0, len(pt), 16384):
            if not np.array_equal(attack.encrypt_blocks(master, pt[start:start+16384]), ct[start:start+16384]):
                raise ValueError('Master-key verification failed on local dataset')
        dataset_arrays = ct, timing
    mode = 'original_joint_score' if exact_dataset else 'saved_profile_surrogate'
    report = {'dataset': simple['dataset'], 'dataset_sha256': simple['dataset_sha256'],
              'master_key_hex': master.hex(), 'round10_key_hex': round10.hex(),
              'ranking_scope': 'post-recovery conditional master-key byte rankings',
              'score_mode': mode,
              'source_report_verification_pairs': {k: v['verification_pairs'] for k, v in sources.items()},
              'generation_checks': ['source reports agree', 'AES-128 forward/reverse key schedules agree',
                                     'all 256 values present exactly once at all 16 positions per method'],
              'methodology': {
                  'candidate_construction': 'Vary one K0 byte through 00..ff; hold the other 15 recovered K0 bytes fixed.',
                  'key_schedule': 'Expand each complete candidate K0 into its candidate K10.',
                  'saved_profile_score': 'Sum_j(profile_j(candidate_coordinate_j) - profile_j(recovered_coordinate_j)). '
                      'Coordinates are K10[j] for expanded and K10[j] XOR K10[0] for simple.',
                  'joint_score': 'When --exact-dataset is used, evaluate the original sum of 120 pair scores '
                      'on each expanded candidate instead of adding saved profile supports.',
                  'ordering': 'Descending score; exact ties are ordered by ascending hexadecimal byte value.',
                  'limitations': 'Conditional on the other 15 recovered bytes. Saved-profile mode uses an additive '
                      'surrogate over correlated max-profile supports, not the original joint score or a probability. '
                      'These are not independent pre-recovery byte rankings.'},
              'methods': {}}
    for method, source in sources.items():
        if dataset_arrays is None:
            profile = read_support(inputs / f'{method}_rankings.csv', method)
            scores = saved_profile_scores(profile, rounds, reference, method)
        else:
            prep = source['preprocessing']
            args = SimpleNamespace(upper_quantile=prep['upper_quantile'], transform=prep['transform'],
                                   max_traces=0, seed=source['seed'])
            ct, timing = dataset_arrays
            chosen_ct, chosen_t, selected = attack.preprocess(ct, timing, args)
            if selected['selected_traces'] != prep['selected_traces']:
                raise ValueError('This script cannot reproduce a subsampled source run from the available metadata')
            pair_scores = attack.build_scores(chosen_ct, chosen_t, method,
                source['table_offset'] or 0, source['statistic'])
            baseline = attack.objective(pair_scores, reference[None, :], method)[0]
            scores = (attack.objective(pair_scores, rounds, method) - baseline).reshape(16, 256)
            del pair_scores
        rows, ranks = records_for(scores, master)
        report['methods'][method] = {'rankings': rows, 'correct_value_ranks': ranks,
            'mean_correct_value_rank': float(np.mean(ranks)), 'candidate_count': 4096,
            'tied_score_groups': [int(256 - len(np.unique(scores[i]))) for i in range(16)]}
    return report


def write_pdf(report, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='IntroText', fontName='Helvetica', fontSize=10.5,
                              leading=15, spaceAfter=9, textColor=colors.HexColor('#253047')))
    styles.add(ParagraphStyle(name='TableCaption', fontName='Helvetica', fontSize=8.5,
                              leading=11, spaceAfter=7))
    title_color = colors.HexColor('#163454')
    styles['Title'].textColor = title_color
    doc = SimpleDocTemplate(str(path), pagesize=landscape(letter),
                            leftMargin=32, rightMargin=32, topMargin=30, bottomMargin=32,
                            title='Week 15 - Conditional master-key candidate rankings',
                            author='Research project team')
    story = [Paragraph('Appendix: master-key candidate rankings', styles['Title']),
             Paragraph('Week 15 | Simple and expanded final-round attacks', styles['Heading2']),
             Spacer(1, 8)]
    mode_text = ('Original joint timing score, recomputed from the local traces.'
                 if report['score_mode'] == 'original_joint_score' else
                 'Additive surrogate derived from the saved per-byte max-profile support tables. '
                 'The original joint timing score cannot be reconstructed from those CSVs alone.')
    paragraphs = [
        '<b>Scope.</b> Each method lists every value 00-ff exactly once for each master-key byte '
        'K0[0]-K0[15]. Byte indices follow the left-to-right byte order of the hexadecimal key.',
        f'<b>Recovered master key:</b> <font face="Courier">{report["master_key_hex"]}</font><br/>'
        f'<b>Recovered round-10 key:</b> <font face="Courier">{report["round10_key_hex"]}</font>',
        '<b>Conditional construction.</b> For byte i and candidate v, replace only byte i of '
        'the recovered master key with v. Keep the other 15 recovered master-key bytes fixed. '
        'Expand each resulting complete key to obtain its round-10 key.',
        '<b>Scoring mode.</b> ' + mode_text,
        '<b>Saved-support formula.</b> Let p[j,x] be the supplied support for coordinate x at '
        'round-key position j. Use x[j] = K10[j] for the expanded method and '
        'x[j] = K10[j] XOR K10[0] for the simple method. The candidate score is '
        'sum over j of {p[j,x[j](candidate)] - p[j,x[j](recovered)]}. '
        'In exact joint-score mode this sum is replaced by the original 120-pair objective.',
        '<b>Ordering and interpretation.</b> Higher scores rank first; exact ties use ascending '
        'hexadecimal value. These are post-recovery conditional rankings, not independent '
        'pre-recovery estimates. Saved supports are correlated, and their sum is neither a '
        'probability nor the original joint attack objective. The full scoring note must '
        'accompany these tables.',
        '<b>Checks.</b> The two supplied recovery reports agree and each records verification '
        'of 1,048,576 plaintext/ciphertext pairs. This appendix generation checks key-schedule '
        'consistency and candidate completeness; saved-support mode does not reprocess raw traces.',
        '<b>Reading the tables.</b> Each row is a rank and each column is a master-key byte. '
        'Read down a column to obtain its ordered list of 256 candidates. All entries are hexadecimal.',
        '<b>Source files.</b> simple_result.json, expanded_result.json, simple_rankings.csv, '
        'and expanded_rankings.csv supplied from the Week 15 run.'
    ]
    for text in paragraphs:
        story.append(Paragraph(text, styles['IntroText']))
    for method in ['simple', 'expanded']:
        info = report['methods'][method]
        accent = colors.HexColor('#163454' if method == 'simple' else '#15645D')
        for first in range(0, 256, 40):
            stop = min(first + 40, 256)
            story.append(PageBreak())
            story.append(Paragraph(f'{method.title()} final-round attack: master-key candidates', styles['Heading2']))
            story.append(Paragraph(f'Ranks {first+1}-{stop} of 256 | K0 byte positions 0-15 | '
                f'Post-recovery conditional rankings | Score mode: {report["score_mode"]}', styles['TableCaption']))
            data = [['Rank'] + [f'K0[{i}]' for i in range(16)]]
            for rank in range(first, stop):
                data.append([str(rank + 1)] + [info['rankings'][i][rank]['candidate_hex'] for i in range(16)])
            table = Table(data, colWidths=[32] + [43.5] * 16,
                          rowHeights=[16] + [11] * (stop - first), repeatRows=1)
            commands = [('BACKGROUND', (0, 0), (-1, 0), accent),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 1),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, accent),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F4F8')])]
            if first == 0:
                commands += [('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#DFEEE6')),
                             ('FONTNAME', (0, 1), (-1, 1), 'Courier-Bold')]
            table.setStyle(TableStyle(commands))
            story.append(table)
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor('#526071'))
        canvas.setFont('Helvetica', 8)
        canvas.drawString(32, 17, 'Week 15 | Conditional master-key rankings | Include methodology page')
        canvas.drawRightString(760, 17, f'Page {document.page}')
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path(__file__).resolve().parent / 'inputs')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'rankings')
    p.add_argument('--exact-dataset', type=Path, help='Optional local t15 dataset for original joint-score rescoring')
    args = p.parse_args(argv)
    report = build_report(args.inputs, args.exact_dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'master_key_rankings.json').write_text(json.dumps(report, indent=2))
    write_pdf(report, args.output / 'Week15_Master_Key_Rankings.pdf')
    for method, info in report['methods'].items():
        print(method, 'entries', info['candidate_count'], 'correct-byte ranks', info['correct_value_ranks'])
    print('Scoring mode:', report['score_mode'])
    print('Output:', args.output)


if __name__ == '__main__':
    main()
