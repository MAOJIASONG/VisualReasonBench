"""Independent receipt and terminal-state checks for real Unity validation."""
from fractions import Fraction
import hashlib
import json


def verify_runtime_manifest(package):
    manifest = json.loads((package/'repair_manifest.json').read_text())
    for filename, key in [('Assembly-CSharp.dll', 'output_sha256'),
                          ('OriginalEngine.dll', 'original_engine_sha256'),
                          ('LubanOriginalRuntime.dll', 'extension_sha256')]:
        actual = hashlib.sha256((package/'LuBanSuoEnv_Data/Managed'/filename).read_bytes()).hexdigest()
        if actual.lower() != manifest[key].lower():
            raise ValueError('Runtime differs from its manifest: '+filename)
    for name, expected in manifest['rule_file_sha256'].items():
        actual = hashlib.sha256((package/'LuBanSuoEnv_Data/StreamingAssets/OriginalRules'/name).read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            raise ValueError('Runtime rule file differs from manifest: '+name)
    if len(manifest['rule_file_sha256']) != 32:
        raise ValueError('Runtime manifest must cover all 32 level files')
    return manifest


def exact_terminal_match(expected, actual):
    def indexed(pieces):
        by_id = {p['Id']: p for p in pieces}
        if len(by_id) != len(pieces):
            raise AssertionError('Duplicate piece IDs in state')
        return by_id
    expected, actual = indexed(expected), indexed(actual)
    if expected.keys() != actual.keys():
        raise AssertionError('Piece ID sets differ')
    for piece_id, a in expected.items():
        b = actual[piece_id]
        for coordinate in 'XYZ':
            if Fraction(a[coordinate]) != Fraction(b[coordinate]):
                raise AssertionError(f'Piece {piece_id} coordinate {coordinate} differs')
        for key in ['Orientation', 'HalfRotateAxis', 'Finished']:
            if a[key] != b[key]:
                raise AssertionError(f'Piece {piece_id} {key} differs')


def certify_scene_separation(bounds, expected_ids, clearance=1e-5):
    """Require positive separation between every pair of actual scene AABBs."""
    indexed = {b['id']: b for b in bounds}
    if len(indexed) != len(bounds) or set(indexed) != set(expected_ids):
        raise AssertionError('Scene bounds have missing or duplicate piece IDs')
    pairs = []
    import math
    for b in bounds:
        for axis in 'xyz':
            if not all(math.isfinite(b[key][axis]) for key in ['min', 'max']) or b['min'][axis] >= b['max'][axis]:
                raise AssertionError('Invalid actual scene bounds')
    for index, a in enumerate(bounds):
        for b in bounds[index+1:]:
            gaps = [max(a['min'][axis]-b['max'][axis], b['min'][axis]-a['max'][axis]) for axis in 'xyz']
            if max(gaps) <= clearance:
                raise AssertionError(f'Actual scene pieces {a["id"]} and {b["id"]} lack a positive AABB gap')
            pairs.append({'pieces': [a['id'], b['id']], 'largest_axis_gap_world': max(gaps)})
    return {'all_pairs_separated': True, 'clearance_world': clearance, 'pairs': pairs}
