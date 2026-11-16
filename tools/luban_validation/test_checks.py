import copy
import unittest
from checks import certify_scene_separation, exact_terminal_match


class TerminalChecks(unittest.TestCase):
    def test_interlocked_group_cannot_be_certified(self):
        bounds = [{'id': i, 'min': dict(x=300+i*.1, y=0, z=0),
                   'max': dict(x=301+i*.1, y=1, z=1)} for i in range(2)]
        with self.assertRaises(AssertionError):
            certify_scene_separation(bounds, [0, 1])
        bounds[1]['min']['x'] += 10
        bounds[1]['max']['x'] += 10
        self.assertTrue(certify_scene_separation(bounds, [0, 1])['all_pairs_separated'])

    def test_invalid_bounds_cannot_certify(self):
        a = {'id': 0, 'min': dict(x=0, y=0, z=0), 'max': dict(x=1, y=1, z=1)}
        for bad in [[a, a], [a]]:
            with self.assertRaises(AssertionError):
                certify_scene_separation(bad, [0, 1])
        a['min']['x'] = float('nan')
        with self.assertRaises(AssertionError):
            certify_scene_separation([a], [0])

    def test_terminal_requires_exact_coordinates_and_unique_ids(self):
        a = {'Id': 0, 'X': '-400000/1', 'Y': '0/1', 'Z': '1/2',
             'Orientation': [0, 2, 4], 'HalfRotateAxis': 0, 'Finished': True}
        b = copy.deepcopy(a)
        b['Z'] = '2/4'
        exact_terminal_match([a], [b])
        b['X'] = '-399999999/1000'
        with self.assertRaises(AssertionError):
            exact_terminal_match([a], [b])
        with self.assertRaises(AssertionError):
            exact_terminal_match([a, a], [a, a])


if __name__ == '__main__':
    unittest.main()
