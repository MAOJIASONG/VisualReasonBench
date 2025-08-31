# HANDOVER

## Supplementary Documentation

**Environment Harness**: `docs/ENVIRONMENT_HARNESS.md`
**Blender**: `docs/BLENDER.md`

## TODO

- [ ] Add color (+ number) combination to current environments using Blender (**IMPORTANT**)
- [ ] Finish up `luban_env.py` on the constrained implementation (should be roughly finished, but not properly tested yet)
  - [ ] Create solutions for luban-style puzzles (luban-*-piece, liliput, tangled-nails, three-plate-burr-puzzle, interlocking cube)
  - [ ] Implement the scoring mechanism for `luban_env.py` where the optimal number of steps is the minimum steps taken
  - [ ] Count number of steps needed for each puzzle, easier puzzles can simply be done manually, harder ones usually have solutions online (the puzzles selected here have solutions) which can be found under the [Google Slides](https://docs.google.com/presentation/d/13wSxfrFKtroNnvdFSS0EKtA56oYau_soGPTUynzVstI/edit?usp=sharing)
- [ ] Continue implementing "PLACE" mechanism (primarily for lego use-cases)
