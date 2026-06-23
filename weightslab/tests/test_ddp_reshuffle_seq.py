"""Unit tests for DDP shard reshuffle semantics + reproducibility (no DDP spawn).

The per-rank shard permutation is a pure function of (ddp_seed, reshuffle_seq,
rank, world). Two properties matter:

  1. Re-generating indices WITHOUT advancing the reshuffle generation yields the
     SAME permutation — so a mid-loop discard (which resets the iterator) merely
     re-filters the same shard instead of reshuffling it. advance_reshuffle()
     (called only on a genuine pass-end) DOES change the permutation.
  2. restore_reshuffle_seq reproduces the exact permutation a checkpoint saw, so
     a DDP restore resumes the same per-rank order (the deny-list that filters it
     is checkpointed separately).

DistributedSampler is built with explicit num_replicas/rank, so this needs no
live process group — we just force world>1 on the sampler.
"""
import unittest

from weightslab.backend.dataloader_interface import WeightsLabDataSampler


def _sampler(n=24, rank=0, world=2, seed=0):
    s = WeightsLabDataSampler(list(range(n)), shuffle=True)
    s._ddp_rank, s._ddp_world_size = rank, world   # force the DDP branch
    s._ddp_seed = seed
    s._reshuffle_seq = 0
    s._dist_sampler = None
    return s


class ReshuffleSeqTests(unittest.TestCase):
    def test_regen_without_advance_is_stable(self):
        s = _sampler()
        p_a = s._generate_indices()
        p_b = s._generate_indices()          # discard-style re-gen, no advance
        self.assertEqual(p_a, p_b, "re-gen without advance must not reshuffle")

    def test_advance_changes_permutation(self):
        s = _sampler()
        p0 = s._generate_indices()
        s.advance_reshuffle()                # genuine pass end
        p1 = s._generate_indices()
        self.assertNotEqual(p0, p1, "advance_reshuffle must reshuffle the shard")

    def test_restore_reproduces_permutation(self):
        live = _sampler()
        live._generate_indices()             # gen 0
        live.advance_reshuffle()             # → gen 1
        p_at_save = live._generate_indices()
        saved_seq = live._reshuffle_seq      # what capture_iteration_state stores

        fresh = _sampler()                   # a restarted process: seq back to 0
        fresh.restore_reshuffle_seq(saved_seq, seed=0)
        self.assertEqual(fresh._generate_indices(), p_at_save,
                         "restore must reproduce the saved per-rank permutation")

    def test_ranks_partition_disjoint_cover(self):
        # The two ranks' shards for the same generation must be disjoint and cover
        # the universe — sanity that sharding still works after the refactor.
        r0 = set(_sampler(rank=0, world=2)._generate_indices())
        r1 = set(_sampler(rank=1, world=2)._generate_indices())
        self.assertEqual(r0 & r1, set(), "shards must be disjoint")
        self.assertEqual(r0 | r1, set(range(24)), "shards must cover the universe")

    def test_world3_uneven_shards_pad_and_cover(self):
        # n=25, world=3 → DistributedSampler(drop_last=False) pads total to 27, so
        # each shard is 9 long and the union still covers the whole universe
        # (padding repeats a couple of real indices). Exercises uneven/padded shards.
        shards = [set(_sampler(n=25, rank=r, world=3)._generate_indices())
                  for r in range(3)]
        lens = [len(_sampler(n=25, rank=r, world=3)._generate_indices())
                for r in range(3)]
        self.assertEqual(lens, [9, 9, 9], "padded shards must be equal length")
        self.assertEqual(set().union(*shards), set(range(25)),
                         "padded shards must still cover the whole universe")

    def test_seed_mismatch_warns_but_sets(self):
        s = _sampler(seed=0)
        # Different seed → warn, but still set the seq (best-effort).
        s.restore_reshuffle_seq(3, seed=999)
        self.assertEqual(s._reshuffle_seq, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
