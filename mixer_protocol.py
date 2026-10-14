import os
import numpy as np
from quapy.protocol import AbstractProtocol
from quapy.data._lequa import SamplesFromDir
import quapy.functional as F


class MixerLequaSamples(AbstractProtocol):

    def __init__(self, lequa_bag_generator: SamplesFromDir, repeats:int=1000, cached=True, random_state=None):
        self.lequa_bag_generator = lequa_bag_generator
        self.repeats = repeats
        self.rng = np.random.default_rng(random_state)
        self.cached = cached
        self._cache = {}

    def __call__(self):
        idx_trueprevs = list(self.lequa_bag_generator.true_prevs.iterrows())
        iterations = self.repeats
        while iterations>0:
            p1, p2 = self.rng.choice(self.total(), size=2, replace=False)

            idx1, prev1 = idx_trueprevs[p1]
            idx2, prev2 = idx_trueprevs[p2]

            bag1 = self.get_sample(idx1)
            bag2 = self.get_sample(idx2)

            yield self.mix(bag1, bag2, prev1, prev2)

            iterations -= 1
        return

    def total(self):
        return len(self.lequa_bag_generator.true_prevs)

    def _load_bag(self, idx):
        path = os.path.join(self.lequa_bag_generator.path_dir, f'{idx}.txt')
        sample, _ = self.lequa_bag_generator.load_fn(path)
        return sample

    def get_sample(self, idx):
        sample = self._cache[idx] if idx in self._cache else self._load_bag(idx)
        if self.cached:
            self._cache[idx] = sample
        return sample

    def mix(self, bag1, bag2, prev1, prev2, prop1=0.5, prop2=0.5):
        size1=int(len(bag1)*prop1)
        mix1 = self.rng.choice(len(bag1), size=size1, replace=False)

        size2 = int(len(bag2) * prop2)
        mix2 = self.rng.choice(len(bag2), size=size2, replace=False)

        mixed_bag = np.vstack([bag1[mix1], bag2[mix2]])
        mixed_prev = F.l1_norm(np.asarray(prev1)*prop1 + np.asarray(prev2)*prop2)

        return mixed_bag, mixed_prev


if __name__ == '__main__':
    from quapy.data.datasets import fetch_lequa2022, fetch_lequa2024
    train, val_gen, test_gen = fetch_lequa2024(task='T2')
    mixer = MixerLequaSamples(val_gen, repeats=2, random_state=0)
    # will generate only two mixes, but across runs the two mixes are the replicated
    for i, (bag, prev) in enumerate(mixer()):
        print(f'{i=}')
        print(bag)
        print(prev)
    # the mixes across consecutive calls are different
    for i, (bag, prev) in enumerate(mixer()):
        print(f'{i=}')
        print(bag)
        print(prev)
