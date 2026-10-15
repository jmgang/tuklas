"""Scoped correction of TokenLearn 0.2.1's live best-checkpoint references.

Keep its official training objective, optimizer and export path. Never modify
the installed package or historical artifacts. The replacement is scoped to
one training call and restored even on failure.
"""
from contextlib import contextmanager


def checkpoint_safe_class(original):
    class SnapshotFineTuner(original):
        def state_dict(self, *args, **kwargs):
            state = super().state_dict(*args, **kwargs)
            # Module.state_dict is shallow: TokenLearn retains this object as
            # its best checkpoint. Copies are necessary to retain past weights.
            for key in state:
                state[key] = state[key].detach().clone()
            return state
    return SnapshotFineTuner


@contextmanager
def independent_tokenlearn_checkpoints():
    import tokenlearn.pretrain as pretrain
    original = pretrain.StaticModelFineTuner
    pretrain.StaticModelFineTuner = checkpoint_safe_class(original)
    try:
        yield
    finally:
        pretrain.StaticModelFineTuner = original
