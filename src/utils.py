import random


def stratified_train_test_indices(targets, train_ratio, seed):
    rng = random.Random(seed)
    idx_by_class = {}
    for idx, y in enumerate(targets):
        idx_by_class.setdefault(y, []).append(idx)

    train_indices = []
    test_indices = []

    for y, idxs in idx_by_class.items():
        rng.shuffle(idxs)
        n_train = int(train_ratio * len(idxs))
        train_indices.extend(idxs[:n_train])
        test_indices.extend(idxs[n_train:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def make_train_val_split_from_train_indices(train_indices, targets, val_ratio, seed):
    rng = random.Random(seed)
    idx_by_class = {}
    for idx in train_indices:
        y = targets[idx]
        idx_by_class.setdefault(y, []).append(idx)

    train_idx2 = []
    val_idx = []

    for y, idxs in idx_by_class.items():
        rng.shuffle(idxs)
        n_val = int(val_ratio * len(idxs))
        val_idx.extend(idxs[:n_val])
        train_idx2.extend(idxs[n_val:])

    rng.shuffle(train_idx2)
    rng.shuffle(val_idx)
    return train_idx2, val_idx
