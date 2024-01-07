def simple_binarization(dataset: dict[str, list]) -> list[tuple[list,list,str]]:
    labels = list(dataset.keys()).copy()

    binarized_dataset: list[tuple[list,list,str]] = []

    checked_labels = []
    for label in labels[:-1]:
        checked_labels.append(label)
        X1 = dataset[label].copy()
        X2 = []

        i = 0
        for label_ in labels:
            if label_ in checked_labels:
                continue
            X2.extend(dataset[label_])
            i += 1

        X = X1.copy()
        X.extend(X2)

        y = [label] * len(X1)

        if i == 1:
            y.extend(
                [label_] * len(X2)
            )
        else:

            y.extend(
                ["."] * len(X2)
            )

        binarized_dataset.append(
            (X, y, label)
        )

    return binarized_dataset


def our_binarization(dataset: dict[str, list]):
    """
    TODO
    Our idea for binarization for dataset
    """