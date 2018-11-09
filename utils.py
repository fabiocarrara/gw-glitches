from io import StringIO


def cm2str(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    Pretty print for confusion matrices from https://gist.github.com/zachguo/10296432

    :param cm: sklearn confusion_matrix
    :param labels: list of string with class labels
    :param hide_zeroes:
    :param hide_diagonal:
    :param hide_threshold:
    """

    cm_str = StringIO()

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ", file=cm_str)
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ", file=cm_str)

    print(file=cm_str)
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ", file=cm_str)
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ", file=cm_str)
        print(file=cm_str)

    return cm_str.getvalue()
