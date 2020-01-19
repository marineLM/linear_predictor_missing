from estimators import ConstantImputedMLPR


def add_MLP_method(type_width, width, depth, pretrain, methods, est_params):
    key = 'MLP W{}, D{}'.format(width, depth)
    methods[key] = ConstantImputedMLPR
    est_params[key] = {'type_width': type_width, 'width': width,
                       'depth': depth,
                       'solver': 'adam', 'activation': 'relu',
                       'learning_rate': 'adaptive',
                       'max_iter': 1000, 'tol': 1e-4,
                       'warm_start': True,
                       'mask': True, 'imputation': 0}


def choose_filename(file_root, n_sizes, p_sizes, data_type, n_iter):
    if len(n_sizes) == 1 and len(p_sizes) == 1:
        filename = "{}_{}_n{}_dim{}_{}iter".format(
            file_root, data_type, n_sizes[0], p_sizes[0], n_iter)
    elif len(n_sizes) == 1:
        filename = "{}_{}_n{}_{}iter".format(
            file_root, data_type, n_sizes[0], n_iter)
    elif len(p_sizes) == 1:
        filename = "{}_{}_dim{}_{}iter".format(
            file_root, data_type, p_sizes[0], n_iter)
    else:
        filename = "{}_{}_{}iter".format(file_root, data_type, n_iter)
    return filename
