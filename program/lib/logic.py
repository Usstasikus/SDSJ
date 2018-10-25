import os
import time
import unittest.mock

import hp_util
import portfolio as portfolio_module
from hpbandster.core.utils import start_local_nameserver
from autosklearn.ensemble_builder import EnsembleBuilder
import autosklearn.util.backend
from autosklearn.data import competition_data_manager
from autosklearn.metrics import roc_auc
from autosklearn.constants import BINARY_CLASSIFICATION
import numpy as np
import pynisher
import sklearn.decomposition
import sklearn.feature_selection
import sklearn.pipeline
import sklearn.preprocessing


def project_data_via_feature_selection(X_train, X_test, y_train, logger,
                                       subset=None):
    n_features = X_train.shape[1]
    n_keep = 500
    if n_features > n_keep:
        logger.info(
            'Transforming the data from %d to %d features!',
            n_features,
            n_keep
        )
        imp = sklearn.preprocessing.Imputer(strategy='median')
        pca = sklearn.feature_selection.SelectKBest(k=n_keep)
        pipeline = sklearn.pipeline.Pipeline((('imp', imp), ('pca', pca)))
        if subset is not None and X_train.shape[0] > subset:

            logger.info(
                'Selecting subset of features on only %d data points!',
                subset
            )

            subset_indices = list(
                np.random.choice(list(range(X_train.shape[0])), size=subset)
            )
            subset_indices.sort()
            X_train_ = X_train[subset_indices]
            y_train_ = y_train[subset_indices]
        else:
            X_train_ = X_train
            y_train_ = y_train

        pipeline.fit(X_train_, y_train_)
        X_train = pipeline.transform(X_train)
        X_test = pipeline.transform(X_test)
        mask = pca._get_support_mask()
        assert np.sum(mask) == n_keep
    else:
        mask = np.ones(X_train.shape[1], dtype=bool)
    return X_train, X_test, mask


def run_automl(args, logger, input_dir, output_dir, tmp_output_dir, dataset_name,
               budget, seed=3, sleep=5):
    start_task = float(time.time())

    logger.info("Using %s as tmp outputdir and %s as outdir" %
                (tmp_output_dir, output_dir))

    # Compute time left for this task with 5 sec slack
    D = competition_data_manager.CompetitionDataManager(
        name=os.path.join(input_dir, dataset_name), args=args,
        max_memory_in_mb=1048576)
    n_data_points = D.data['X_train'].shape[0]

    to_pynish = pynisher.enforce_limits(mem_in_mb=6000, wall_time_in_s=60)(project_data_via_feature_selection)
    rval = to_pynish(
        D.data['X_train'], D.data['X_test'], D.data['Y_train'], logger,
    )
    print(f'___________FEAT TYPE___________: {D.feat_type}')
    print(f'___________INFO___________: {D.info}')
    print(f'___________RVAL___________: {len(rval[2])}')
    if rval is None:
        logger.warning(
            'Error projecting data via full feature selection: %s',
            to_pynish.exit_status
        )

        to_pynish = pynisher.enforce_limits(mem_in_mb=6000, wall_time_in_s=60)(
            project_data_via_feature_selection
        )
        rval = to_pynish(
            D.data['X_train'], D.data['X_test'], D.data['Y_train'], logger, 1000
        )
        if rval is None:
            logger.warning(
                'Error projecting data via Feature Selection on data subset: %s',
                to_pynish.exit_status
            )
            logger.warning('Taking a random subset now!')
            indices = list(sorted(np.random.choice(
                list(range(D.data['X_train'].shape[1])), size=500
            )))
            D.data['X_train'] = D.data['X_train'][:, indices]
            D.data['X_test'] = D.data['X_test'][:, indices]
            D.feat_type = [
                ft for i, ft in enumerate(D.feat_type) if i in indices
                ]
        else:
            D.data['X_train'] = rval[0]
            D.data['X_test'] = rval[1]
            D.feat_type = [
                ft for i, ft in enumerate(D.feat_type) if rval[2][i] == True
            ]

    else:
        D.data['X_train'] = rval[0]
        D.data['X_test'] = rval[1]
        D.feat_type = [
            ft for i, ft in enumerate(D.feat_type) if rval[2][i] == True
        ]
    logger.info(
        'Dataset dimensions: %s %s', D.data['X_train'].shape, D.data['Y_train'].shape,
    )

    # We take the min in case there is less time left
    time_budget = min(budget, float(D.info['time_budget']))

    backend = autosklearn.util.backend.create(
                temporary_directory=tmp_output_dir,
                output_directory=output_dir,
                delete_tmp_folder_after_terminate=False,
                delete_output_folder_after_terminate=False)

    backend.save_datamanager(datamanager=D)
    shuffle = not bool(D.info.get("is_chronological_order", False))
    del D

    hp_util._do_dummy_prediction(
        backend=backend,
        seed=seed,
        num_run=1,
        logger=logger,
        shuffle=shuffle,
        time_for_task=time_budget,
        n_data_points=n_data_points,
    )

    # Start Ensemble script with time_left
    time_left_for_this_task = time_budget - (time.time() - start_task)
    ensemble_builder = EnsembleBuilder(backend=backend,
                                       dataset_name=dataset_name,
                                       task_type=BINARY_CLASSIFICATION,
                                       metric=roc_auc,
                                       limit=time_left_for_this_task,
                                       ensemble_size=50,
                                       ensemble_nbest=50,
                                       seed=seed,
                                       shared_mode=False,
                                       max_iterations=None,
                                       precision="32",
                                       sleep_duration=sleep)
    ensemble_builder.start()

    # max_iter based
    #max_budget = 512
    #min_budget = 32
    #eta = 4

    # subset based
    max_budget = 1.0
    min_budget = 1.0 / 16
    eta = 4

    if n_data_points < 1000:
        min_budget = max_budget
        total_budget = 16
    else:
        # calculate total budget
        total_budget = 0
        n_algos = max_budget / min_budget
        budget = min_budget
        while n_algos > 1:
            total_budget += budget * n_algos
            budget = budget * eta
            n_algos = n_algos / eta
        if total_budget == 0:
            total_budget = max_budget

    ns_host, ns_port = start_local_nameserver()
    # (Note) ID serves as worker.id and seed for TargetAlgorithmEvaluator
    # If we use more than one worker this number needs to be unique
    run_id = '0'
    worker = hp_util.AutoMLWorker(
        dataset_name=dataset_name,
        n_data_points=n_data_points,
        backend=backend,
        total_budget=total_budget,
        total_time=time_left_for_this_task,
        run_id=run_id,
        nameserver=ns_host,
        nameserver_port=ns_port,
        mode=None, # set to 'subsets' or 'iterations' to overwrite algorithm specific treatment
        id=seed,
        shuffle=shuffle,
    )
    worker.run(background=True)

    autosklearn_portfolio = portfolio_module.get_hydra_portfolio(dataset_name)
    for entry in autosklearn_portfolio:
        for constant in worker.constant_values:
            if constant in entry:
                del entry[constant]
    logger.info(
        'Retrieved portfolio of length %d: %s',
        len(autosklearn_portfolio),
        autosklearn_portfolio,
    )

    SSB = hp_util.SideShowBOHB(
        configspace=worker.get_config_space(),
        initial_configs=autosklearn_portfolio,
        run_id=run_id,
        eta=eta, min_budget=min_budget, max_budget=max_budget,
        SH_only=True,       # suppresses Hyperband's outer loop and runs SuccessiveHalving only
        nameserver=ns_host,
        nameserver_port=ns_port,
        ping_interval=sleep,
        job_queue_sizes=(-1, 0),
        dynamic_queue_size=True,
    )

    if min_budget == max_budget:
        res = SSB.run(len(autosklearn_portfolio), min_n_workers=1)
    else:
        res = SSB.run(1, min_n_workers=1)

    runs = res.get_all_runs()
    all_losses = np.array([r.loss for r in runs], dtype=np.float)
    times = [r.time_stamps['finished'] - r.time_stamps['started']
             for r in runs]
    time_taken = np.nansum(times)
    if time_taken is not None and np.isfinite(time_taken) and min_budget != max_budget:
        time_left_for_this_worker = time_left_for_this_task - time_taken
    else:
        time_left_for_this_worker = time_left_for_this_task

    if not np.any(np.isfinite(all_losses)):
        logger.error('Found no succesful runs so far, will continue with a '
                     'fallback configuration space!')
        SSB.shutdown(shutdown_workers=True)

        portfolio_ = [{
                            'balancing:strategy': 'weighting',
                            'categorical_encoding:__choice__': 'no_encoding',
                            'classifier:__choice__': 'extra_trees',
                            'classifier:extra_trees:bootstrap': 'True',
                            'classifier:extra_trees:criterion': 'entropy',
                            'classifier:extra_trees:max_features': 0.5,
                            'classifier:extra_trees:min_samples_leaf': 5,
                            'classifier:extra_trees:min_samples_split': 10,
                            'imputation:strategy': 'mean',
                            'preprocessor:__choice__': 'no_preprocessing',
                            'rescaling:__choice__': 'none'
                }]
        include = { 'classifier': [ 'extra_trees' ],
                    'preprocessor': [ 'no_preprocessing'],
                  }

        bohb_worker = hp_util.AutoMLWorker(
            dataset_name=dataset_name,
            n_data_points=n_data_points,
            backend=backend,
            total_budget=total_budget,
            total_time=time_left_for_this_worker,
            run_id=run_id,
            include=include,
            # gives access to the whole autosklearn configspace
            nameserver=ns_host,
            nameserver_port=ns_port,
            mode=None,
            # set to 'subsets' or 'iterations' to overwrite algorithm specific treatment
            id=seed,
            shuffle=shuffle,
            counter=worker.counter + 1,
            use_backup_budgets=True,
        )
        bohb_worker.run(background=True)

        SSB = hp_util.SideShowBOHB(
            configspace=bohb_worker.get_config_space(),
            initial_configs=portfolio_,
            run_id=run_id,
            eta=eta, min_budget=min_budget, max_budget=max_budget,
            SH_only=False,
            # suppresses Hyperband's outer loop and runs SuccessiveHalving only
            random_fraction=0.1,
            nameserver=ns_host,
            nameserver_port=ns_port,
            ping_interval=sleep,
            job_queue_sizes=(-1, 0),
            dynamic_queue_size=True,
        )

    else:
        worker = hp_util.AutoMLWorker(
            dataset_name=dataset_name,
            n_data_points=n_data_points,
            backend=backend,
            total_budget=total_budget,
            total_time=time_left_for_this_worker,
            run_id=run_id,
            nameserver=ns_host,
            nameserver_port=ns_port,
            mode=None,
            # set to 'subsets' or 'iterations' to overwrite algorithm
            # specific treatment
            id=seed,
            shuffle=shuffle,
            counter=worker.counter + 1,
        )
        worker.run(background=True)

    res = SSB.run(1000, min_n_workers=1)

    ensemble_builder.join(10)
    if ensemble_builder.is_alive():
        # Note: terminate only asks the process to terminate - we need to add
        # a mechanism to kill it!
        # [Katha]: Don't think this is necessary here, as we kill process anyway
        # from the outside
        ensemble_builder.terminate()
