import copy
import logging
import json
import multiprocessing
import unittest.mock

import autosklearn
import autosklearn.pipeline.classification
from autosklearn.evaluation import ExecuteTaFuncWithQueue
from autosklearn.metrics import roc_auc
from ConfigSpace import Configuration, ConfigurationSpace
import ConfigSpace
import ConfigSpace.util
from hpbandster.core.worker import Worker
from hpbandster.core.master import Master
from hpbandster.iterations.base import BaseIteration
from hpbandster.config_generators.bohb import BOHB
import numpy as np
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import StatusType


N_KEEP_DATA = 30000
TA_MEMORY_LIMIT = 6000
N_FOLDS = 10
MIN_N_DATA_FOR_SH = 1000


class Dummy(object):
    def __init__(self):
        self.name = 'Dummy'


class AutoMLWorker(Worker):
    def __init__(self, dataset_name, n_data_points, backend, total_budget,
                 total_time, shuffle=True,
                 mode = None, # set to 'subsamples' or 'iterations' to overwrite algorithm specific treatment
                 include = {    'classifier': [ 'xgradient_boosting', 'sgd', 'random_forest', 'libsvm_svc' ],
                                'preprocessor': [ 'no_preprocessing'],
                            },
                 counter=2, use_backup_budgets=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(
            '%s(%s)' % (__class__.__name__, dataset_name)
        )

        self.n_data_points = n_data_points
        self.total_budget = total_budget
        self.total_time = total_time
        self.shuffle = shuffle

        self.use_backup_budgets = use_backup_budgets
        if use_backup_budgets:
            self.budget_converter = {
                'libsvm_svc': lambda b: b,
                'random_forest': lambda b: int(b * 32),
                'sgd': lambda b: int(b * 32),
                'xgradient_boosting': lambda b: int(b * 32),
                'extra_trees': lambda b: int(b * 32),
            }
        else:
            self.budget_converter = {
                        'libsvm_svc': lambda b: b,
                     'random_forest': lambda b: int(b*128),
                               'sgd': lambda b: int(b*512),
                'xgradient_boosting': lambda b: int(b*512),
                       'extra_trees': lambda b: int(b*1024)
            }

        if not mode is None:
            if not mode in ['subsets', 'iterations']:
                raise ValueError("mode argument has to be either 'subsets' or 'iterations', but got %s"%mode)

        self.modes = {
                    'libsvm_svc': 'subsets'    if mode is None else mode,
                 'random_forest': 'iterations' if mode is None else mode,
                           'sgd': 'iterations' if mode is None else mode,
            'xgradient_boosting': 'iterations' if mode is None else mode,
                   'extra_trees': 'iterations' if mode is None else mode
        }

        # setup autosklearn compoments here:
        #       - backend
        #       - datamanager
        #       - anything else?
        self.include = include

        self.pipeline = autosklearn.pipeline.classification.SimpleClassificationPipeline(
            include=self.include,
        )
        self.config_space = self.pipeline.get_hyperparameter_search_space()
        self.logger.info(
            'Working on %d-dimensional configuration space.',
            len(self.config_space.get_hyperparameters()),
        )
        
        self.reduced_config_space = ConfigurationSpace()
        
        self.constant_values = {}
        
        for HP in self.config_space.get_hyperparameters():
            
            if hasattr(HP, 'choices') and len(HP.choices) == 1:
                self.constant_values[HP.name] = HP.choices[0]
                print('found constant %s with value %s'%(HP.name, HP.choices[0]))
            
            elif type(HP) == ConfigSpace.hyperparameters.Constant:
                self.constant_values[HP.name] = HP.default_value
                print('found constant %s with value %s'%(HP.name, HP.default_value))
            else:
                self.reduced_config_space.add_hyperparameter(copy.copy(HP))

        for condition in self.config_space.get_conditions():
            try:
                self.reduced_config_space.add_condition(
                    condition.__class__(
                        self.reduced_config_space.get_hyperparameter(condition.child.name),
                        self.reduced_config_space.get_hyperparameter(condition.parent.name),
                        condition.value
                    )
                )
            except KeyError:
                print('Not copying condition', condition)
                pass
            except:
                #import pdb; pdb.set_trace()
                pass


        self.logger.info(
            'Reduced it to a %d-dimensional configuration space.',
            len(self.reduced_config_space.get_hyperparameters()),
        )
            
        # Counter of 1 is the dummy prediction!
        self.counter = counter

        if "id" in kwargs:
            self.id = int(float(kwargs["id"]))

        self.backend = backend
        self.queue = multiprocessing.Queue()

    def compute(self, config=None, budget=1, working_directory='/tmp'):
        if config is None:
            config = self.config_space.sample_configuration()
        else:
            if 'rescaling:quantile_transformer:n_quantiles' in config and \
                    config['rescaling:quantile_transformer:n_quantiles'] > 2000:
                config['rescaling:quantile_transformer:n_quantiles'] = 2000

            # add the constants back in
            config.update(self.constant_values)
            # and deactivate the inactive parameters
            config = ConfigSpace.util.deactivate_inactive_hyperparameters(
                                configuration_space=self.config_space,
                                configuration=config)

        classifier = config['classifier:__choice__']

        instance = {}
        if self.n_data_points > int(N_KEEP_DATA * 1.5):
            self.logger.info(
                'Changing train/test split. Using only %f (%d data points) of '
                'the data.',
                N_KEEP_DATA/ self.n_data_points,
                N_KEEP_DATA,
            )
            instance['subsample'] = N_KEEP_DATA
            n_data_points = N_KEEP_DATA
        else:
            n_data_points = self.n_data_points

        kwargs = {}
        kwargs["shuffle"] = self.shuffle

        if n_data_points < MIN_N_DATA_FOR_SH:
            resampling_strategy = 'cv'
            kwargs['folds'] = N_FOLDS
        else:
            resampling_strategy = 'holdout'
        self.logger.info('Using resampling strategy %s.', resampling_strategy)

        percentage_budget = budget / self.total_budget
        cutoff = self.total_time * percentage_budget
        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = self.total_time
        scenario_mock.algo_runs_timelimit = self.total_time
        scenario_mock.ta_run_limit = np.inf
        stats = Stats(scenario_mock)
        stats.ta_runs = 2
        stats.start_timing()
        tae = ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=self.id,
            # As we have max_iter it runs with holdout and iterative fit!!!
            resampling_strategy=resampling_strategy,
            metric=roc_auc,
            logger=self.logger,
            initial_num_run=self.counter,
            stats=stats,
            runhistory=None,
            run_obj='quality',
            par_factor=1,
            all_scoring_functions=False,
            output_y_hat_optimization=True,
            include=self.include,
            exclude=None,
            memory_limit=TA_MEMORY_LIMIT,
            disable_file_output=False,
            init_params=None,
            **kwargs
        )

        mode = self.modes[classifier]
        budget = self.budget_converter[classifier](budget)

        if resampling_strategy == 'cv':
            pass
        elif mode == 'iterations':
            instance['max_iter'] = budget
        elif mode == 'subsets':
            budget = int(budget * n_data_points)
            instance['subsample'] = budget
        else:
            raise ValueError(mode)
        instance = json.dumps(instance)

        status, cost, runtime, additional_run_info = tae.start(
            config=config,
            instance=instance,
            cutoff=int(np.ceil(cutoff)),
            instance_specific=None,
            capped=False,
        )
        self.counter += 1

        if status != StatusType.SUCCESS:
            cost = float('inf')
        # Never advance a support vector machine if we do iterations,
        # because it's always trained on the full amount of tolerance!
        elif config['classifier:__choice__'] == 'libsvm_svc' and \
            mode=='iterations':
                cost = float('inf')


        return ({
            'loss': cost,
            'info': {
                'status': status,
                'runtime': runtime,
                'additional_run_info': additional_run_info,
                'config': config.get_dictionary()
            }
        })

    def get_config_space(self):
        return (self.reduced_config_space)


class PortfolioBOHB(BOHB):
    """ subclasses the config_generator BOHB"""
    def __init__(self, initial_configs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if initial_configs is None:
            # dummy initial portfolio
            self.initial_configs = [self.configspace.sample_configuration().get_dictionary() for i in range(5)]
        else:
            self.initial_configs = initial_configs

    def get_config(self, budget):

        # return a portfolio member first
        if len(self.initial_configs) > 0 and True:
            c = self.initial_configs.pop()
            return (c, {'portfolio_member': True})

        return (super().get_config(budget))

    def new_result(self, job):
        # notify ensemble script or something
        super().new_result(job)

class SuccessivePanicking(BaseIteration):

    def _advance_to_next_stage(self, config_ids, losses):
        """
            SuccessiveHalving simply continues the best based on the current loss.
        """
        
        if len(config_ids)==0:
            for i in range(self.stage, len(self.num_configs)):
                self.num_configs[i] = 0     
        
        ranks = np.argsort(np.argsort(losses))
        return(ranks < self.num_configs[self.stage])





class SideShowBOHB(Master):
    def __init__(self, initial_configs=None, configspace=None,
                 eta=3, min_budget=0.01, max_budget=1,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=64, random_fraction=0.5, bandwidth_factor=3,
                 SH_only=False,
                 *args, **kwargs):
        # MF I changed the parameters a bit to be more aggressive after the
        # portfolio evaluation, but also to still do some random search.

        cg = PortfolioBOHB(
                     initial_configs=initial_configs,
                     configspace=configspace,
                     min_points_in_model=min_points_in_model,
                     top_n_percent=top_n_percent,
                     num_samples=num_samples,
                     random_fraction=random_fraction,
                     bandwidth_factor=bandwidth_factor,
                     )

        super().__init__(config_generator=cg, *args, **kwargs)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        self.SH_only = SH_only

        # precompute some HB stuff
        self.max_SH_iter = -int(
            np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta,
                                             -np.linspace(self.max_SH_iter - 1,
                                                          0, self.max_SH_iter))
        self.logger.info('Using budgets %s.', self.budgets)

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'min_points_in_model': min_points_in_model,
            'top_n_percent': top_n_percent,
            'num_samples': num_samples,
            'random_fraction': random_fraction,
            'bandwidth_factor': bandwidth_factor
        })

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        """
            BO-HB uses (just like Hyperband) SuccessiveHalving for each
            iteration. See Li et al. (2016) for reference.

            Parameters:
            -----------
                iteration: int
                    the index of the iteration to be instantiated
            Returns:
            --------
                SuccessiveHalving: the SuccessiveHalving iteration with the
                    corresponding number of configurations
        """

        # number of 'SH rungs'
        if self.SH_only:
            s = self.max_SH_iter - 1
        else:
            s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        return SuccessivePanicking(HPB_iter=iteration,
                                 num_configs=ns,
                                 budgets=self.budgets[(-s - 1):],
                                 config_sampler=self.config_generator.get_config,
                                 **iteration_kwargs)


def _do_dummy_prediction(backend, seed, num_run, logger, shuffle, time_for_task,
                         n_data_points):
    if n_data_points < MIN_N_DATA_FOR_SH:
        resampling_strategy = 'cv'
        kwargs = {'folds': N_FOLDS}
    else:
        resampling_strategy = 'holdout'
        kwargs = {}
    logger.info('Using resampling strategy %s.', resampling_strategy)

    logger.info("Starting to create dummy predictions.")
    memory_limit = TA_MEMORY_LIMIT
    scenario_mock = unittest.mock.Mock()
    scenario_mock.wallclock_limit = int(time_for_task)
    # This stats object is a hack - maybe the SMAC stats object should
    # already be generated here!
    stats = Stats(scenario_mock)
    stats.start_timing()
    ta = ExecuteTaFuncWithQueue(backend=backend,
                                autosklearn_seed=seed,
                                resampling_strategy=resampling_strategy,
                                initial_num_run=num_run,
                                logger=logger,
                                stats=stats,
                                metric=roc_auc,
                                memory_limit=memory_limit,
                                disable_file_output=False,
                                shuffle=shuffle,
                                **kwargs
                                )

    status, cost, runtime, additional_info = \
        ta.run(1, cutoff=int(time_for_task))
    if status == StatusType.SUCCESS:
        logger.info("Finished creating dummy predictions.")
    else:
        logger.error(
            'Error creating dummy predictions: %s ',
            str(additional_info)
        )

    return ta.num_run
