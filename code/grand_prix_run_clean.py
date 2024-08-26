import os
import shutil
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autogluon.common.savers import save_pd
from autogluon.core.constants import BINARY
from autogluon.core.utils import generate_train_test_split_combined
from autogluon.tabular import TabularPredictor, TabularDataset


@dataclass
class KaggleConf:
    drop_id_columns = True
    holdout_frac = None
    sample = None  # Number of rows to use to train
    predict_holdout = True
    predict_test = True
    cache_oof_proba = True


@dataclass
class CompetitionInfo:
    label = 'class'
    id_columns = ["id"]
    eval_metric = "mcc"
    problem_type = BINARY


@dataclass
class DataInfo:
    data_path = "../data"
    artifacts_path = "../artifacts"

    @property
    def path_train(self):
        return f'{self.data_path}/train.csv'

    @property
    def path_test(self):
        return f'{self.data_path}/test.csv'

    def load_train(self) -> pd.DataFrame:
        return pd.DataFrame(TabularDataset(self.path_train))

    def load_test(self) -> pd.DataFrame:
        return pd.DataFrame(TabularDataset(self.path_test))


def cache_source_code(predictor: TabularPredictor, artifacts_path: str):
    artifact_save_path = os.path.join(artifacts_path, predictor.path)
    os.makedirs(name=artifact_save_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(artifact_save_path, "run.py"))
    return artifact_save_path


def set_unique_categories_to_nan(train_data: pd.DataFrame, test_data: pd.DataFrame, label: str, id_columns: list = None):
    """
    Set categories that are unique to train or test to NaN.
    Running this twice in a row results in no changes in the second run.
    """
    features = list(train_data.columns)
    features = [f for f in features if f != label]
    if id_columns:
        features = [f for f in features if f not in id_columns]
    cat_features = [f for f in features if train_data[f].dtype == "object"]

    train_data = train_data.copy()
    test_data = test_data.copy()

    for f in cat_features:
        print(f"{f}")

        a = train_data[f].value_counts()
        b = test_data[f].value_counts()

        a_d = a.to_dict()
        b_d = b.to_dict()

        missing_in_a = {}
        missing_in_b = {}
        present_in_both = {}
        for k in b_d:
            if k not in a_d:
                missing_in_a[k] = b_d[k]
        for k in a_d:
            if k not in b_d:
                missing_in_b[k] = a_d[k]
        for k in a_d:
            if k in b_d:
                present_in_both[k] = a_d[k] + b_d[k]

        missing_in_a = pd.Series(missing_in_a).sort_values()
        missing_in_b = pd.Series(missing_in_b).sort_values()
        present_in_both = pd.Series(present_in_both).sort_values()

        print(f"\tPresent in Both: {len(present_in_both)}\t| {present_in_both.sum()}")
        print(f"\tMissing in A: {len(missing_in_a)}\t| {missing_in_a.sum()}")
        print(f"\tMissing in B: {len(missing_in_b)}\t| {missing_in_b.sum()}")

        if len(missing_in_b) > 0:
            missing_in_b = missing_in_b.to_dict()
            replace_in_a = {k: np.nan for k in missing_in_b}
            train_data[f] = train_data[f].replace(replace_in_a)

        if len(missing_in_a) > 0:
            missing_in_a = missing_in_a.to_dict()
            replace_in_b = {k: np.nan for k in missing_in_a}
            test_data[f] = test_data[f].replace(replace_in_b)

    return train_data, test_data


if __name__ == '__main__':
    config = KaggleConf()
    comp_info = CompetitionInfo()
    data_info = DataInfo()

    print(f"config:")
    print(config.__dict__)

    train_data_og = data_info.load_train()
    test_data_og = data_info.load_test()

    train_data = train_data_og.copy()
    test_data = test_data_og.copy()

    initial_features = list(train_data.columns)
    initial_features = [f for f in initial_features if f != comp_info.label]

    train_data_partial_og = train_data
    if config.holdout_frac:
        train_data_partial_og, holdout_data_og = generate_train_test_split_combined(train_data_partial_og, label=comp_info.label, problem_type=comp_info.problem_type, test_size=config.holdout_frac, random_state=0)
    else:
        holdout_data_og = None
    if config.sample is not None and (config.sample < len(train_data)):
        train_data_partial_og, _ = generate_train_test_split_combined(
            train_data_partial_og, label=comp_info.label, problem_type=comp_info.problem_type, test_size=1 - config.sample, random_state=0,
        )

    train_data_partial = train_data_partial_og.copy()
    if holdout_data_og is not None:
        holdout_data = holdout_data_og.copy()
        train_data_partial, holdout_data = set_unique_categories_to_nan(train_data=train_data_partial, test_data=holdout_data, label=comp_info.label, id_columns=comp_info.id_columns)
    else:
        holdout_data = None

    train_data_partial, test_data = set_unique_categories_to_nan(train_data=train_data_partial, test_data=test_data, label=comp_info.label, id_columns=comp_info.id_columns)

    if config.drop_id_columns:
        train_data_partial = train_data_partial.drop(columns=comp_info.id_columns)
        if holdout_data is not None:
            holdout_data = holdout_data.drop(columns=comp_info.id_columns)

    print(f"Train Data Rows: {len(train_data_partial)}")
    print(f"Train Data Cols: {len(train_data_partial.columns)}")
    if holdout_data is not None:
        print(f"Holdout Data Rows: {len(holdout_data)}")
        print(f"Holdout Data Cols: {len(holdout_data.columns)}")

    predictor = TabularPredictor(
        label=comp_info.label,
        eval_metric=comp_info.eval_metric,
        problem_type=comp_info.problem_type,
    )

    artifact_save_path = cache_source_code(predictor=predictor, artifacts_path=data_info.artifacts_path)

    predictor.fit(
        train_data_partial,
        presets="best_quality",
        excluded_model_types=["KNN", "CAT"],
        time_limit=3600,
        dynamic_stacking=False,
        ag_args_fit={"ag.stopping_metric": "log_loss"},
    )

    leaderboard_val = predictor.leaderboard(display=True)

    if config.cache_oof_proba:
        oof_pred_probas = predictor.predict_proba_multi(as_multiclass=False)
        oof_pred_probas = pd.DataFrame(oof_pred_probas)
        oof_pred_probas["id"] = train_data_og.loc[oof_pred_probas.index, "id"]
        gt = train_data_og.loc[oof_pred_probas.index, comp_info.label]
        gt_transform = predictor.transform_labels(gt)
        gt_transform = gt_transform.to_frame()
        gt_transform["id"] = train_data_og.loc[gt_transform.index, "id"]

        save_pd.save(path=os.path.join(artifact_save_path, "y_pred_proba_oof.csv"), df=oof_pred_probas[["id", predictor.model_best]])
        save_pd.save(path=os.path.join(artifact_save_path, "y_pred_proba_multi_oof.csv"), df=oof_pred_probas)
        save_pd.save(path=os.path.join(artifact_save_path, "gt_transform_oof.csv"), df=gt_transform)

    if holdout_data is not None and config.predict_holdout:
        ts = time.time()
        leaderboard = predictor.leaderboard(data=holdout_data, display=True)
        te = time.time()

        print(f"Leaderboard Runtime: {te-ts:.2f}s")

        ts = time.time()
        y_pred_proba_holdout = predictor.predict_proba(holdout_data)
        te = time.time()
        print(f"predict_proba | holdout | Runtime: {te - ts:.2f}s")

        score = predictor.evaluate_predictions(y_pred=y_pred_proba_holdout, y_true=holdout_data[comp_info.label])
        print(f"Score: {score[predictor.eval_metric.name]}")

        y_pred_proba_holdout = y_pred_proba_holdout.iloc[:, 1]
        y_pred_proba_holdout = y_pred_proba_holdout.to_frame(name=comp_info.label)
        y_pred_proba_holdout["id"] = holdout_data_og.loc[y_pred_proba_holdout.index, "id"]
        save_pd.save(path=os.path.join(artifact_save_path, "y_pred_proba_holdout.csv"), df=y_pred_proba_holdout)

        gt_holdout = holdout_data_og.loc[y_pred_proba_holdout.index, comp_info.label]
        gt_transform_holdout = predictor.transform_labels(gt_holdout)
        gt_transform_holdout = gt_transform_holdout.to_frame()
        gt_transform_holdout["id"] = holdout_data_og.loc[gt_transform_holdout.index, "id"]
        save_pd.save(path=os.path.join(artifact_save_path, "gt_transform_holdout.csv"), df=gt_transform_holdout)

        y_pred_proba_multi_holdout = predictor.predict_proba_multi(holdout_data, as_multiclass=False)
        y_pred_proba_multi_holdout = pd.DataFrame(y_pred_proba_multi_holdout)
        y_pred_proba_multi_holdout["id"] = holdout_data_og.loc[y_pred_proba_multi_holdout.index, "id"]
        save_pd.save(path=os.path.join(artifact_save_path, "y_pred_proba_multi_holdout.csv"), df=y_pred_proba_multi_holdout)

        ts = time.time()
        fi = predictor.feature_importance(holdout_data)
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(fi)
        te = time.time()
        print(f"feature_importance Runtime: {te - ts:.2f}s")

    if config.predict_test:
        ts = time.time()
        y_pred_proba_test = predictor.predict_proba(test_data)
        te = time.time()
        print(f"predict_proba | test | Runtime: {te - ts:.2f}s")
        y_pred_test = predictor.predict_from_proba(y_pred_proba_test)
        y_pred_proba_test = y_pred_proba_test.iloc[:, 1]

        save_pd.save(path=os.path.join(artifact_save_path, "y_pred_proba_test.csv"), df=y_pred_proba_test.to_frame(name=comp_info.label))
        save_pd.save(path=os.path.join(artifact_save_path, "y_pred_test.csv"), df=y_pred_test.to_frame(name=comp_info.label))

        y_pred_proba_test_w_id = test_data[["id"]].copy()
        y_pred_proba_test_w_id.loc[:, comp_info.label] = y_pred_proba_test
        y_pred_test_w_id = test_data[["id"]].copy()
        y_pred_test_w_id.loc[:, comp_info.label] = y_pred_test

        save_pd.save(path=os.path.join(artifact_save_path, "y_pred_proba_test_w_id.csv"), df=y_pred_proba_test_w_id)
        save_pd.save(path=os.path.join(artifact_save_path, "y_pred_test_w_id.csv"), df=y_pred_test_w_id)

        y_pred_proba_multi_test = predictor.predict_proba_multi(test_data, as_multiclass=False)
        y_pred_proba_multi_test = pd.DataFrame(y_pred_proba_multi_test)
        y_pred_proba_multi_test["id"] = test_data_og.loc[y_pred_proba_multi_test.index, "id"]
        save_pd.save(path=os.path.join(artifact_save_path, "y_pred_proba_multi_test.csv"), df=y_pred_proba_multi_test)
