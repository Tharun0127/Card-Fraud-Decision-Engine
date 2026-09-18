"""Single entry point: rebuild everything from a clean clone.

    python -m src.run_all                  # all stages on the real data
    python -m src.run_all --stage decide   # one stage (earlier artifacts must exist)
    python -m src.run_all --synthetic      # smoke run on a synthetic IEEE-shaped dataset

Stages: data -> features -> train -> decide -> monitor -> report.
Every number that reaches README.md / REPORT.md / FINDINGS.md is written to
outputs/results.json by one of these stages.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ROOT, cost_config, paths, pipeline_config, set_global_seed

STAGES = ("data", "features", "train", "decide", "monitor", "report")


# --------------------------------------------------------------------------- utils

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _clean(o):
    """Make numpy / pandas values JSON-serialisable; NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load_results(p) -> dict:
    return json.loads(p.results.read_text(encoding="utf-8")) if p.results.exists() else {}


def save_results(p, res: dict) -> None:
    p.results.write_text(json.dumps(_clean(res), indent=1, sort_keys=True), encoding="utf-8")


def configs(synthetic: bool) -> tuple[dict, dict]:
    cfg = pipeline_config()
    ccfg = cost_config()
    if synthetic:
        cfg = copy.deepcopy(cfg)
        cfg["paths"] = {"raw_dir": "data/synthetic/raw", "interim_dir": "data/synthetic/interim",
                        "processed_dir": "data/synthetic/processed", "model_dir": "data/synthetic/models",
                        "outputs_dir": "data/synthetic/outputs"}
        cfg["model"]["tuning_grid"] = {"num_leaves": [15], "min_child_samples": [50], "feature_fraction": [0.8]}
        cfg["model"]["max_rounds"] = 300
        cfg["model"]["early_stopping_rounds"] = 30
        cfg["evaluation"]["n_bootstrap"] = 100
        cfg["features"]["v_sample_rows"] = 20_000
        cfg["features"]["v_max_keep"] = 20
        cfg["synthetic"] = True
    return cfg, ccfg


# --------------------------------------------------------------------------- stages

def stage_data(cfg: dict, ccfg: dict) -> None:
    from src.data.load import build_merged
    from src.data.profile import profile, write_data_dictionary
    from src.data.split import time_split
    from src.features.keys import add_keys

    p = paths(cfg).ensure()
    provenance = None
    if cfg.get("synthetic"):
        from src.data.synthetic import make_raw
        tx, ids = make_raw(seed=cfg["seed"])
        tx.to_csv(p.raw / "train_transaction.csv", index=False)
        ids.to_csv(p.raw / "train_identity.csv", index=False)
        (p.interim / "merged.parquet").unlink(missing_ok=True)
    else:
        from src.data.download import download
        provenance = download()
    log("data: merging transaction and identity tables")
    df = build_merged(cfg)
    raw_cols = list(df.columns)
    expected = cfg["kaggle"].get("expected_rows")
    if not cfg.get("synthetic") and expected and len(df) != expected:
        raise SystemExit(f"merged data has {len(df):,} rows, expected {expected:,}; refusing to continue")
    df["split"] = time_split(df, cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    df.to_parquet(p.interim / "base.parquet", index=False)
    log("data: profiling")
    keyed = add_keys(df[["TransactionDT", "card1", "card2", "card3", "card4", "card5", "card6",
                         "addr1", "D1"]].copy())
    prof_df = df.assign(card_key=keyed["card_key"], uid=keyed["uid"])
    prof = profile(prof_df)
    pd.Series(prof["missingness"], name="missing_rate").rename_axis("column").to_csv(p.tables / "missingness.csv")
    pd.DataFrame(prof["by_period"]).to_csv(p.tables / "class_balance_by_period.csv", index=False)
    if not cfg.get("synthetic"):
        write_data_dictionary(prof, raw_cols)
    res = load_results(p)
    res["data"] = {k: v for k, v in prof.items() if k not in ("missingness",)}
    res["data"]["split_boundaries_ok"] = True
    if provenance is not None:
        res["data"]["provenance"] = {"source": cfg["kaggle"]["source"],
                                     "mirror_dataset": cfg["kaggle"].get("mirror_dataset"),
                                     "files": provenance, "expected_rows": cfg["kaggle"]["expected_rows"]}
    res["data"]["split_fractions"] = {"train": cfg["split"]["train_frac"], "valid": cfg["split"]["valid_frac"],
                                      "test": round(1 - cfg["split"]["train_frac"] - cfg["split"]["valid_frac"], 10)}
    save_results(p, res)


def stage_features(cfg: dict, ccfg: dict) -> None:
    from src.data.load import load_base
    from src.features.build import build_features, manifest_tables, save_manifest

    p = paths(cfg).ensure()
    base = load_base(cfg)
    log(f"features: building from {len(base):,} rows")
    frame, manifest = build_features(base, cfg)
    del base
    frame.to_parquet(p.processed / "features.parquet", index=False)
    save_manifest(manifest, p.processed / "feature_manifest.json")
    kept, dropped = manifest_tables(manifest)
    kept.to_csv(p.tables / "feature_manifest.csv", index=False)
    dropped.to_csv(p.tables / "dropped_columns.csv", index=False)
    res = load_results(p)
    fam = kept["family"].value_counts().to_dict()
    res["features"] = {"n_features": manifest["n_features"], "by_family": fam,
                       "n_categorical": len(manifest["categorical"]), "n_dropped": len(manifest["dropped"]),
                       "max_features": cfg["features"]["max_features"],
                       "te_smoothing": cfg["features"]["te_smoothing"], "te_folds": cfg["features"]["te_folds"],
                       "v_corr_threshold": cfg["features"]["v_corr_threshold"],
                       "n_v_kept": int(fam.get("vesta", 0))}
    save_results(p, res)
    log(f"features: {manifest['n_features']} features, {len(manifest['dropped'])} columns dropped")


def _load_features(p):
    frame = pd.read_parquet(p.processed / "features.parquet")
    manifest = json.loads((p.processed / "feature_manifest.json").read_text(encoding="utf-8"))
    for c in manifest["categorical"]:
        frame[c] = pd.Categorical(frame[c].astype("string"), categories=manifest["category_vocab"][c])
    return frame, manifest


def stage_train(cfg: dict, ccfg: dict) -> None:
    from src.models import lgbm

    p = paths(cfg).ensure()
    frame, manifest = _load_features(p)
    feats = manifest["features"]
    cats = manifest["categorical"]
    novel_feats = [f for f in feats if manifest["families"][f] != "velocity"]
    tr = frame["meta_split"].to_numpy() == "train"
    X_tr, y_tr = frame.loc[tr, feats], frame.loc[tr, "meta_isFraud"].to_numpy()
    log(f"train: tuning on {tr.sum():,} training rows with expanding-window CV")
    tuned = lgbm.tune(X_tr, y_tr, cats, cfg, log=log)
    tuned["cv_table"].to_csv(p.tables / "cv_folds_full_model.csv", index=False)
    tuned["cv_summary"].to_csv(p.tables / "cv_summary_full_model.csv", index=False)
    params = {**lgbm.base_params(cfg), **tuned["params"]}
    log(f"train: chosen {tuned['params']} with {tuned['num_rounds']} rounds")
    full = lgbm.fit(X_tr, y_tr, cats, params, tuned["num_rounds"])
    log("train: no-velocity model, same params, own round count from the same CV")
    nv_rounds, nv_cv = lgbm.cv_rounds(frame.loc[tr, novel_feats], y_tr, [c for c in cats if c in novel_feats],
                                      cfg, tuned["params"])
    nv_cv.to_csv(p.tables / "cv_folds_no_velocity_model.csv", index=False)
    novel = lgbm.fit(frame.loc[tr, novel_feats], y_tr, [c for c in cats if c in novel_feats], params, nv_rounds)
    p.models.mkdir(parents=True, exist_ok=True)
    lgbm.save_json(full, p.models / "full_model.json", {"categorical": cats, "params": params,
                                                        "num_rounds": tuned["num_rounds"]})
    lgbm.save_json(novel, p.models / "no_velocity_model.json", {"categorical": cats, "params": params,
                                                                "num_rounds": nv_rounds})
    ev = frame["meta_split"].isin(["valid", "test"]).to_numpy()
    meta_cols = [c for c in frame.columns if c.startswith("meta_")]
    preds = frame.loc[ev, meta_cols].copy()
    preds["score_full"] = full.predict(frame.loc[ev, feats])
    preds["score_no_velocity"] = novel.predict(frame.loc[ev, novel_feats])
    preds.to_parquet(p.processed / "predictions.parquet", index=False)
    gain = full.feature_importance("gain")
    imp = pd.DataFrame({"feature": full.feature_name(), "gain": gain})
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp["family"] = imp["feature"].map(manifest["families"])
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp.to_csv(p.tables / "feature_importance.csv", index=False)
    res = load_results(p)
    summ = tuned["cv_summary"]
    best = summ.sort_values("pr_auc_mean", ascending=False).iloc[0]
    res["model"] = {
        "chosen_params": tuned["params"], "num_rounds_full": tuned["num_rounds"], "num_rounds_no_velocity": nv_rounds,
        "cv_pr_auc_mean": float(best["pr_auc_mean"]), "cv_pr_auc_std": float(best["pr_auc_std"]),
        "cv_no_velocity_pr_auc_mean": float(nv_cv["pr_auc"].mean()),
        "n_grid_configs": int(len(summ)), "cv_folds": cfg["model"]["cv_folds"],
        "n_features_full": len(feats), "n_features_no_velocity": len(novel_feats),
        "learning_rate": cfg["model"]["fixed_params"]["learning_rate"],
        "gain_share_by_family": imp.groupby("family")["gain_share"].sum().to_dict(),
        "n_train_rows": int(tr.sum()),
    }
    save_results(p, res)


def stage_decide(cfg: dict, ccfg: dict) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    from src.decision import sensitivity as sens_mod
    from src.decision.costs import DECLINE, CostParams, policy_summary, row_benefit
    from src.decision.policy import (benefits, evaluate_policies, max_reviews_per_day, pair_net_benefit,
                                     select_policies, split_days)
    from src.decision.swap import characterise, swap_sets
    from src.decision.threshold import CumStats, decide, profit_optimal_threshold, single_sweep, threshold_grid
    from src.models.baselines import amount_threshold, rules_decline
    from src.models.evaluate import mean_diff_ci, metric_diff_ci, precision_recall_at, ranking_metrics, top_k_mask
    from src.reporting import figures

    p = paths(cfg).ensure()
    seed = cfg["seed"]
    ecfg = cfg["evaluation"]
    B, level = ecfg["n_bootstrap"], ecfg["ci_level"]
    preds = pd.read_parquet(p.processed / "predictions.parquet")
    trmeta = pd.read_parquet(p.processed / "features.parquet",
                             columns=["meta_split", "meta_TransactionAmt", "meta_TransactionDT"])
    trmeta = trmeta[trmeta["meta_split"] == "train"]
    va = preds[preds["meta_split"] == "valid"].reset_index(drop=True)
    te = preds[preds["meta_split"] == "test"].reset_index(drop=True)
    yv, yt = va["meta_isFraud"].to_numpy(), te["meta_isFraud"].to_numpy()
    av, at = va["meta_TransactionAmt"].to_numpy(), te["meta_TransactionAmt"].to_numpy()
    dv, dt_days = split_days(va["meta_TransactionDT"]), split_days(te["meta_TransactionDT"])
    base_p = CostParams.from_config(ccfg)
    cap = max_reviews_per_day(ccfg, trmeta["meta_TransactionDT"].to_numpy())
    grid_cfg = ccfg["threshold_grid"]
    log(f"decide: review capacity {cap:.1f}/day; valid {len(va):,} rows / {dv:.1f} days; "
        f"test {len(te):,} rows / {dt_days:.1f} days")

    # rules baseline (pre-registered; fixed percentile of training amounts)
    amt_thr = amount_threshold(trmeta["meta_TransactionAmt"].to_numpy(), cfg["rules_baseline"]["amount_percentile"])
    rules_t = np.where(rules_decline(at, te["meta_uid_prior_cnt"].to_numpy(), amt_thr), DECLINE, 0).astype(np.int8)
    rules_v = np.where(rules_decline(av, va["meta_uid_prior_cnt"].to_numpy(), amt_thr), DECLINE, 0).astype(np.int8)

    # policy selection on validation, evaluation on test
    sel, dec, summ = {}, {}, {}
    for m in ("full", "no_velocity"):
        sel[m] = select_policies(va[f"score_{m}"].to_numpy(), yv, av, dv, base_p, cap, grid_cfg)
        extra = {"rules": rules_t} if m == "full" else None
        dec[m], summ[m] = evaluate_policies(sel[m], te[f"score_{m}"].to_numpy(), yt, at, dt_days, base_p, extra)
    sf, sn = te["score_full"].to_numpy(), te["score_no_velocity"].to_numpy()
    ben = benefits(dec["full"], yt, at, base_p)
    ben_nv = benefits(dec["no_velocity"], yt, at, base_p)

    # daily review load on test for the three-way policy
    day_t = ((te["meta_TransactionDT"] - te["meta_TransactionDT"].min()) // 86400).to_numpy()
    rev_by_day = pd.Series(dec["full"]["profit_three_way"] == 1).groupby(day_t).sum()
    val_rules = policy_summary(rules_v, yv, av, base_p, dv)

    # ranking metrics
    fr = ecfg["precision_at"]
    met = {"full": ranking_metrics(yt, sf, fr), "no_velocity": ranking_metrics(yt, sn, fr)}
    rflag = rules_t == DECLINE
    k_rules = int(rflag.sum())
    rp = float(yt[rflag].mean()) if k_rules else float("nan")
    rr = float(yt[rflag].sum() / yt.sum())
    met["rules"] = {"roc_auc_degenerate": float(roc_auc_score(yt, rflag)),
                    "pr_auc_degenerate": float(average_precision_score(yt, rflag)),
                    "n_flagged": k_rules, "flag_rate": k_rules / len(yt), "precision": rp, "recall": rr,
                    "amount_threshold": amt_thr, "amount_percentile": cfg["rules_baseline"]["amount_percentile"]}
    pm, rm = precision_recall_at(yt, sf, k=k_rules)
    met["full"]["precision_at_rules_volume"], met["full"]["recall_at_rules_volume"] = pm, rm

    log(f"decide: paired bootstrap, {B} resamples")
    lift = {}
    for metric in ("pr_auc", "roc_auc"):
        lift[f"{metric}_full_minus_no_velocity"] = metric_diff_ci(yt, sf, sn, metric, B, seed, level)
    # precision / recall at fixed flag sets (flags fixed on the full test set, rows resampled)
    for f in fr:
        tag = f"{f * 100:g}pct"
        a, b = top_k_mask(sf, frac=f), top_k_mask(sn, frac=f)
        lift[f"precision_at_{tag}_full_minus_no_velocity"] = _ratio_ci(yt, a, b, "precision", B, seed, level)
        lift[f"recall_at_{tag}_full_minus_no_velocity"] = _ratio_ci(yt, a, b, "recall", B, seed, level)
    a = top_k_mask(sf, k=k_rules)
    lift["precision_at_rules_volume_full_minus_rules"] = _ratio_ci(yt, a, rflag, "precision", B, seed, level)
    lift["recall_at_rules_volume_full_minus_rules"] = _ratio_ci(yt, a, rflag, "recall", B, seed, level)
    comps = {
        "profit_single_minus_youden_single": mean_diff_ci(ben["profit_single"], ben["youden_single"], B, seed, level, 1000),
        "three_way_minus_rules": mean_diff_ci(ben["profit_three_way"], ben["rules"], B, seed, level, 1000),
        "three_way_minus_profit_single": mean_diff_ci(ben["profit_three_way"], ben["profit_single"], B, seed, level, 1000),
        "three_way_full_minus_three_way_no_velocity": mean_diff_ci(ben["profit_three_way"], ben_nv["profit_three_way"],
                                                                   B, seed, level, 1000),
        "three_way_minus_approve_all": mean_diff_ci(ben["profit_three_way"], ben["approve_all"], B, seed, level, 1000),
        "rules_minus_approve_all": mean_diff_ci(ben["rules"], ben["approve_all"], B, seed, level, 1000),
        "youden_minus_approve_all": mean_diff_ci(ben["youden_single"], ben["approve_all"], B, seed, level, 1000),
    }

    # AUC-optimal vs profit-optimal, with test-oracle reference
    cs_t = CumStats.build(sf, yt, at, dt_days)
    grid_t = threshold_grid(sf, grid_cfg["n_quantiles"], grid_cfg["n_linear"])
    test_sweep = single_sweep(cs_t, grid_t, base_p)
    t_oracle = profit_optimal_threshold(test_sweep)
    oracle_nb = pair_net_benefit(cs_t, t_oracle, t_oracle, base_p)
    fs = sel["full"]
    auc_vs_profit = {
        "t_youden": fs["t_youden"], "t_profit": fs["t_profit"],
        "threshold_gap": fs["t_profit"] - fs["t_youden"],
        "test_oracle_threshold": t_oracle,
        "test_oracle_net_benefit_per_1000": oracle_nb / len(yt) * 1000,
        "money_left_on_table_per_1000": comps["profit_single_minus_youden_single"]["point"],
        "money_left_on_table_total_test": float(ben["profit_single"].sum() - ben["youden_single"].sum()),
        "money_left_on_table_per_day": float((ben["profit_single"].sum() - ben["youden_single"].sum()) / dt_days),
        "decline_rate_ratio_youden_over_profit": (summ["full"]["youden_single"]["decline_rate"]
                                                  / max(summ["full"]["profit_single"]["decline_rate"], 1e-12)),
        "false_declines_avoided": summ["full"]["youden_single"]["n_false_declines"]
                                  - summ["full"]["profit_single"]["n_false_declines"],
        "validation": {k: policy_summary(decide(va["score_full"].to_numpy(), fs[k2]), yv, av, base_p, dv)
                       for k, k2 in (("youden_single", "t_youden"), ("profit_single", "t_profit"))},
    }

    # swap set: Youden declines that the profit-optimal cutoff approves (and vice versa)
    sw = swap_sets(dec["full"]["profit_single"], dec["full"]["youden_single"])
    frame = pd.DataFrame({"amount": at, "isFraud": yt, "tenure_days": te["meta_card_tenure_days"].to_numpy(),
                          "prior_cnt": te["meta_uid_prior_cnt"].to_numpy(), "ProductCD": te["meta_ProductCD"].to_numpy()})
    both_decline = (dec["full"]["profit_single"] == DECLINE) & (dec["full"]["youden_single"] == DECLINE)
    both_approve = (dec["full"]["profit_single"] == 0) & (dec["full"]["youden_single"] == 0)
    swap_rows = [characterise(sw["a_approves_b_declines"], frame, "profit approves, Youden declines"),
                 characterise(sw["a_declines_b_approves"], frame, "profit declines, Youden approves"),
                 characterise(both_decline, frame, "both decline"),
                 characterise(both_approve, frame, "both approve")]
    swap_df = pd.DataFrame(swap_rows)
    swap_df.to_csv(p.tables / "swap_set.csv", index=False)
    swap_mask = sw["a_approves_b_declines"] | sw["a_declines_b_approves"]
    swap_benefit = {"swap_set_net_benefit_profit": float(ben["profit_single"][swap_mask].sum()),
                    "swap_set_net_benefit_youden": float(ben["youden_single"][swap_mask].sum()),
                    "profit_declines_fewer": bool(sw["a_approves_b_declines"].sum() >= sw["a_declines_b_approves"].sum())}

    # sensitivity (pre-registered grid)
    log("decide: sensitivity sweep")
    cs_v = CumStats.build(va["score_full"].to_numpy(), yv, av, dv)
    sgrid = threshold_grid(va["score_full"].to_numpy(), grid_cfg["n_quantiles"], grid_cfg["n_linear"])
    pgrid = threshold_grid(va["score_full"].to_numpy(), grid_cfg["n_policy_grid"] // 2, 0)
    sens = sens_mod.run(ccfg["sensitivity"], base_p, cs_v, cs_t, sgrid, pgrid, cap,
                        {"rules": rules_t}, yt, at, fs["t_youden"])
    sens["decline_rate_val_at_t_profit"] = cs_v.k_at(sens["t_profit_single"].to_numpy()) / cs_v.n
    sens.to_csv(p.tables / "sensitivity_grid.csv", index=False)
    stab = sens_mod.ranking_stability(sens)
    cap_rows = []
    for share in ccfg["sensitivity"]["review_capacity_share"]:
        c2 = max_reviews_per_day(ccfg, trmeta["meta_TransactionDT"].to_numpy(), share=share)
        s2 = select_policies(va["score_full"].to_numpy(), yv, av, dv, base_p, c2, grid_cfg)
        d2 = decide(sf, s2["t_decline"], s2["t_review"])
        cap_rows.append({"review_capacity_share": share, "max_reviews_per_day": c2, "t_review": s2["t_review"],
                         "t_decline": s2["t_decline"], **policy_summary(d2, yt, at, base_p, dt_days)})
    pd.DataFrame(cap_rows).to_csv(p.tables / "capacity_sensitivity.csv", index=False)
    # range of absolute dollars across the grid
    nb_range = {c: {"min": float(sens[c].min()), "max": float(sens[c].max())}
                for c in sens.columns if c.startswith("nb1000_")}

    # tables
    headline = _headline_table(summ, dec)
    headline.to_csv(p.tables / "headline_policies.csv", index=False)
    pd.DataFrame([{"model": m, **v} for m, v in met.items() if m != "rules"]).to_csv(
        p.tables / "ranking_metrics.csv", index=False)
    pd.DataFrame([{"comparison": k, **v} for k, v in {**lift, **comps}.items()]).to_csv(
        p.tables / "bootstrap_comparisons.csv", index=False)
    rev_by_day.rename("reviews").rename_axis("test_day").to_csv(p.tables / "test_reviews_per_day.csv")

    # figures
    val_sweep = fs["val_sweep"]
    marks = {
        "Youden-J (AUC-optimal)": (summ["full"]["youden_single"]["decline_rate"],
                                   summ["full"]["youden_single"]["net_benefit_per_1000"], figures.ORANGE),
        "Profit-optimal": (summ["full"]["profit_single"]["decline_rate"],
                           summ["full"]["profit_single"]["net_benefit_per_1000"], figures.AQUA),
    }
    figures.profit_curve(val_sweep, test_sweep, marks, p.figures / "profit_curve.png")
    fpr, tpr, _ = roc_curve(yt, sf)
    pts = {}
    for name, key, color in (("Youden-J", "youden_single", figures.ORANGE), ("Profit-optimal", "profit_single", figures.AQUA)):
        d = dec["full"][key] == DECLINE
        pts[name] = (float(d[yt == 0].mean()), float(d[yt == 1].mean()), color)
    figures.roc_with_cutoffs(fpr, tpr, pts, p.figures / "roc_cutoffs.png")
    pr_f = precision_recall_curve(yt, sf)
    pr_n = precision_recall_curve(yt, sn)
    figures.pr_curves({"Full model": (pr_f[1], pr_f[0]), "No-velocity model": (pr_n[1], pr_n[0])},
                      (rr, rp), p.figures / "pr_curves.png")
    tw = fs["val_three_way"]
    t_min = float(np.quantile(va["score_full"].to_numpy(), 0.80))  # region declining <= 20% of traffic
    figures.three_way_heatmap(tw["matrix"], tw["thresholds"], (fs["t_review"], fs["t_decline"]), len(va),
                              t_min, p.figures / "three_way_policy.png")
    if sw["a_approves_b_declines"].sum() >= sw["a_declines_b_approves"].sum():
        sets = {"Swap set (profit approves, Youden declines)": frame[sw["a_approves_b_declines"]]}
    else:
        sets = {"Swap set (profit declines, Youden approves)": frame[sw["a_declines_b_approves"]]}
    sets["Both decline"] = frame[both_decline]
    figures.swap_figure(sets, p.figures / "swap_set.png")
    d0 = base_p
    sub = sens[(sens["review_cost"] == d0.review_cost) & (sens["review_catch_rate"] == d0.review_catch_rate)]
    piv = sub.pivot_table(index="churn_penalty", columns="margin_rate", values="decline_rate_val_at_t_profit")
    figures.sensitivity_heatmap(piv * 100, "Profit-optimal single threshold:\nshare of validation declined (%)",
                                "% of transactions declined", p.figures / "sensitivity_threshold_margin_churn.png",
                                fmt="{:.2f}")
    piv_t = sub.pivot_table(index="churn_penalty", columns="margin_rate", values="t_profit_single")
    figures.sensitivity_heatmap(piv_t, "Profit-optimal single threshold (score)", "score threshold",
                                p.figures / "sensitivity_threshold_score.png")
    sub2 = sens[(sens["margin_rate"] == d0.margin_rate) & (sens["churn_penalty"] == d0.churn_penalty)]
    piv2 = sub2.pivot_table(index="review_catch_rate", columns="review_cost", values="t_review")
    figures.sensitivity_heatmap(piv2, "Three-way policy: optimal t_review", "score threshold",
                                p.figures / "sensitivity_review_threshold.png")
    sens["gap_three_way_minus_youden"] = sens["nb1000_profit_three_way"] - sens["nb1000_youden_single"]
    piv3 = sens.pivot_table(index="churn_penalty", columns="margin_rate", values="gap_three_way_minus_youden", aggfunc="min")
    figures.sensitivity_heatmap(piv3, "Worst-case (over review params) test advantage of\nthree-way policy over Youden-J, $ per 1,000",
                                "$ per 1,000", p.figures / "sensitivity_ranking_gap.png", fmt="{:.0f}")
    imp = pd.read_csv(p.tables / "feature_importance.csv")
    figures.importance_fig(imp, p.figures / "feature_importance.png")

    # hypotheses (PRE_REGISTRATION §6)
    h = {
        "H1": {"pass": comps["profit_single_minus_youden_single"]["point"] > 0
               and comps["profit_single_minus_youden_single"]["excludes_zero"]},
        "H2": {"pass": comps["three_way_minus_rules"]["point"] > 0 and comps["three_way_minus_rules"]["excludes_zero"]},
        "H3": {"pass": lift["pr_auc_full_minus_no_velocity"]["point"] > 0
               and lift["pr_auc_full_minus_no_velocity"]["excludes_zero"]},
        "H4": {"pass": stab["share_all_orderings"] >= 0.8, "threshold_share": 0.8},
    }
    h["project_success"] = h["H1"]["pass"] and h["H2"]["pass"]

    res = load_results(p)
    res["costs"] = {**base_p.to_dict(), "max_reviews_per_day": cap, "review_capacity_share": ccfg["review_capacity_share"],
                    "sensitivity_grid": ccfg["sensitivity"]}
    res["evaluation"] = {"test": met, "lift": lift, "n_bootstrap": B, "ci_level": level,
                         "test_rows": int(len(te)), "test_days": dt_days, "valid_rows": int(len(va)), "valid_days": dv,
                         "test_fraud_rate": float(yt.mean()), "test_fraud_dollars": float(at[yt == 1].sum())}
    res["decision"] = {
        "thresholds": {m: {k: sel[m][k] for k in ("t_youden", "t_profit", "t_review", "t_decline")} for m in sel},
        "policies_test": summ, "validation_three_way": {k: v for k, v in fs["val_three_way"].items()
                                                         if k not in ("matrix", "thresholds")},
        "validation_rules": val_rules,
        "comparisons": comps, "auc_vs_profit": auc_vs_profit,
        "test_reviews_per_day_max": float(rev_by_day.max()), "test_reviews_per_day_p95": float(rev_by_day.quantile(0.95)),
        "test_days_over_capacity": int((rev_by_day > cap).sum()), "test_n_days": int(len(rev_by_day)),
        "swap": {"rows": swap_rows, **swap_benefit},
    }
    res["sensitivity"] = {"ranking": stab, "net_benefit_per_1000_range": nb_range, "capacity": cap_rows,
                          "t_profit_range": [float(sens["t_profit_single"].min()), float(sens["t_profit_single"].max())],
                          "decline_rate_range": [float(sens["decline_rate_val_at_t_profit"].min()),
                                                 float(sens["decline_rate_val_at_t_profit"].max())],
                          "n_cells": int(len(sens))}
    res["hypotheses"] = h
    save_results(p, res)
    log(f"decide: H1={h['H1']['pass']} H2={h['H2']['pass']} H3={h['H3']['pass']} H4={h['H4']['pass']}")


def _ratio_ci(y, flag_a, flag_b, kind: str, B: int, seed: int, level: float) -> dict:
    """Paired bootstrap of precision/recall difference for two fixed flag sets."""
    from src.models.evaluate import paired_bootstrap
    y = np.asarray(y)

    def val(yy, f):
        if kind == "precision":
            return yy[f].mean() if f.any() else np.nan
        return yy[f].sum() / yy.sum() if yy.sum() else np.nan

    point = val(y, flag_a) - val(y, flag_b)
    ci = paired_bootstrap(lambda idx: val(y[idx], flag_a[idx]) - val(y[idx], flag_b[idx]), len(y), B, seed, level)
    return {"point": float(point), **ci}


def _headline_table(summ: dict, dec: dict) -> pd.DataFrame:
    order = [("full", "approve_all", "Approve everything"),
             ("full", "rules", "Rules baseline (amount > P90 and new card)"),
             ("full", "youden_single", "Full model, Youden-J cutoff (AUC-optimal)"),
             ("full", "profit_single", "Full model, profit-optimal cutoff"),
             ("no_velocity", "profit_three_way", "No-velocity model, three-way policy"),
             ("full", "profit_three_way", "Full model, three-way policy (capacity-constrained)")]
    rows = []
    for m, k, label in order:
        s = summ[m][k]
        rows.append({"policy": label, "model": m, "key": k, **s})
    return pd.DataFrame(rows)


def stage_monitor(cfg: dict, ccfg: dict) -> None:
    from src.monitoring.psi import feature_psi, psi_numeric
    from src.monitoring.stability import score_stability
    from src.reporting import figures

    p = paths(cfg).ensure()
    mc = cfg["monitoring"]
    frame, manifest = _load_features(p)
    feats, cats = manifest["features"], manifest["categorical"]
    tr = frame[frame["meta_split"] == "train"]
    te = frame[frame["meta_split"] == "test"]
    log("monitor: PSI per feature, train vs test")
    psi = feature_psi(tr, te, feats, cats, mc["psi_bins"], mc["psi_warn"], mc["psi_alert"])
    psi["family"] = psi["feature"].map(manifest["families"])
    psi.to_csv(p.tables / "psi_train_vs_test.csv", index=False)
    figures.psi_bar(psi, mc["psi_warn"], mc["psi_alert"], p.figures / "psi_top_features.png")
    preds = pd.read_parquet(p.processed / "predictions.parquet")
    va, tp = preds[preds["meta_split"] == "valid"], preds[preds["meta_split"] == "test"]
    res = load_results(p)
    th = res["decision"]["thresholds"]["full"]
    stab = score_stability(va["score_full"].to_numpy(), tp["meta_TransactionDT"].to_numpy(), tp["score_full"].to_numpy(),
                           tp["meta_isFraud"].to_numpy(), mc["score_window_days"], mc["psi_bins"],
                           mc["psi_warn"], mc["psi_alert"], th["t_decline"], th["t_review"])
    stab.to_csv(p.tables / "score_stability_test.csv", index=False)
    figures.score_stability_fig(stab, mc["psi_warn"], mc["psi_alert"], p.figures / "score_stability.png")
    sig = psi[psi["band"] == "significant"]
    mod = psi[psi["band"] == "moderate"]
    res["monitoring"] = {
        "psi_warn": mc["psi_warn"], "psi_alert": mc["psi_alert"], "n_features": len(psi), "plan": mc["plan"],
        "n_significant": int(len(sig)), "n_moderate": int(len(mod)),
        "n_stable": int((psi["band"] == "stable").sum()),
        "significant_features": sig[["feature", "psi", "family"]].to_dict("records"),
        "moderate_features": mod[["feature", "psi", "family"]].to_dict("records"),
        "significant_by_family": sig["family"].value_counts().to_dict(),
        "max_psi": float(psi["psi"].max()),
        "score_windows": stab.to_dict("records"),
        "score_psi_max": float(stab["score_psi_vs_valid"].max()),
        "score_window_days": mc["score_window_days"],
        "pr_auc_window_min": float(stab["pr_auc"].min()), "pr_auc_window_max": float(stab["pr_auc"].max()),
        "score_psi_valid_vs_test": float(psi_numeric(va["score_full"].to_numpy(), tp["score_full"].to_numpy(),
                                                   mc["psi_bins"])),
    }
    save_results(p, res)
    log(f"monitor: {len(sig)} significant, {len(mod)} moderate PSI breaches")


def run_test_suite(out_xml) -> dict:
    """Run pytest and return pass / fail / skip counts for the verification section.

    The committed-documents check is deselected here because the documents are
    re-rendered after this runs; the report stage performs that check itself.
    """
    import subprocess
    import xml.etree.ElementTree as ET

    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={out_xml}",
           "--deselect", "tests/test_documents.py::test_committed_documents_only_contain_numbers_from_results"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    root = ET.parse(out_xml).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    total, fail, err, skip = (int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    files = sorted({(c.get("classname") or "").split(".")[1] for c in suite.iter("testcase")
                    if "." in (c.get("classname") or "")})
    return {"tests_total": total, "tests_passed": total - fail - err - skip, "tests_failed": fail + err,
            "tests_skipped": skip, "test_files": len(files), "pytest_exit_code": proc.returncode}


def stage_report(cfg: dict, ccfg: dict) -> None:
    from src.reporting import render
    from src.reporting.check_numbers import check_documents

    p = paths(cfg)
    res = load_results(p)
    res["meta"] = {"seed": cfg["seed"], "python_version": platform.python_version(),
                   "python_minor": float(".".join(platform.python_version().split(".")[:2])),
                   "lightgbm_version": lgb.__version__, "n_threads": cfg["n_threads"],
                   "per_n_transactions": 1000}
    log("report: running the test suite for the verification record")
    res["verification"] = run_test_suite(p.outputs / "pytest_report.xml")
    log(f"report: tests {res['verification']['tests_passed']} passed, {res['verification']['tests_failed']} failed")
    save_results(p, res)
    out_dir = p.outputs / "docs" if cfg.get("synthetic") else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    render.render_all(res, out_dir)
    problems = check_documents(out_dir, p.results)
    if problems:
        raise SystemExit("Numbers in documents not found in results.json:\n" + "\n".join(problems))
    log("report: documents rendered and number check passed")


# --------------------------------------------------------------------------- main

def run(stages=STAGES, synthetic: bool = False) -> None:
    cfg, ccfg = configs(synthetic)
    set_global_seed(cfg["seed"])
    fns = {"data": stage_data, "features": stage_features, "train": stage_train, "decide": stage_decide,
           "monitor": stage_monitor, "report": stage_report}
    for s in stages:
        t0 = time.time()
        log(f"=== stage {s} ===")
        fns[s](cfg, ccfg)
        log(f"=== stage {s} done in {time.time() - t0:.0f}s ===")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, action="append", help="run only these stages (repeatable)")
    ap.add_argument("--synthetic", action="store_true", help="smoke run on synthetic data")
    a = ap.parse_args(argv)
    run(a.stage or STAGES, synthetic=a.synthetic)


if __name__ == "__main__":
    main()
