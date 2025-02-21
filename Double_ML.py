import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


class DoubleML:
    """Double ML model leverage Frish-Waugh-Lovell theorem which debiase the data using Orthgonalization

    Chernozhukov et al (2016), Double/Debiased Machine Learning for Treatment and Causal Parameters

    This three stages algorithms estimator
    Stage one:
        - Train an ML regression model to predict outcome Y using covariates X
        - Train an ML regression model to predict treatment T using covariates X

    Stage two:
        - calculate out of fold prediction residual from the outcome model
        - calculate out of fold prediction residual from the treatment model

    Stage Three:
        - Regress residuals of the outcome on residuals of the treatment

    """

    def __init__(
        self,
        outcome_model=None,
        treatment_model=None,
        final_model=None,
        base_model_params=None,
        final_model_params=None,
        cv=5,
        binary_outcome=True,
    ):
        if outcome_model == None:
            if binary_outcome:
                self.outcome_model = GradientBoostingClassifier()
            else:
                self.outcome_model = GradientBoostingRegressor()
        else:
            self.outcome_model = outcome_model

        if treatment_model == None:
            self.treatment_model = GradientBoostingRegressor()
        else:
            self.treatment_model = treatment_model

        if final_model == None:
            self.final_model = GradientBoostingRegressor()
        else:
            self.final_model = final_model

        if base_model_params == None:
            self.base_model_params = dict()
        else:
            self.base_model_params = base_model_params

        if final_model_params == None:
            self.final_model_params = dict()
        else:
            self.final_model_params = final_model_params

        self.cv = cv
        self.binary_outcome = binary_outcome

    def _cv_estimate(self, train_data, n_splits, model, model_params, X, y, binary):

        cv = KFold(n_splits=n_splits)
        models = []
        cv_pred = pd.Series(np.nan, index=train_data.index)
        for train, test in cv.split(train_data):
            m = model(**model_params)
            m.fit(train_data[X].iloc[train], train_data[y].iloc[train])
            if binary:
                cv_pred.iloc[test] = m.predict_proba(train_data[X].iloc[test])[:, 1]
            else:
                cv_pred.iloc[test] = m.predict(train_data[X].iloc[test])
            models += [m]

        return cv_pred, models

    def _ensamble_cv_pred(self, df, models, X, binary):
        if binary:
            ensambele_res = np.mean(
                [m.predict_proba(df[X])[:, 1] for m in models], axis=0
            )
        else:
            ensambele_res = np.mean([m.predict(df[X]) for m in models], axis=0)

        return ensambele_res

    def fit(
        self,
        data,
        y,
        T,
        X,
        outcome_model=None,
        treatment_model=None,
        cv=None,
    ):
        if outcome_model is None:
            outcome_model = self.outcome_model

        if treatment_model is None:
            treatment_model = self.treatment_model

        self.T = T
        self.y = y
        self.X = X
        self.data = data

        ## Step 1: train outcome model using X predict Y
        y_hat, self.outcome_model_fitted = self._cv_estimate(
            train_data=data,
            n_splits=self.cv,
            model=self.outcome_model,
            model_params=self.base_model_params,
            X=X,
            y=y,
            binary=self.binary_outcome,
        )

        ## Step 1: train treatment model using X predict T
        t_hat, self.treatment_model_fitted = self._cv_estimate(
            train_data=data,
            n_splits=self.cv,
            model=self.treatment_model,
            model_params=self.base_model_params,
            X=X,
            y=T,
            binary=False,
        )

        ## Step 2: perform orthognoalzation calculate prediction residul from outcome and treatment model

        y_res = data[y] - y_hat
        t_res = data[T] - t_hat

        self.y_res = y_res
        self.t_res = t_res

        ## Step 3: fit a regression model on residual of treatment and outcome

        # create treatment weight based on causal loss (R loss) function
        w_t = t_res**2

        # create transformed response
        y_star = y_res / t_res

        ### create a causal curve model
        final_model_curve = self.final_model(**self.final_model_params)
        self.final_model_curve_fitted = final_model_curve.fit(
            X=data[X].assign(**{T: t_res}), y=y_res, sample_weight=w_t
        )

        # final_model_curve = self.final_model(**self.final_model_params)
        # self.final_model_curve_fitted = final_model_curve.fit(
        #     X=data[X].assign(**{T: t_res}), y=y_res)

        ### create a ITE model
        # to predict ITE, we train another model without
        final_model_ite = self.final_model(**self.final_model_params)
        self.final_model_ite_fitted = final_model_ite.fit(
            X=data[X], y=y_star, sample_weight=w_t
        )

        return (
            self.outcome_model_fitted,
            self.treatment_model_fitted,
            self.final_model_curve_fitted,
            self.final_model_ite_fitted,
        )

    def predict(self, X):
        # make ITE prediction for each unit
        return self.final_model_ite_fitted.predict(X=X)

    def estimate_causal_curve(
        self,
        data=None,
        X=None,
        T=None,
        y=None,
        split_dim=None,
        stratified_sample=True,
        n_samples=10,
        sim_range=np.linspace(0, 10, 11),
        seed=123,
    ):
        """
        Estimating the effect of continuous treatments with non-parametric methods like DoubleML comes with limitations.
        These methods can only approximate the treatment effect locally, meaning they estimate the immediate impact of a treatment change on the outcome at the observed treatment level.
        To overcome this limitation, we simulate what the outcome would be if the treatment were set to different levels. This technique helps us reconstruct the entire response curve and understand the full range of treatment effects.

        """

        # if self.binary_treatment:
        #     raise ValueError("make_causal_curve only works on continuous treatment")
        if self.outcome_model_fitted is None and self.treatment_model_fitted is None:
            raise ValueError("no outcome_model and treatment_model detected")

        if data is None:
            data = self.data

        if X is None:
            X = self.X

        if T is None:
            T = self.T

        if y is None:
            y = self.y

        if stratified_sample:
            # stratified sampling (with replacement) the data by treatment so that each level of treatment will have N random samples
            data_curve = (
                data.groupby(T, group_keys=False)
                .apply(lambda x: x.sample(n_samples, replace=True, random_state=seed))
                .reset_index(drop=True)
            )
        else:
            data_curve = data.sample(n=n_samples, random_state=seed, replace=True)

        # For each unit from the samples above, insert a range of artifical treatment levels
        pred_curve = (
            data_curve.rename(columns={T: f"{T}_FACTUAL"})
            .assign(key=1)
            .reset_index()
            .merge(pd.DataFrame(dict(key=1, sim_trt=sim_range)), on="key", how="right")
            .drop(columns=["key"])
        )
        pred_curve.rename(columns={"sim_trt": f"{T}_SIM"}, inplace=True)

        # get CV prediction from trained outcome and treatment models
        y_hat_curve = self._ensamble_cv_pred(
            df=pred_curve,
            models=self.outcome_model_fitted,
            X=X,
            binary=self.binary_outcome,
        )

        t_hat_curve = self._ensamble_cv_pred(
            df=pred_curve,
            models=self.treatment_model_fitted,
            X=X,
            binary=False,
        )

        ## Orthogonalization ##
        # calculate treatment model out of fold prediction residual
        treatment_model_res = pred_curve[f"{T}_SIM"] - t_hat_curve

        self.treatment_model_res_sim = treatment_model_res

        pred_curve["yhat_impact"] = self.final_model_curve_fitted.predict(
            X=pred_curve[X].assign(**{T: treatment_model_res})
        )

        pred_curve["yhat_base"] = y_hat_curve

        pred_curve["yhat_base_plus_impact"] = (
            pred_curve["yhat_base"] + pred_curve["yhat_impact"]
        )

        self.pred_curve = pred_curve

        curve_output = pred_curve[
            [
                "index",
                f"{T}_FACTUAL",
                f"{T}_SIM",
                f"{y}",
                f"{split_dim}",
                "yhat_base",
                "yhat_impact",
                "yhat_base_plus_impact",
            ]
        ]

        return curve_output

    def _process_curve_result(
        self, df_curve, df_model=None, ground_truth_col=None, T=None, y=None
    ):

        if df_model is None:
            df_model = self.data
        if T is None:
            T = self.T
        if y is None:
            y = self.y

        alpha = 0.05

        def q1(x):
            return x.quantile(0.25)

        def q3(x):
            return x.quantile(0.75)

        f = {
            "yhat_impact": ["mean", "median", "var", "std", q1, q3],
        }

        df_agg = df_curve.groupby([f"{T}_SIM"]).agg(f)
        df_agg.columns = (
            df_agg.columns.get_level_values(0)
            + "_"
            + df_agg.columns.get_level_values(1)
        )
        df_agg = df_agg.reset_index()

        if ground_truth_col is None:

            sample_size_df = (
                df_model.groupby(T)[T]
                .describe()["count"]
                .reset_index()
                .rename(columns={"count": "sample_size"})
            )

        else:
            sample_size_df = (
                df_model.groupby(T)[ground_truth_col]
                .describe()[["count", "50%"]]
                .reset_index()
                .rename(columns={"count": "sample_size", "50%": "ground_truth"})
            )

        actual_df = (
            df_curve.groupby(f"{T}_FACTUAL")["yhat_base"]
            .median()
            .reset_index()
            .rename(columns={"yhat_base": "actual"})
        )

        df_agg_sample_size = pd.merge(
            df_agg, sample_size_df, left_on=f"{T}_SIM", right_on=T, how="left"
        ).drop(columns=T)

        df_agg_final = pd.merge(
            df_agg_sample_size,
            actual_df,
            left_on=f"{T}_SIM",
            right_on=f"{T}_FACTUAL",
            how="left",
        )

        df_agg_final["SE"] = np.sqrt(
            df_agg_final["yhat_impact_std"] / (df_agg_final["sample_size"] * (0.05))
        )

        df_agg_final["yhat_impact_upper"] = df_agg_final["yhat_impact_q3"]
        df_agg_final["yhat_impact_lower"] = df_agg_final["yhat_impact_q1"]

        df_agg_final["actual_baseline"] = df_model[y].mean()

        df_agg_final["actual_plus_impact_median"] = (
            df_agg_final["actual_baseline"] + df_agg_final["yhat_impact_median"]
        )
        df_agg_final["actual_plus_impact_q3"] = (
            df_agg_final["actual_baseline"] + df_agg_final["yhat_impact_q3"]
        )
        df_agg_final["actual_plus_impact_q1"] = (
            df_agg_final["actual_baseline"] + df_agg_final["yhat_impact_q1"]
        )

        df_agg_final["actual_plus_impact_mean"] = (
            df_agg_final["actual_baseline"] + df_agg_final["yhat_impact_mean"]
        )

        df_agg_final["actual_plus_impact_upper"] = (
            df_agg_final["actual_plus_impact_mean"] + df_agg_final["SE"]
        )

        df_agg_final["actual_plus_impact_lower"] = (
            df_agg_final["actual_plus_impact_mean"] - df_agg_final["SE"]
        )

        return df_agg_final

    def plot_causal_curve(
        self,
        est_curve_output,
        ground_truth_col=None,
        agg_func="median",
        figsize=(15, 10),
    ):

        T = self.T

        df_curve_agg = self._process_curve_result(
            df_curve=est_curve_output, ground_truth_col=ground_truth_col
        )

        df_curve_agg[f"{T}_SIM"] = df_curve_agg[f"{T}_SIM"].astype(str)

        sns.set_context("talk")

        fig, ax1 = plt.subplots(figsize=figsize)

        if agg_func == "median":
            sns.lineplot(
                data=df_curve_agg,
                x=f"{T}_SIM",
                y="yhat_impact_median",
                label=f"Est.Impact",
                color="r",
                ax=ax1,
            )

            sns.lineplot(
                data=df_curve_agg,
                x=f"{T}_SIM",
                y="actual_plus_impact_median",
                label=f"Est.Debiased Curve",
                color="b",
                ax=ax1,
            )

            ax1.fill_between(
                x=f"{T}_SIM",
                y1="actual_plus_impact_q1",
                y2="actual_plus_impact_q3",
                color="b",
                alpha=0.4,
                data=df_curve_agg,
            )
            intersetion = abs(
                df_curve_agg["actual_baseline"]
                - df_curve_agg["actual_plus_impact_median"]
            ).idxmin()
        else:
            sns.lineplot(
                data=df_curve_agg,
                x=f"{T}_SIM",
                y="yhat_impact_mean",
                label=f"Est.Impact",
                color="r",
                ax=ax1,
            )

            sns.lineplot(
                data=df_curve_agg,
                x=f"{T}_SIM",
                y="actual_plus_impact_mean",
                label=f"Est.Debiased Curve",
                color="b",
                ax=ax1,
            )

            ax1.fill_between(
                x=f"{T}_SIM",
                y1="actual_plus_impact_lower",
                y2="actual_plus_impact_upper",
                color="b",
                alpha=0.4,
                data=df_curve_agg,
            )

            intersetion = abs(
                df_curve_agg["actual_baseline"]
                - df_curve_agg["actual_plus_impact_mean"]
            ).idxmin()

        sns.lineplot(
            data=df_curve_agg,
            x=f"{T}_SIM",
            y="actual_baseline",
            label=f"Observed Baseline",
            color="black",
            ax=ax1,
        )

        if ground_truth_col is not None:

            sns.lineplot(
                data=df_curve_agg,
                x=f"{T}_SIM",
                y="ground_truth",
                label=f"Ground Truth",
                color="green",
                linestyle="-.",
                ax=ax1,
            )
        else:
            pass

        ax1.axvline(
            x=intersetion,
            color="gray",
            linestyle="--",
            label="Intersection",
        )
        ax1.set_ylabel(f"{self.y}")

        ax2 = plt.twinx(ax=ax1)
        ax2 = sns.barplot(data=df_curve_agg, x=f"{T}_SIM", y="sample_size", alpha=0.3)
        ax2.set_ylabel("Sample Size")
        ax2.grid(False)

        plt.tight_layout()
        plt.title("Aggregated Causal Curve")
        plt.show()

        return df_curve_agg
