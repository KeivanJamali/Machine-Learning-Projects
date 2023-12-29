from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import f_oneway

from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy.stats import friedmanchisquare


class Normality:
    """
    statistical tests that you can use to check if your data has a -Gaussian distribution-.
    """

    def shapiro(self, data):
        """
        Shapiro Wilk Test
        Tests whether a data sample has a Gaussian distribution.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
        Interpretation: H0: the sample has a Gaussian distribution.
                        H1: the sample does not have a Gaussian distribution.

        stat: is the test statistic value calculated by the Shapiro-Wilk test.
        It measures the deviation of the sample data from the expected normal distribution.

        p: is the p-value associated with the test statistic.It indicates the probability
        of obtaining the observed data if the null hypothesis (data is normally distributed) is true.

        :param data: not str
        """
        stat, p = shapiro(data)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably Gaussian")
        else:
            print("Probably not Gaussian")

    def d_agostinos_k2_test(self, data):
        """
        D’Agostino’s K^2 Test
        Tests whether a data sample has a Gaussian distribution.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
        Interpretation: H0: the sample has a Gaussian distribution.
                        H1: the sample does not have a Gaussian distribution.

        stat: is the test statistic value calculated by the Shapiro-Wilk test.
        It measures the deviation of the sample data from the expected normal distribution.

        p: is the p-value associated with the test statistic.It indicates the probability
        of obtaining the observed data if the null hypothesis (data is normally distributed) is true.

        :param data: not str
        """
        stat, p = normaltest(data)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably Gaussian")
        else:
            print("Probably not Gaussian")

    def anderson_darling(self, data):
        """
        Anderson-Darling Test
        Tests whether a data sample has a Gaussian distribution.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
        Interpretation: H0: the sample has a Gaussian distribution.
                        H1: the sample does not have a Gaussian distribution.

        result.statistic: represents the test statistic calculated by the Anderson-Darling test.
        It measures the discrepancy between the observed data and the expected values under the specified
        distribution (in this case, likely the normal distribution). A larger test statistic indicates a greater
        departure from the specified distribution.

        sl (significance level): represents the significance level at which the critical value is calculated.

        cv (critical value): represents the critical value associated with the specified significance level

        :param data: not str
        """
        result = anderson(data)
        print(f"stat={result.statistic:.3f}")
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < cv:
                print(f"Probably Gaussian at the {sl:.1f}% level")
            else:
                print(f"Probably not Gaussian at the {sl:.1f}% level")


class Correlation:
    """
    This section lists statistical tests that you can use to check if two samples are related.
    """

    def pearson(self, data1, data2):
        """
        Pearson’s Correlation Coefficient
        Tests whether two samples have a linear relationship.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample are normally distributed.
                     Observations in each sample have the same variance.
        Interpretation: H0: the two samples are independent.
                     H1: there is a dependency between the samples.

        stat: This variable represents the Pearson correlation coefficient, which indicates the strength and
        direction of the linear relationship between data1 and data2. The correlation coefficient ranges from -1 to 1,
        where -1 indicates a perfect negative linear relationship, 1 indicates a perfect positive linear relationship,
        and 0 indicates no linear relationship.

        p: This variable represents the p-value associated with the Pearson correlation coefficient.
        The p-value measures the probability of observing the obtained correlation coefficient if the null hypothesis
        (no correlation between the variables) is true. A smaller p-value suggests stronger
        evidence against the null hypothesis.

        :param data1: not str.
        :param data2: not str.
        """
        stat, p = pearsonr(data1, data2)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably independent")
        else:
            print("Probably dependent")

    def spearman(self, data1, data2):
        """
        Spearman’s Rank Correlation
        Tests whether two samples have a monotonic relationship.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample can be ranked.
        Interpretation: H0: the two samples are independent.
                        H1: there is a dependency between the samples.

        stat: This variable represents the Spearman's rank correlation coefficient, which quantifies the strength
        and direction of the monotonic relationship between data1 and data2. The coefficient ranges from -1 to 1,
        where -1 indicates a perfect negative monotonic relationship, 1 indicates a perfect positive monotonic
        relationship, and 0 indicates no monotonic relationship.

        p: This variable represents the p-value associated with the Spearman's rank correlation coefficient.
        The p-value measures the probability of observing the obtained correlation coefficient if the null hypothesis
        (no monotonic relationship between the variables) is true. A smaller p-value suggests stronger evidence against
        the null hypothesis.
        """
        stat, p = spearmanr(data1, data2)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably independent")
        else:
            print("Probably dependent")

    def kendall(self, data1, data2):
        """
        Kendall_Rank_Correlation
        Tests whether two samples have a monotonic relationship.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample can be ranked.
        Interpretation: H0: the two samples are independent.
                        H1: there is a dependency between the samples.

        stat: This variable represents the Spearman's rank correlation coefficient, which quantifies the strength
        and direction of the monotonic relationship between data1 and data2. The coefficient ranges from -1 to 1,
        where -1 indicates a perfect negative monotonic relationship, 1 indicates a perfect positive monotonic
        relationship, and 0 indicates no monotonic relationship.

        p: This variable represents the p-value associated with the Spearman's rank correlation coefficient.
        The p-value measures the probability of observing the obtained correlation coefficient if the null hypothesis
        (no monotonic relationship between the variables) is true. A smaller p-value suggests stronger evidence against
        the null hypothesis.
        """
        stat, p = kendalltau(data1, data2)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably independent")
        else:
            print("Probably dependent")

    def chi_squared(self, data1, data2):
        """
        Chi-Squared Test
        Tests whether two categorical variables are related or independent.
        Assumptions: Observations used in the calculation of the contingency table are independent.
                     25 or more examples in each cell of the contingency table.
        Interpretation: H0: the two samples are independent.
                        H1: there is a dependency between the samples.

        table: a two-dimensional table, represents the frequencies or counts of different categories for two variables.

        stat: This variable represents the test statistic value calculated by the chi-square test. It measures
        the deviation from the expected frequencies under the assumption of independence between the variables.

        p: This variable represents the p-value associated with the test statistic. The p-value measures
        the probability of observing the obtained test statistic if the null hypothesis
        (independence between the variables) is true. A smaller p-value suggests stronger evidence against
        the null hypothesis.

        dof: This variable represents the degrees of freedom associated with the chi-square test.
        It is the number of categories minus 1 for each variable in the contingency table.

        expected: This variable represents the expected frequencies under the assumption of independence
        between the variables. It is a two-dimensional array that has the same shape as the contingency table.

        :param data1: not str.
        :param data2: not str.
        """
        table = [data1, data2]
        stat, p, dof, expected = chi2_contingency(table)
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Probably independent')
        else:
            print('Probably dependent')


class Stationary:
    """
    This section lists statistical tests that you can use to check if a time series is stationary or not.
    """

    def adf(self, data):
        """
        Augmented Dickey-Fuller Unit Root Test
        Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive.
        Assumptions: Observations in are temporally ordered.
        Interpretation: H0: a unit root is present (series is non-stationary).
                        H1: a unit root is not present (series is stationary).

        stat: This variable represents the test statistic value calculated by the ADF test. It is used to assess the
        stationarity of the data. The interpretation of this statistic depends on the context and the critical values.

        p: This variable represents the p-value associated with the test statistic. The p-value measures
        the probability of obtaining the observed data if the null hypothesis (data is non-stationary) is true.
        A smaller p-value suggests stronger evidence against the null hypothesis.

        lags: This variable represents the number of lags used in the ADF test. Lags are the past values of the time
        series used to model and analyze its behavior.

        obs: This variable represents the number of observations used in the ADF test.
        It indicates the length of the time series data.

        crit: This variable represents the critical values for different significance levels. These critical values are
        used to determine whether the test statistic is significant at a given level of confidence.

        t: This variable represents the estimated trend in the time series data.
        """
        stat, p, lags, obs, crit, t = adfuller(data)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably not Stationary")
        else:
            print("Probably Stationary")

    def kpss(self, data):
        """
        Kwiatkowski-Phillips-Schmidt-Shin
        Tests whether a time series is trend stationary or not.
        Assumptions: Observations in are temporally ordered.
        Interpretation: H0: the time series is trend-stationary.
                        H1: the time series is not trend-stationary.

        stat: This variable represents the test statistic value calculated by the KPSS test. It measures the deviation
        from the null hypothesis of stationarity.

        p: This variable represents the p-value associated with the test statistic. The p-value measures
        the probability of observing the obtained test statistic if the null hypothesis (stationarity) is true.
        A smaller p-value suggests stronger evidence against the null hypothesis.

        lags: This variable represents the number of lags used in the KPSS test. Lags are the past values of the
        time series used to model and analyze its behavior.

        crit: This variable represents the critical values for different significance levels. These critical values
        are used to determine whether the test statistic is significant at a given level of confidence.
        """
        stat, p, lags, crit = kpss(data)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably not Stationary")
        else:
            print("Probably Stationary")


class Parametric_Statistical_Hypothesis:
    """
    This section lists statistical tests that you can use to compare data samples.
    """

    def t_test(self, data1, data2):
        """
        Tests whether the means of two independent samples are significantly different.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample are normally distributed.
                     Observations in each sample have the same variance.

        Interpretation: H0: the means of the samples are equal.
                        H1: the means of the samples are unequal.

        :param data1: not str.
        :param data2: not str.
        """
        stat, p = ttest_ind(data1, data2)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably the same distribution")
        else:
            print("Probably different distributions")

    def paired_t_test(self, data1, data2):
        """
        Paired Student’s t-test: Tests whether the means of two paired samples are significantly different.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample are normally distributed.
                     Observations in each sample have the same variance.
                     Observations across each sample are paired.

        Interpretation: H0: the means of the samples are equal.
                        H1: the means of the samples are unequal.

        :param data1: not str.
        :param data2: not str.
        """
        stat, p = ttest_rel(data1, data2)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably the same distribution")
        else:
            print("Probably different distributions")

    def anova(self, *args):
        """
        Analysis of Variance Test
        Tests whether the means of two or more independent samples are significantly different.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample are normally distributed.
                     Observations in each sample have the same variance.

        Interpretation: H0: the means of the samples are equal.
                        H1: one or more of the means of the samples are unequal.

        :param args: not str.

        """
        stat, p = f_oneway(*args)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably the same distribution")
        else:
            print("Probably different distributions")


class Nonparametric_Statistical_Hypothesis:
    """
    Tests whether the distributions of two independent samples are equal or not.
    """

    def mann_whitney_u(self, data1, data2):
        """
        Mann-Whitney U Test: Tests whether the distributions of two independent samples are equal or not.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample can be ranked.

        Interpretation: H0: the distributions of both samples are equal.
                        H1: the distributions of both samples are not equal.

        :param data1: not str.
        :param data2: not str.
        """
        stat, p = mannwhitneyu(data1, data2)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably the same distribution")
        else:
            print("Probably different distributions")

    def wilcoxon(self, data1, data2):
        """
        Wilcoxon Signed-Rank Test: Tests whether the distributions of two paired samples are equal or not.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample can be ranked.
                     Observations across each sample are paired.

        Interpretation: H0: the distributions of both samples are equal.
                        H1: the distributions of both samples are not equal.

        :param data1: not str.
        :param data2: not str.
        """
        stat, p = wilcoxon(data1, data2)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably the same distribution")
        else:
            print("Probably different distributions")

    def kruskal_wallis_h(self, data1, data2):
        """
        Kruskal-Wallis H Test: Tests whether the distributions of two or more independent samples are equal or not.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample can be ranked.

        Interpretation: H0: the distributions of all samples are equal.
                        H1: the distributions of one or more samples are not equal.

        :param data1: str both or not str both.
        :param data2: str both or not str both.
        :return:
        """
        stat, p = kruskal(data1, data2)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably the same distribution")
        else:
            print("Probably different distributions")

    def friedman(self, *args):
        """
        Friedman Test: Tests whether the distributions of two or more paired samples are equal or not.
        Assumptions: Observations in each sample are independent and identically distributed (iid).
                     Observations in each sample can be ranked.
                     Observations across each sample are paired.

        Interpretation: H0: the distributions of all samples are equal.
                        H1: the distributions of one or more samples are not equal.

        :param args:
        :return:
        """
        stat, p = friedmanchisquare(*args)
        print(f"stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print("Probably the same distribution")
        else:
            print("Probably different distributions")
