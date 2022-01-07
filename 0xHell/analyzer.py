import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# plt.rcParams["figure.figsize"] = (16, 10)

class Analyzer:
    df = None
    groupBy = "round_number"

    def loadData(self):
        self.df = pd.read_csv('data.csv')
        self.df["region"].replace(
            ['none', 'nan', 'undefined', np.nan], "unknown", inplace=True)
        self.df["total"] = self.df["total"].apply(
            lambda x: getFloatFromPrice(x))
        self.df["match_amount"] = self.df["match_amount"].apply(
            lambda x: getFloatFromPrice(x))
        self.df["num_contributions"] = self.df["num_contributions"].apply(
            lambda x: getFloatFromPrice(x))
        self.df["num_unique_contributors"] = self.df["num_unique_contributors"].apply(
            lambda x: getFloatFromPrice(x))
        self.df["crowdfund_amount_contributions_usd"] = self.df["crowdfund_amount_contributions_usd"].apply(
            lambda x: getFloatFromPrice(x))

        self.df["region_s"] = self.df["region"]
        self.df["region_s"].replace(['oceania'], "OC", inplace=True)
        self.df["region_s"].replace(['north_america'], "NA", inplace=True)
        self.df["region_s"].replace(['latin_america'], "SA", inplace=True)
        self.df["region_s"].replace(['india'], "IND", inplace=True)
        self.df["region_s"].replace(['southeast_asia'], "SEA", inplace=True)
        self.df["region_s"].replace(['middle_east'], "ME", inplace=True)
        self.df["region_s"].replace(['east_asia'], "EA", inplace=True)
        self.df["region_s"].replace(['africa'], "AF", inplace=True)
        self.df["region_s"].replace(['europe'], "EU", inplace=True)
        self.df["region_s"].replace(['unknown'], "*UNK*", inplace=True)

    def plotNestedPie(self, outerVals, outerLabels, innerLabels):
        fig, ax = plt.subplots()
        ax.pie([k.sum() for k in outerVals], radius=1,
               wedgeprops=dict(width=0.3, edgecolor='w'),
               labels=outerLabels,
               labeldistance=0.82)
        ax.pie(np.hstack(outerVals), radius=1-0.3,
               wedgeprops=dict(width=0.7, edgecolor='w'),
               labels=innerLabels,
               labeldistance=0.73)

        ax.tick_params(axis="x", direction="in", pad=-10)
        ax.set(aspect="equal", title='Pie plot with `ax.pie`')
        ax.set_title("plotNestedPie")
        # ax.legend(loc="upper left")

        plt.show()

    def plotNested(self, target="total", subGroup="region_s"):
        sums = self.df.groupby([self.groupBy, subGroup])[target].sum()
        (v, l) = self.getRoundData(target, subGroup)
        self.plotNestedPie(v,  np.unique([g for (g, r) in sums.keys()]), [
            r for (g, r) in sums.keys()])

    # TODO replace with unique
    def getMaxRound(self):
        return self.df[self.groupBy].max()

    def getRoundSums(self, target="total", subGroup="region_s"):
        (v, l) = self.getRoundData(target, subGroup)
        sums = [k.sum() for k in v]
        return sums

    def getRoundData(self, target="total", subGroup="region_s"):
        groups = self.df.groupby([self.groupBy, subGroup])
        sums = groups[target].sum()
        v = [[] for _ in range(self.getMaxRound())]
        l = [[] for _ in range(self.getMaxRound())]
        for g, r in sums.keys():
            l[g-1].append(r)
            v[g-1].append(sums[(g, r)])
        v = [np.array(x) for x in v]
        l = [np.array(x) for x in l]
        return (v, l)

    def plotByTotal(self, target="total", subGroup="region_s", operation="div"):
        regionSums = self.getGroupedBySub(target=target, subGroup=subGroup)
        regions = self.df[subGroup].unique()
        roundSums = self.getRoundSums(target=target)
        regionDiff = {key: [regionSums[key][i] / roundSums[i] if operation == "div" else roundSums[i] - regionSums[key][i]
                            for i in range(self.getMaxRound())] for key in regions}

        fig, ax = plt.subplots()
        labels = [i+1 for i in range(self.getMaxRound())]
        for i in regionDiff:
            pred = self.getAutoReg(regionDiff[i])
            ax.plot([labels[-1], labels[-1]+1], [regionDiff[i]
                    [-1], pred[0]],  marker="o", c="black")
            ax.plot(labels, regionDiff[i], label=i, marker="o")
        ax.set_title("plotByTotal: " + target)
        ax.legend(loc="upper left")
        plt.show()

    def plotByRegion(self, target="total", subGroup="region_s"):
        regionSums = self.getGroupedBySub(target=target, subGroup=subGroup)
        fig, ax = plt.subplots()
        labels = [i+1 for i in range(self.getMaxRound())]
        for i in regionSums:
            pred = self.getAutoReg(regionSums[i])
            ax.plot([labels[-1], labels[-1]+1], [regionSums[i]
                    [-1], pred[0]],  marker="o", c="black")
            ax.plot(labels, regionSums[i], label=i, marker="o")
        ax.set_title("plotByRegion: " + target)
        ax.legend(loc="upper left")
        plt.show()

    def getGroupedBySub(self, target="total", subGroup="region_s"):
        (_, l) = self.getRoundData(target=target, subGroup=subGroup)
        sumDict = self.df.groupby([self.groupBy, subGroup])[target].sum()
        regions = self.df[subGroup].unique()
        regionSums = {key: np.zeros(self.getMaxRound()) for key in regions}
        for rIndex in range(len(l)):
            for r in l[rIndex]:
                regionSums[r][rIndex] = sumDict[(rIndex+1, r)]
        return regionSums

    def plotRatio(self, col1="num_contributions", col2="num_unique_contributors", subGroup="region_s"):
        regions = self.df[subGroup].unique()
        col1Group = self.getGroupedBySub(target=col1)
        col2Group = self.getGroupedBySub(target=col2)
        ratioDiff = {key: [col2Group[key][i] / col1Group[key][i] if col1Group[key][i] > 0 else 0
                           for i in range(self.getMaxRound())] for key in regions}
        fig, ax = plt.subplots()
        labels = [i+1 for i in range(self.getMaxRound())]
        for i in ratioDiff:
            pred = self.getAutoReg(ratioDiff[i])
            ax.plot([labels[-1], labels[-1]+1], [ratioDiff[i]
                    [-1], pred[0]],  marker="o", c="black")
            ax.plot(labels, ratioDiff[i], label=i, marker="o")
        ax.set_title("plotRatio: " + col2 + "/" + col1)
        ax.legend(loc="upper left")
        plt.show()

    def plotForRound(self, round="12", target="total"):
        self.df["round_number" == round].sum()

    def getAutoReg(self, data):
        # data = self.getRoundSums()
        model = AutoReg(data, lags=1)
        model_fit = model.fit()
        pred = model_fit.predict(len(data), len(data))
        return pred

    def getExpSmoothing(self):
        print("TODO")

    def plotPrediction(self, target="total"):
        data = self.getRoundSums(target=target)
        pred = self.getAutoReg(data)
        labels = [i+1 for i in range(self.getMaxRound())]
        fig, ax = plt.subplots()
        ax.plot(labels, data, label="Real", marker="o", color="b")
        ax.plot([labels[-1], labels[-1]+1], [data[-1], pred[0]],
                label="Prediction", marker="o", color="r")
        ax.set_title("plotPrediction: " + target)
        ax.legend(loc="upper left")
        plt.show()


def getFloatFromPrice(x):
    if x is NaN:
        return 0
    if not isinstance(x, str):
        x = str(x)
    return float("".join(d for d in x if d.isdigit() or d == '.'))
