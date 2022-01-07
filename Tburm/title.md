# Gitcoin Grants Analysis

An analysis of Gitcoin Grant data for rounds 1-12. This repository is an entry for [this bounty](https://github.com/gitcoinco/skunkworks/issues/252#issue-1084213288) on Gitcoin. You can view a full version of this analysis at [grants.troyb.xyz](https://grants.troyb.xyz/title.html).

This workbook contains the analysis from two main notebooks. You can view the source code for the data cleaning and all visuals in the following notebooks. All results and findings and recommendations are summarized in this document.

## The Data
First, some notes on the data that is available. There are a few data quality issues that are mentioned in the "Future Improvements" section. Otherwise, this dataset is relatively limited given the lack of temporal data and individual contributions. More complex modeling would be possible given a more thorough dataset, however this data gives an excellent high-level view of the past 12 rounds of funding at Gitcoin.

## Findings

Below you will find a high-level summary of findings. For more details, continue further into this book to view the analysis and data cleaning notebooks.

![](./images/contrib_per_round_match.jpeg?raw=true)
![](./images/contributions_per_round.jpeg?raw=true)

* Contributions in round 12 were significantly larger than the past 3 rounds, setting a new all-time high since Round 8
  * This increase looks like it is due to increases in both contributions and matches
  * Matches were 3.2x larger than Round 11, contributions were 2x greater
* While contribution amounts are up significantly, the number of contributions in Round 12 is only slightly larger previous rounds
  * Number of contributions is consistently increasing between rounds. Even though rounds 9-11 had lower amounts than round 8, there were more individual contributions.
  * This could mean the contributions are larger during this round compared to previous rounds, or that matching amounts are significantly higher.

![](./images/proportion_from_match.jpeg?raw=true)

* The proportion of contributions coming from matches is increasing, reaching a new high in Round 12 since Round 7.
* Round 8 (the previous record for contributions) had the lowest proportion of funds coming from a match


![](./images/contrib_per_category.jpeg?raw=true)
![](./images/grants_per_category.jpeg?raw=true)

* The largest category increases in Round 12 came from "Community" and "dapp tech"
  * Each of these more than doubled between Rounds 11 and 12
* The category with the most steady growth is "NFTs" which makes sense given current trends in the wider market
* Round 12 has a general "Grants Round 12" category which could be throwing off these results

## Future Improvement

The dataset is limited in many ways. To do more complex or interesting analysis, I have a few suggestions:
* Provide a dataset with weekly or daily donations
  * There could be some trends where people wait until the final days or weeks to donate, which we can not determine from this data
* Provide a "contributor" level dataset
  * We have unique contributions in each grant, but can't aggregate that data since contributors could have donated to many grants
  * We could get a better idea of "whale" contributors and how they compare to others
* Fix some data quality issues
  * Region is missing for 44% of the grants. There could be interested data here that we are unable to mine with this dataset.
  * Code all currency values as floats instead of strings including `$` character for easier analysis
  * Fix missing data in currency columns
  * "Grants Round 12" category in Round 12 categories.
