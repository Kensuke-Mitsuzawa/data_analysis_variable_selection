date: 2026-09-22


# Problem with Current MMD variable selection

MMD variable selection tends to select variables/features that represents the trivial changes. In the high dimensional space, the detected variable may play a significant role. But when human analysts look at the detected variable with the univariate distribution, it is hard to verify. 

Moreover, picked up variables are often different from the human's intuition. Human intuitively spot on "Gr Liv Area (Continuous): Above grade (ground) living area square feet" or "Lot Area (Continuous): Lot size in square feet". Actually, the simple stats analysis indicates the huge gap between before and after the crash (see the attached markdown document). 

# Requirement toward more practical analysis tool

MMD variable selection may have captured the significant variables in the high dim. space. But, it is hardly connected with the human's intuition. How do we bridge this gap? Please give me your suggestion.


# Self-defense of MMD variable selection - "Disconnect: Mathematical Discrepancy vs. Human Intuition"

Your MMD variable selection algorithm operates in a standardized, high-dimensional space. It searches for the features that provide the cleanest mathematical separation between the two distributions.

-   **Categorical Dominance:** In a dataset heavily populated by dummy-encoded categorical variables (like Ames), a severe shift in a specific category (e.g., the collapse of new construction represented by `SaleType_New`) creates a highly distinct mathematical boundary.

-   **Continuous Variance:** Features like `GrLivArea` (mean shift of -3.03%) or `LotArea` (mean shift of -3.86%) certainly shift post-crash, but they possess massive natural variance. The distributions of these continuous variables heavily overlap between the two eras. While a human sees a "huge gap" in the drop of the median or mean, the algorithm sees two highly overlapping bell curves that are difficult to separate mathematically.

# Possible solution: Cross-Mapping Anchors to Intuitive Metrics

To bridge the gap, you must actively demonstrate how the mathematically selected "Anchor Variables" correlate with, or explicitly cause, the shifts in the continuous variables humans intuitively care about.
**Actionable Steps:**

-   **Conditional Distributions (The "Why"):** Do not just present the univariate distribution of the MMD-selected variables. Instead, plot the human-intuitive variables _conditional_ upon the MMD-selected variables.
    -   _Example:_ Plot the distribution of `GrLivArea` _only_ for houses where `SaleType_New = 1`, and compare it to the distribution of `GrLivArea` where `SaleType_New = 0`.
    -   _The Insight:_ This will likely show that newly constructed homes pre-crash were significantly larger than standard resales. By showing this, you explain to the human analyst: _"The algorithm selected `SaleType_New` because the disappearance of these specific new builds is the direct mechanism that caused the average `GrLivArea` to drop post-crash."_
-   **Bivariate Scatter Analysis:** Create scatter plots linking an MMD anchor to an intuitive metric.
    -   _Example:_ Plot `SalePrice` on the Y-axis and `YearBuilt` on the X-axis. Color the points by the MMD anchor `SaleCondition_Normal` (e.g., Red for Normal, Blue for other).
    -   _The Insight:_ This visualizes how the high-dimensional anchor variable fundamentally alters the relationship between the variables humans naturally track.

----
