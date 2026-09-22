# {{ title_report }}

{{ summary_note }}

| Report Metadata | Description |
| :--- | :--- |
| **Generated At** | {{ report_generation_timestamp }} |
| **Git Commit** | `{{ git_commit_id }}` |
| **Distribution $X$ Label** | {{ label_x_description }} |
| **Distribution $Y$ Label** | {{ label_y_description }} |

---

## Discovered Anchor Variables ($\hat{S}$)

Anchor variables represent the core intrinsic dimensions exhibiting maximum discrepancy between distributions:

{{ table_anchor_variables }}

---

## Comparison of marginal univariate distributions

{{ section_marginal_univariate_distributions }}

---

## Variable Correlation

{{ section_variable_correlation }}

### Constellation Network Graph

Larger nodes are the selected variables $\hat{S}$.

{{ section_constellation_network }}

---

## Cluster Themes & Augmented Feature Sets ($S_\text{tilde}$)

Features correlated with anchor variables are grouped into thematic clusters to provide business/domain interpretability:

{{ table_cluster_themes }}

### Thematic Cluster Tornado Charts

{{ section_tornado_charts }}

---

## Representative Prototype Exemplars

Prototypical samples representing the central density of each distribution in the discrepancy subspace:

{{ table_representative_prototypes }}

### Persona Comparison Radar Charts

Persona radar charts contrast the multi-feature profile of the Top-1 prototypical sample of Distribution $X$ ($x^* \in X$) against the Top-1 prototypical sample of Distribution $Y$ ($y^* \in Y$) across each thematic cluster.

**Mathematical Definition & Value Interpretation**:
For each cluster $k$ and each displayed feature $j \in \{1, \dots, p_k\}$, let $x^*_j$ and $y^*_j$ denote the unscaled feature values of the top prototype of $X$ and $Y$, respectively. The plotted coordinate values $\bar{x}_j, \bar{y}_j \in [0, 1]$ represent relative contrast scaling between the two archetypes:

$$\bar{x}_j = \frac{x^*_j - \min(x^*_j, y^*_j)}{\max(x^*_j, y^*_j) - \min(x^*_j, y^*_j)}, \quad \bar{y}_j = \frac{y^*_j - \min(x^*_j, y^*_j)}{\max(x^*_j, y^*_j) - \min(x^*_j, y^*_j)}$$

*(If $x^*_j = y^*_j$, then $\bar{x}_j = \bar{y}_j = 0.5$).*

- **Value = 1.0**: Indicates that this prototype exhibits the maximum value between the two exemplar samples along feature dimension $j$.
- **Value = 0.0**: Indicates that this prototype exhibits the minimum value between the two exemplar samples along feature dimension $j$.
- The values reflect **individual prototype exemplar realizations** $x^*$ and $y^*$, *not* the marginal distribution means $\mathbb{E}_X[j]$ or $\mathbb{E}_Y[j]$.

{{ section_persona_radar_charts }}

