# ACOM Experimental Findings

## Best-performing ACOM variant

The strongest ACOM configuration in the completed sweep was **acom_v1_wider_swap_annealed**. This variant combined a wider local swap search with annealed acceptance, and it produced the best ACOM result on all tracked internal criteria in the current batch: final cost, cost improvement, neighborhood preservation, trustworthiness, and stress.

## Improvement over the baseline ACOM version

Compared with the baseline ACOM configuration (`acom_v1_baseline`), the tuned variant improved substantially. Cost improvement increased from 25.063 to 87.815. Neighborhood preservation increased from 0.134 to 0.367, and trustworthiness increased from 0.567 to 0.787. This indicates that the tuned variant did not only optimize its internal objective more strongly; it also improved the external embedding-space preservation metrics that matter for comparison.

## Conceptual comparison with continuous baselines

The tuned ACOM layout remains a discrete grid-based mapping, so it solves a different visualization problem from PCA, t-SNE, and UMAP. In the tuned comparison run, ACOM reached neighborhood preservation of 0.367 and trustworthiness of 0.787. This exceeded PCA on both neighborhood preservation (0.329) and trustworthiness (0.758), but t-SNE remained stronger at preserving local structure with neighborhood preservation 0.523 and trustworthiness 0.893. UMAP still delivered stronger neighborhood preservation (0.505) and trustworthiness (0.897) than the tuned ACOM layout.

## Remaining limitations

The main limitation is that ACOM still trades off some structural faithfulness in order to satisfy the discrete grid constraint. Although the tuned variant improved stress to 5.107, PCA still achieved much lower stress in the same embedding space, which shows that continuous methods remain better at preserving global pairwise geometry. In addition, the tuned ACOM result is still sensitive to optimization design choices such as search breadth and acceptance behavior, so future work should continue to test objective refinements and more targeted swap proposals rather than only increasing computation.
