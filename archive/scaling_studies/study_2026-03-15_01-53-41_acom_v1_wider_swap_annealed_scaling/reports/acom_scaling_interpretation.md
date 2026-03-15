# ACOM Scaling Interpretation

## Overall behavior

The tuned ACOM configuration (`acom_v1_wider_swap_annealed`) remained stable across all completed dataset sizes in this study. Every completed run improved from its initial random placement, which indicates that the optimization procedure continued to function reliably as document count increased.

## Metric trends

Trustworthiness was highest at dataset size 100 with a value of 0.780. The weakest trustworthiness was observed at dataset size 200 with a value of 0.705. Neighborhood preservation and stress should be interpreted together with runtime because the discrete grid becomes progressively less flexible as the number of occupied cells increases.

## Optimization stability

The optimization remained numerically stable across the completed sizes. Initial cost, final cost, and cost improvement all stayed well defined, and no alignment or grid-capacity failures occurred in the successful runs. Runtime increased with dataset size as expected: [{'dataset_size': 50, 'runtime_seconds': 13.3686}, {'dataset_size': 100, 'runtime_seconds': 15.1116}, {'dataset_size': 150, 'runtime_seconds': 39.2241}, {'dataset_size': 200, 'runtime_seconds': 35.5981}].

## Practical limitations

The main limitation is computational. As document count increases, the swap search and cost evaluation become more expensive, and the optimization trajectory becomes harder to improve quickly. The grid-based representation also remains more constrained than a continuous layout, so preserving local neighborhoods at larger scales is still challenging.

## Semantic readability at larger scales

The discrete map remains interpretable at larger sizes because it preserves explicit grid occupancy and category-colored structure. However, as the number of documents grows, more careful objective design and possibly more targeted proposal mechanisms may be needed to maintain strong semantic separation without substantial runtime growth.
