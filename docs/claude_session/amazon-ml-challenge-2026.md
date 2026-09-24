---
name: amazon-ml-challenge-2026
description: "The project in this directory is Swaraj's team entry for the Amazon ML Challenge 2026 entity-resolution hackathon, 25-27 Sep 2026."
metadata:
  node_type: memory
  type: project
  originSessionId: 18514015-4ed8-417e-899a-407f137edb6c
  modified: 2026-09-24T18:57:50.721Z
---

Swaraj is competing in the Amazon ML Challenge 2026 (Unstop) with a team of friends. Round 1 runs 25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST; top 50 announced 2 Oct 2026, grand finale 7 Oct 2026.

**Problem:** Business Entity Resolution. Three noisy TSV sources (S1/S2/S3) of business records (`business_name`, `business_address`, `country`) with no shared identifiers. Source 1 is the deduplicated reference; for each S1 entity, output all matching S2/S3 IDs (zero, one, or many). Training covers US + India; the test set adds France, unseen in training, nothing may be hard-coded to the observed country set.

**Scored on:** F_0.5 (precision-weighted), macro-averaged per Source 1 entity, singletons included, correctly predicting an empty list scores 1.0, any false merge on a singleton scores 0.0.

**Hard constraints:** 5 leaderboard uploads/day (15 total); final model must be MIT/Apache-2.0 and ≤8B params; external data lookup (entity APIs, business registries, geocoding APIs, any internet augmentation) is instant disqualification; only the registered team leader can access the portal, and Swaraj is not the leader.

Deliverables: `matching_results.tsv` (scored) + `candidate_pairs.tsv` (blocking set, audited), a runnable `code/business_entity_resolution/` package, and a filled `Documentation_template.md`.

Team plan and strategy live at https://claude.ai/artifact/EsYm9oNVup8Adbcw3G2ABW. Source documents are in the project root; AWS setup notes in `reference/aws-prep-guide-notes.md`.
