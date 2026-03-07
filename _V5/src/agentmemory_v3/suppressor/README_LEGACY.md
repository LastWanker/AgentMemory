# Legacy Suppressor Notice

This package remains as a compatibility backend (`SUPPRESSOR_BACKEND=legacy`) during migration.

Current direction is `mem_network_suppressor`:

1. Keep coarse/association retrieval frozen.
2. Keep suppressor as detachable post-processing backend.
3. Avoid adding new feature work into this legacy package unless for bugfix fallback.
