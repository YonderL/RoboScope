# Preparing a GitHub release

The source archive is built with an explicit allowlist. It contains the package, recipes, tests, documentation, figures and lightweight result records. Local `outputs/`, old `experiments/`, simulator assets, weights and data are excluded. The builder rejects symlinks, model/data artifacts and personal absolute paths in textual source material, including shell scripts. Local hardware probes and continuation notes are explicitly excluded. Git stores PNG previews; PDF/SVG versions are regenerated with the reporting command. Only completed key evaluations are exported; logs, smoke runs, videos and trajectories stay local.

```bash
python scripts/build_release.py
# Extract the source archive into a NEW directory before initializing Git.
mkdir -p /path/to/release-workspace
tar -xzf dist/roboscope-source.tar.gz -C /path/to/release-workspace
cd /path/to/release-workspace/roboscope
git init
git add .
git commit -m 'Initial release: reproducible ACT and DP experiments on LIBERO-Spatial'
```

Create a repository under your own GitHub account and add its remote before pushing. No remote repository, publication, or upload is performed by the build script. Check the rendered README and Actions results after pushing. Preserve the difference between tested features and roadmap items.

The current source package includes no downloadable policy checkpoint. Others can rebuild figures immediately; reproducing policy rollouts requires training or an independently supplied compatible checkpoint. A future model release should include architecture/config, normalization, split manifest, dependency versions, checkpoint hash and model license.
