# Attribution and third-party boundaries

RoboScope's project code is distributed under Apache-2.0. Imported dependencies, benchmark assets and data retain their own licenses; they are not relicensed by this repository.

- **LeRobot**: ACT and diffusion model components are imported from the installed package. The project-specific wrappers implement task conditioning, normalization, losses and execution. https://github.com/huggingface/lerobot
- **ACT** (Zhao et al.): action chunking and temporal ensembling research. https://github.com/tonyzhaozh/act
- **Diffusion Policy** (Chi et al.): conditional action diffusion, horizon conventions and CNN/U-Net training design. https://github.com/real-stanford/diffusion_policy
- **LIBERO**: benchmark definitions, demonstrations and initial states. These must be obtained upstream and are excluded from source releases. https://github.com/Lifelong-Robot-Learning/LIBERO
- **robosuite / robomimic / hf-libero**: simulator, learning utilities and packaging used by the experiments; installed as external dependencies.
- **verl-vla**: structural inspiration for separating workflows, runtime and model/environment integrations. No verl-vla source is vendored, and RoboScope is not affiliated with that project. https://github.com/verl-project/verl-vla

The packaged implementation was migrated from this workspace's ACT/DP experiment code. Published CSV records and plots are project-generated experimental outputs. Third-party weights and rendered demonstration assets are not included in the release archive.
