# Pre-outcome runtime verification amendment

2026-09-16, after the plan freeze and before validation or MusicNet predictions.

The local 512-frame backend comparison stopped on its numerical assertion (maximum absolute difference 3.7711e-5). No training, OOM, long-sequence run or new-data evaluation occurred. The original failed directory/logs are retained. A separate bounded audit on that same original training sequence compared both GPU paths with the CPU float32 model. With cuDNN TF32 disabled, the memory-efficient full-attention path agrees with CPU to 3.7253e-8; the native GPU fused fastpath differs by 3.7760e-5. The audit is saved as attention_audit.json and audit_attention.py. The tolerance is not relaxed.

Runtime verification will therefore compare memory-efficient GPU predictions against the original CPU float32 path at the unchanged allclose tolerance (rtol=1e-4, atol=1e-6). Both cuDNN and CUDA matmul TF32 are explicitly disabled for the new validation and transfer inference. Full attention, context, precision, architecture, weights and data remain unchanged. This is a documented runtime difference from the historical GPU predictions, not bitwise historical replay. Every new inference uses this same setting.

The initial remote Start-Process launch ended with SSH before Python entered the runner: no output directory, status, model allocation or predictions were created. Logs remain empty and the process is absent. This infrastructure launch will be replaced by a hidden WMI-created parent outside the SSH process tree, with persistence checked after SSH disconnect. No killed/OOM training is restarted or migrated. New runtime-audit job directories are used; original artifacts are not overwritten.
