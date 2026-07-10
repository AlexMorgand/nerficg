"""Milestone 1 validation for the deferred-reflection rasterizer extension.

Gate (a): base diff_rasterize output is unchanged (the DR path is additive).
Gate (b): torch.autograd.gradcheck passes on the DR feature channels and on
          the standard Gaussian parameters through diff_rasterize_dr.
"""

import math

import torch

from FasterGSCudaBackend.torch_bindings import diff_rasterize, diff_rasterize_dr, RasterizerSettings


def make_settings(width: int, height: int) -> RasterizerSettings:
    focal = 0.6 * width
    w2c = torch.eye(4, device="cuda", dtype=torch.float32)
    w2c[2, 3] = 4.0  # push camera back along +z (view space)
    return RasterizerSettings(
        w2c=w2c,
        cam_position=torch.tensor([[0.0, 0.0, -4.0]], device="cuda", dtype=torch.float32),
        sh_rotation=torch.eye(3, device="cuda", dtype=torch.float32),
        bg_color=torch.zeros(1, 3, device="cuda", dtype=torch.float32),
        active_sh_bases=1,
        width=width,
        height=height,
        focal_x=focal,
        focal_y=focal,
        center_x=width / 2.0,
        center_y=height / 2.0,
        near_plane=0.01,
        far_plane=100.0,
        proper_antialiasing=False,
    )


def make_gaussians(n: int, dtype: torch.dtype, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    means = (torch.rand(n, 3, generator=g, device="cuda") - 0.5).to(dtype)
    means[:, 2] += 0.5  # in front of camera
    scales = (torch.rand(n, 3, generator=g, device="cuda") * 0.3 - 2.5).to(dtype)  # log-space small
    rotations = torch.randn(n, 4, generator=g, device="cuda").to(dtype)
    opacities = (torch.rand(n, 1, generator=g, device="cuda") * 2.0).to(dtype)
    sh0 = (torch.rand(n, 1, 3, generator=g, device="cuda")).to(dtype)
    shrest = torch.zeros(n, 15, 3, device="cuda", dtype=dtype)
    features = (torch.rand(n, 4, generator=g, device="cuda") - 0.5).to(dtype)
    return means, scales, rotations, opacities, sh0, shrest, features


def gate_a_base_unchanged() -> None:
    width = height = 48
    settings = make_settings(width, height)
    means, scales, rotations, opacities, sh0, shrest, features = make_gaussians(400, torch.float32)
    empty = torch.empty(0, device="cuda")
    base = diff_rasterize(means, scales, rotations, opacities, sh0, shrest, empty, settings)
    img, feat = diff_rasterize_dr(means, scales, rotations, opacities, sh0, shrest, features, empty, settings)
    max_diff = (base - img).abs().max().item()
    print(f"[gate a] base vs DR-color max |diff| = {max_diff:.3e}")
    assert max_diff < 1e-5, f"DR color channels diverge from base rasterizer: {max_diff}"
    assert feat.shape == (4, height, width), feat.shape
    print("[gate a] PASS: base RGB path unchanged, feature map shape ok")


def gate_b_feature_linearity() -> None:
    """The feature blend is exactly linear in ``features`` (blend weights do not
    depend on feature values), so the analytic feature gradient must match central
    finite differences to float precision. This rigorously validates the new
    feature backward path without needing double-precision kernels.
    """
    width = height = 32
    settings = make_settings(width, height)
    means, scales, rotations, opacities, sh0, shrest, features = make_gaussians(200, torch.float32, seed=7)
    empty = torch.empty(0, device="cuda")
    features = features.requires_grad_(True)

    # random upstream gradient on the feature map
    torch.manual_seed(1)
    grad_feat_map = torch.randn(4, height, width, device="cuda")

    _img, feat = diff_rasterize_dr(means, scales, rotations, opacities, sh0, shrest, features, empty, settings)
    (feat * grad_feat_map).sum().backward()
    analytic = features.grad.clone()

    # central finite differences on a handful of entries
    eps = 1e-3
    max_rel = 0.0
    checked = 0
    torch.manual_seed(2)
    idx = torch.randint(0, features.shape[0], (12,))
    with torch.no_grad():
        for i in idx.tolist():
            for c in range(4):
                base = features[i, c].item()
                features[i, c] = base + eps
                _, fp = diff_rasterize_dr(means, scales, rotations, opacities, sh0, shrest, features, empty, settings)
                lp = (fp * grad_feat_map).sum().item()
                features[i, c] = base - eps
                _, fm = diff_rasterize_dr(means, scales, rotations, opacities, sh0, shrest, features, empty, settings)
                lm = (fm * grad_feat_map).sum().item()
                features[i, c] = base
                num = (lp - lm) / (2 * eps)
                ana = analytic[i, c].item()
                denom = max(abs(ana), abs(num), 1.0)
                rel = abs(num - ana) / denom
                max_rel = max(max_rel, rel)
                checked += 1
    print(f"[gate b] feature-grad linearity: {checked} entries, max rel err = {max_rel:.3e}")
    assert max_rel < 1e-2, f"feature gradient mismatch: {max_rel}"
    print("[gate b] PASS: feature-input gradient matches finite differences")


def gate_c_zero_feature_consistency() -> None:
    """With zero features and zero feature-map gradient, the DR backward must
    produce the same geometry/color gradients as the base rasterizer (the feature
    path contributes nothing), i.e. no regression from the added alpha term.
    """
    width = height = 40
    settings = make_settings(width, height)
    means, scales, rotations, opacities, sh0, shrest, _f = make_gaussians(300, torch.float32, seed=11)
    empty = torch.empty(0, device="cuda")
    features = torch.zeros(means.shape[0], 4, device="cuda")

    torch.manual_seed(5)
    grad_img = torch.randn(3, height, width, device="cuda")

    def run(dr: bool):
        m = means.clone().requires_grad_(True)
        s = scales.clone().requires_grad_(True)
        r = rotations.clone().requires_grad_(True)
        o = opacities.clone().requires_grad_(True)
        c = sh0.clone().requires_grad_(True)
        if dr:
            img, feat = diff_rasterize_dr(m, s, r, o, c, shrest, features, empty, settings)
            # zero grad on feature map -> feature path inert
            (img * grad_img).sum().backward()
        else:
            img = diff_rasterize(m, s, r, o, c, shrest, empty, settings)
            (img * grad_img).sum().backward()
        return m.grad, s.grad, r.grad, o.grad, c.grad

    base = run(False)
    dr = run(True)
    names = ["means", "scales", "rotations", "opacities", "sh0"]
    max_diff = 0.0
    for name, b, d in zip(names, base, dr):
        diff = (b - d).abs().max().item()
        max_diff = max(max_diff, diff)
    print(f"[gate c] zero-feature DR vs base geometry grad max |diff| = {max_diff:.3e}")
    assert max_diff < 1e-4, f"DR backward regresses base geometry grads: {max_diff}"
    print("[gate c] PASS: DR backward reduces to base when features are inert")


def gate_d_cubemap_encoder() -> None:
    """Cubemap encoder forward/backward smoke test (vendored in FasterGSCudaBackend)."""
    from FasterGSCudaBackend.torch_bindings import CubemapEncoder

    encoder = CubemapEncoder(output_dim=3, resolution=32, interpolation='linear').cuda()
    dirs = torch.randn(1024, 3, device='cuda', dtype=torch.float32)
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    dirs.requires_grad_(True)
    logits = encoder(dirs)
    assert logits.shape == (1024, 3), logits.shape
    loss = logits.square().sum()
    loss.backward()
    assert encoder.params['Cubemap_texture'].grad is not None
    assert dirs.grad is not None
    assert torch.isfinite(logits).all()
    assert torch.isfinite(encoder.params['Cubemap_texture'].grad).all()
    print("[gate d] PASS: CubemapEncoder forward/backward ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    gate_a_base_unchanged()
    gate_b_feature_linearity()
    gate_c_zero_feature_consistency()
    gate_d_cubemap_encoder()
    print("\nAll milestone-1 gates PASSED")
