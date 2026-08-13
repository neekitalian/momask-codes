"""
ZoneBlendPipeline — orchestrates all five Phase 2 steps.

Usage
-----
    from semantic_spectrum.pipeline import ZoneBlendPipeline

    pipeline = ZoneBlendPipeline(
        vq_model=vq_model,
        mask_transformer=t2m_transformer,
        res_model=res_model,
        vq_opt=vq_opt,
        mean=mean,
        std=std,
        alpha=0.5,
        zone_mode='standard',
    )
    output_joints = pipeline.run(original_joints, text_prompt='[walk:0.91] A person walks.')

Steps
-----
    1. original_joints arrive from MediaPipe (caller's responsibility)
    2. extract original movement feature vectors per zone
    3. MoMask edit: generate new joints conditioned on prompt + source
    4. extract MoMask movement feature vectors per zone
    5. feature-space blend + autoregressive reconstruction
"""

from __future__ import annotations

import numpy as np
import torch

from semantic_spectrum.zones import ZoneConfig
from semantic_spectrum.zone_features import ZoneFeatureExtractor
from semantic_spectrum.blend import FeatureBlender


class ZoneBlendPipeline:
    """
    Orchestrates the full Phase 2 zone-aware blending pipeline.

    Parameters
    ----------
    vq_model         : loaded RVQVAE model (eval mode)
    mask_transformer : loaded MaskTransformer (eval mode)
    res_model        : loaded ResidualTransformer (eval mode)
    vq_opt           : option namespace from vq checkpoint
    mean             : (263,) normalisation mean  (from meta/mean.npy)
    std              : (263,) normalisation std   (from meta/std.npy)
    alpha            : blending strength in [0, 1]
    zone_mode        : 'standard' or 'side_specific'
    device           : torch device (default: cuda:0 if available)
    time_steps       : MaskTransformer edit timesteps
    cond_scale       : classifier-free guidance scale
    temperature      : sampling temperature
    topkr            : top-k filter threshold
    """

    def __init__(
        self,
        vq_model,
        mask_transformer,
        res_model,
        vq_opt,
        mean:         np.ndarray,
        std:          np.ndarray,
        alpha:        float = 0.5,
        zone_mode:    str   = "standard",
        device:       torch.device | None = None,
        time_steps:   int   = 18,
        cond_scale:   float = 4.0,
        temperature:  float = 1.0,
        topkr:        float = 0.9,
    ):
        self.vq_model         = vq_model
        self.mask_transformer = mask_transformer
        self.res_model        = res_model
        self.vq_opt           = vq_opt
        self.mean             = mean.astype(np.float32)
        self.std              = std.astype(np.float32)
        self.device           = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # sampling hyper-parameters
        self.time_steps  = time_steps
        self.cond_scale  = cond_scale
        self.temperature = temperature
        self.topkr       = topkr

        # Phase 2 components
        self.zone_config = ZoneConfig(zone_mode)
        self.extractor   = ZoneFeatureExtractor(self.zone_config)
        self.blender     = FeatureBlender(alpha=alpha, zone_mode=zone_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        original_joints:    np.ndarray,
        text_prompt:        str,
        return_intermediates: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the full five-step pipeline.

        Parameters
        ----------
        original_joints      : (T, 22, 3)  from MediaPipe (Step 1 — caller provides)
        text_prompt          : spectrum-conditioned caption
        return_intermediates : if True, return (original_joints, momask_joints, output_joints)

        Returns
        -------
        output_joints : (T, 22, 3)  — or tuple of three if return_intermediates=True
        """
        M_original    = self.extractor.extract(original_joints)
        momask_joints = self._momask_edit(original_joints, text_prompt)
        M_momask      = self.extractor.extract(momask_joints)
        M_output      = self.blender.feature_blend(M_original, M_momask)
        output_joints = self.blender.reconstruct(M_output, original_joints)

        if return_intermediates:
            return original_joints, momask_joints, output_joints, M_output
        return output_joints

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _momask_edit(
        self,
        original_joints: np.ndarray,
        text_prompt:     str,
    ) -> np.ndarray:
        """
        Use MoMask to generate new joints conditioned on text_prompt,
        with original_joints as the source motion.

        Returns (T, 22, 3) in the same coordinate frame as original_joints.
        """
        from utils.motion_process import recover_from_ric

        T          = len(original_joints)
        max_frames = 196

        # Convert (T, 22, 3) → (T, 263) via recover_from_ric inverse is not
        # straightforward; we load from pre-computed npz when available.
        # Here we accept a (T, 263) motion vec as the canonical path and
        # derive 22-joint output from the MoMask decode.
        #
        # When original_joints is passed directly, we encode it as-is
        # through the VQ model after normalising with training statistics.
        # Note: original_joints are in (T,22,3); we need (T,263) for the VQ.
        # This conversion assumes original_joints came from recover_from_ric
        # and we re-encode via the forward path.  For full accuracy, callers
        # should pass a (T,263) motion vector instead (see run_zone_blend.py).
        motion = self._joints_to_motion_vec(original_joints)  # (T, 263)

        motion_norm = (motion - self.mean) / self.std
        if max_frames > T:
            pad = np.zeros((max_frames - T, motion_norm.shape[1]), dtype=np.float32)
            motion_norm = np.concatenate([motion_norm, pad], axis=0)

        motion_t = torch.from_numpy(motion_norm)[None].float().to(self.device)

        token_len = torch.div(
            torch.LongTensor([T]), 4, rounding_mode='floor'
        ).to(self.device)
        m_length  = token_len * 4

        with torch.no_grad():
            tokens, _ = self.vq_model.encode(motion_t)

            # full-sequence edit mask
            edit_mask = torch.ones_like(tokens[..., 0]).bool()

            mids = self.mask_transformer.edit(
                [text_prompt], tokens[..., 0].clone(), token_len,
                timesteps=self.time_steps,
                cond_scale=self.cond_scale,
                temperature=self.temperature,
                topk_filter_thres=self.topkr,
                gsample=False,
                force_mask=False,
                edit_mask=edit_mask,
            )
            mids       = self.res_model.generate(
                mids, [text_prompt], token_len, temperature=1, cond_scale=2
            )
            pred       = self.vq_model.forward_decoder(mids).detach().cpu().numpy()

        pred_motion = pred[0] * self.std + self.mean   # (max_frames, 263)
        pred_motion = pred_motion[:int(m_length[0])]    # trim to original length

        # Decode (T, 263) → (T, 22, 3)
        pred_joints = recover_from_ric(
            torch.from_numpy(pred_motion).float(), 22
        ).numpy()

        return pred_joints   # (T, 22, 3)

    def _joints_to_motion_vec(self, joints: np.ndarray) -> np.ndarray:
        """
        Approximate (T, 22, 3) → (T, 263) conversion.

        This is a best-effort path for when only raw joint positions are
        available.  For highest fidelity, use motions that were originally
        stored as 263-dim vectors (e.g. from new_joint_vecs/).

        The conversion flattens joint positions and zero-pads to 263 dims.
        Callers with access to the original 263-dim vector should bypass
        this method entirely by passing motion_vec to run_from_motion_vec().
        """
        T = len(joints)
        flat = joints.reshape(T, -1)                   # (T, 66)
        motion = np.zeros((T, 263), dtype=np.float32)
        motion[:, :flat.shape[1]] = flat
        return motion

    # ------------------------------------------------------------------
    # Alternative entry point when 263-dim motion is already available
    # ------------------------------------------------------------------

    def run_from_motion_vec(
        self,
        motion_vec:           np.ndarray,
        original_joints:      np.ndarray,
        text_prompt:          str,
        return_intermediates: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Same as run() but accepts a pre-computed (T, 263) motion vector
        for the MoMask encode step, giving highest-quality results.

        Parameters
        ----------
        motion_vec           : (T, 263)  HumanML3D normalised motion vector
        original_joints      : (T, 22, 3)  used only for feature extraction seed
        text_prompt          : spectrum caption string
        return_intermediates : if True, return (original_joints, momask_joints, output_joints)

        Returns
        -------
        output_joints : (T, 22, 3)  — or tuple of three if return_intermediates=True
        """
        from utils.motion_process import recover_from_ric

        T          = len(motion_vec)
        max_frames = 196

        # Step 2
        M_original = self.extractor.extract(original_joints)

        # Step 3 (high-quality path)
        motion_norm = (motion_vec.astype(np.float32) - self.mean) / self.std
        if max_frames > T:
            pad = np.zeros((max_frames - T, 263), dtype=np.float32)
            motion_norm = np.concatenate([motion_norm, pad], axis=0)

        motion_t  = torch.from_numpy(motion_norm)[None].float().to(self.device)
        token_len = torch.div(torch.LongTensor([T]), 4,
                              rounding_mode='floor').to(self.device)
        m_length  = token_len * 4

        with torch.no_grad():
            tokens, _  = self.vq_model.encode(motion_t)
            edit_mask  = torch.ones_like(tokens[..., 0]).bool()
            mids       = self.mask_transformer.edit(
                [text_prompt], tokens[..., 0].clone(), token_len,
                timesteps=self.time_steps, cond_scale=self.cond_scale,
                temperature=self.temperature, topk_filter_thres=self.topkr,
                gsample=False, force_mask=False, edit_mask=edit_mask,
            )
            mids       = self.res_model.generate(
                mids, [text_prompt], token_len, temperature=1, cond_scale=2
            )
            pred       = self.vq_model.forward_decoder(mids).detach().cpu().numpy()

        pred_motion = pred[0] * self.std + self.mean
        pred_motion = pred_motion[:int(m_length[0])]
        momask_joints = recover_from_ric(
            torch.from_numpy(pred_motion).float(), 22
        ).numpy()

        # Steps 4 + 5
        M_momask      = self.extractor.extract(momask_joints)
        M_output      = self.blender.feature_blend(M_original, M_momask)
        output_joints = self.blender.reconstruct(M_output, original_joints)

        if return_intermediates:
            return original_joints, momask_joints, output_joints, M_output
        return output_joints
