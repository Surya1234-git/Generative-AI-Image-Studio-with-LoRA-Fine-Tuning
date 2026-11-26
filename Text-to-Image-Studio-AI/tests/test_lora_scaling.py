"""Regression test: a saved LoRA must behave like the one that was trained, also when alpha differs from rank."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from peft import LoraConfig
    from transformers import CLIPTextConfig, CLIPTextModel
    HAVE = True
except ImportError:          # the project's other tests skip torch tests the same way
    HAVE = False


class _StubTokenizer:
    model_max_length = 77


@unittest.skipUnless(HAVE, "needs torch, diffusers, transformers and peft")
class LoraScalingTest(unittest.TestCase):
    def _unet(self):
        return UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
                                    block_out_channels=(8, 16), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=16,
                                    attention_head_dim=2, norm_num_groups=4)

    def _roundtrip_gap(self, rank, alpha):
        from modules.lora_training import save_lora_adapter
        torch.manual_seed(1)
        base = self._unet()
        trained = copy.deepcopy(base)
        trained.requires_grad_(False)
        cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
                         target_modules=["to_k", "to_q", "to_v", "to_out.0"])
        trained.add_adapter(cfg)
        with torch.no_grad():
            for name, p in trained.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.3)                      # pretend these were trained
        x, t, c = torch.randn(1, 4, 16, 16), torch.tensor([10]), torch.randn(1, 77, 16)
        with torch.no_grad():
            want = trained(x, t, c).sample
        folder = tempfile.mkdtemp()
        save_lora_adapter(StableDiffusionPipeline, folder, trained, cfg)
        vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",),
                            up_block_types=("UpDecoderBlock2D",), block_out_channels=(8,), latent_channels=4,
                            norm_num_groups=4, sample_size=16)
        te = CLIPTextModel(CLIPTextConfig(vocab_size=100, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                          num_attention_heads=2, max_position_embeddings=77,
                                          bos_token_id=1, eos_token_id=2))
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=_StubTokenizer(),
                                       unet=copy.deepcopy(base), scheduler=DDPMScheduler(), safety_checker=None,
                                       feature_extractor=None, requires_safety_checker=False)
        pipe.load_lora_weights(folder)
        with torch.no_grad():
            got = pipe.unet(x, t, c).sample
        return (got - want).abs().max().item()

    def test_default_rank_8_alpha_16_loads_at_trained_strength(self):
        self.assertLess(self._roundtrip_gap(8, 16), 1e-4)

    def test_equal_rank_and_alpha(self):
        self.assertLess(self._roundtrip_gap(4, 4), 1e-4)


if __name__ == "__main__":
    unittest.main()
