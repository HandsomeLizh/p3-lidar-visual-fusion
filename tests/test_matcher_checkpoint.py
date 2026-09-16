import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/t3_lidar_visual_fusion'))
from t3_lidar_visual_fusion.learned_matching import canonical_state

class CheckpointTests(unittest.TestCase):
    def test_wrapped_extractor_is_separate_from_matcher(self):
        checkpoint={'extractor.model.net.a':1,'matcher.self_attn.0.Wqkv.weight':2,
                    'matcher.cross_attn.1.to_qk.weight':3,'unexpected_layer':4}
        converted=canonical_state(checkpoint,2)
        self.assertEqual(converted,{'transformers.0.self_attn.Wqkv.weight':2,
            'transformers.1.cross_attn.to_qk.weight':3,'unexpected_layer':4})

    def test_unwrapped_modern_checkpoint_is_preserved(self):
        checkpoint={'transformers.0.self_attn.Wqkv.weight':1,'input_proj.weight':2}
        self.assertEqual(canonical_state(checkpoint,6),checkpoint)
