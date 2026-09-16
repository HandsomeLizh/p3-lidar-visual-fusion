import unittest
import numpy as np
import cv2
from t3_lidar_visual_fusion.image_quality import assess_image


class LightingTests(unittest.TestCase):
    def setUp(self):
        self.texture=np.random.default_rng(38).integers(25,220,(240,320),dtype=np.uint8)

    def test_grayscale_is_valid_input(self):
        self.assertTrue(assess_image(self.texture).valid)

    def test_black_and_white_structure_remains_valid(self):
        yy,xx=np.indices((240,320))
        checker=(((xx//16+yy//16)%2)*255).astype(np.uint8)
        self.assertTrue(assess_image(checker).valid)

    def test_roma_does_not_depend_on_classical_corner_count(self):
        q=assess_image(np.full((240,320),120,np.uint8),texture_required=False)
        self.assertTrue(q.valid)
        self.assertFalse(assess_image(np.zeros((240,320),np.uint8),texture_required=False).valid)

    def test_black_screen_is_rejected(self):
        q=assess_image(np.zeros((240,320),np.uint8))
        self.assertFalse(q.valid);self.assertEqual(q.reason,"underexposed")

    def test_exposure_clipping_is_rejected(self):
        q=assess_image(np.minimum(self.texture.astype(float)*12,255).astype(np.uint8))
        self.assertFalse(q.valid);self.assertEqual(q.reason,"overexposed")

    def test_slant_glare_with_only_small_textured_region_is_rejected(self):
        a=np.full((240,320),255,np.uint8)
        a[:80,:120]=self.texture[:80,:120]
        self.assertFalse(assess_image(a).valid)

    def test_featureless_bright_dark_split_is_rejected(self):
        a=np.zeros((240,320),np.uint8);a[:,160:]=255
        self.assertFalse(assess_image(a).valid)

    def test_low_contrast_image_is_rejected(self):
        a=(120+self.texture//40).astype(np.uint8)
        self.assertFalse(assess_image(a).valid)

    def test_blurred_image_is_rejected(self):
        a=cv2.GaussianBlur(self.texture,(51,51),15)
        self.assertFalse(assess_image(a).valid)


if __name__=="__main__":unittest.main()
