from detectron2.structures.instances import *
from detectron2.structures.boxes import *

import unittest
import data_utils


class TestGetRandomWord(unittest.TestCase):
    def testEmptyVocab(self):
        with self.assertRaises(ValueError):
            data_utils.get_random_word([])

    def testOneVocab(self):
        vocab = ["One"]
        self.assertEqual("One", data_utils.get_random_word(vocab))

    def testTwoVocab(self):
        vocab = ["One", "Two"]
        self.assertTrue(data_utils.get_random_word(vocab) in vocab)

    def testSmallVocab(self):
        vocab = ["This", "is", "my", "vocab", "list"]
        self.assertTrue(data_utils.get_random_word(vocab) in vocab)


class TestGetImgTensors(unittest.TestCase):
    def ParameterizeTest(self, test_detections, test_max_detections, test_height, test_width, test_classes,
                         test_fclayers, test_fcdim, test_used_fclayer):
        x1 = torch.rand((test_detections, 1)) * test_width
        y1 = torch.rand((test_detections, 1)) * test_height
        x2 = torch.rand((test_detections, 1)) * test_width
        y2 = torch.rand((test_detections, 1)) * test_height
        test_boxes = Boxes(torch.cat([x1, y1, x2, y2], dim=1))
        # Nx4 (x1, y1, x2, y2) range = image height/width

        test_pred_classes = torch.randint(low=0, high=test_classes, size=[test_detections])
        # shape = [100], in range [0, 1600] number of object classes
        test_box_features = torch.rand((test_detections, test_fclayers * test_fcdim)) * 10
        # shape = [100, 4096] (2048 per layer)
        test_probs = torch.rand((test_detections, test_classes + 1))
        test_probs = test_probs / torch.sum(test_probs, dim=1, keepdim=True)

        test_scores = torch.max(test_probs[:, :-1], dim=1)[0]  # shape = [100], in range [0,1]

        test_pred = {}
        test_instances = Instances((test_height, test_width))
        test_instances.set("pred_boxes", test_boxes)
        test_instances.set("scores", test_scores)
        test_instances.set("pred_classes", test_pred_classes)
        test_instances.set("fc_box_features", test_box_features)
        test_instances.set("probs", test_probs)
        test_pred['instances'] = test_instances

        box_features, vis_pe = data_utils.get_img_tensors(test_pred, fc_layer=test_used_fclayer, fc_dim=test_fcdim,
                                                          num_classes=test_classes, max_detections=test_max_detections)

        self.assertEqual((test_max_detections, test_fcdim), box_features.shape)
        self.assertEqual((test_max_detections, 7 + test_classes), vis_pe.shape)

    def testNormalCase(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=480, test_width=640,
                              test_classes=1600, test_fclayers=2, test_fcdim=2048, test_used_fclayer=0)

    def testOtherLayer(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=480, test_width=640,
                              test_classes=1600, test_fclayers=2, test_fcdim=2048, test_used_fclayer=1)

    def testLessDetections(self):
        self.ParameterizeTest(test_detections=73, test_max_detections=100, test_height=480, test_width=640,
                              test_classes=1600, test_fclayers=2, test_fcdim=2048, test_used_fclayer=0)

    def testLessMaxDetections(self):
        self.ParameterizeTest(test_detections=73, test_max_detections=85, test_height=480, test_width=640,
                              test_classes=1600, test_fclayers=2, test_fcdim=2048, test_used_fclayer=0)

    def testMoreDetectionsThanMax(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=85, test_height=480, test_width=640,
                              test_classes=1600, test_fclayers=2, test_fcdim=2048, test_used_fclayer=0)

    def testOtherSizes(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=640, test_width=480,
                              test_classes=1600, test_fclayers=2, test_fcdim=2048, test_used_fclayer=0)

    def testOtherSizes2(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=580, test_width=817,
                              test_classes=1600, test_fclayers=2, test_fcdim=2048, test_used_fclayer=0)

    def testOtherSizes3(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=57, test_width=712,
                              test_classes=1600, test_fclayers=2, test_fcdim=2048, test_used_fclayer=0)

    def testLessClasses(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=480, test_width=640,
                              test_classes=800, test_fclayers=2, test_fcdim=2048, test_used_fclayer=0)

    def testMoreClasses(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=480, test_width=640,
                              test_classes=3200, test_fclayers=2, test_fcdim=2048, test_used_fclayer=0)

    def testLessLayers(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=480, test_width=640,
                              test_classes=3200, test_fclayers=1, test_fcdim=2048, test_used_fclayer=0)

    def testMoreLayers(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=480, test_width=640,
                              test_classes=3200, test_fclayers=4, test_fcdim=2048, test_used_fclayer=3)

    def testLessFCDims(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=480, test_width=640,
                              test_classes=3200, test_fclayers=2, test_fcdim=1024, test_used_fclayer=0)

    def testMoreFCDims(self):
        self.ParameterizeTest(test_detections=100, test_max_detections=100, test_height=480, test_width=640,
                              test_classes=3200, test_fclayers=2, test_fcdim=4096, test_used_fclayer=0)


class TestPrepVisPE(unittest.TestCase):
    def ParameterizeTest(self, test_batch_size, test_detections, test_classes):
        bbox_preds=torch.rand((test_batch_size, test_detections, 6))
        cls_probs=torch.rand((test_batch_size, test_detections, test_classes+1))

        vis_pe = data_utils.prep_vis_pe(bbox_preds=bbox_preds, cls_probs=cls_probs)

        self.assertEqual((test_batch_size, test_detections, test_classes+7), vis_pe.shape)

    def testNormalCase(self):
        self.ParameterizeTest(test_batch_size=32, test_detections=100, test_classes=1600)

    def testLessBatch(self):
        self.ParameterizeTest(test_batch_size=16, test_detections=100, test_classes=1600)

    def testMoreBatch(self):
        self.ParameterizeTest(test_batch_size=64, test_detections=100, test_classes=1600)

    def testLessDetections(self):
        self.ParameterizeTest(test_batch_size=32, test_detections=50, test_classes=1600)

    def testMoreDetections(self):
        self.ParameterizeTest(test_batch_size=32, test_detections=150, test_classes=1600)

    def testLessClasses(self):
        self.ParameterizeTest(test_batch_size=32, test_detections=100, test_classes=800)

    def testMoreClasses(self):
        self.ParameterizeTest(test_batch_size=32, test_detections=100, test_classes=3200)


if __name__ == '__main__':
    unittest.main()
