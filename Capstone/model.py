import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import math
import utils


################################################################################
# Resnet Class
################################################################################

# Resnet Classes to produce the Feature Pyramid Network
# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py


class IdentityBlock(Layer):
    def __init__(self, kernel_size, filter_sizes, stage, block, use_bias=True):
        """The identity_block is the block that has no conv layer at shortcut
        # Parameters:
          kernel_size: default 3, the kernel size of middle conv layer at main path
          filter_sizes: list of integers, filter sizes of 3 conv layers at main path
          stage: integer, current stage label, used for generating layer names
          block: 'a','b'..., current block label, used for generating layer names
          use_bias: Boolean. To use or not use a bias in conv layers
        """
        super(IdentityBlock, self).__init__()

        self._kernel_size = kernel_size
        self._filter_size_1, self._filter_size_2, self._filter_size_3 = filter_sizes
        self._stage = stage
        self._block = block
        self._conv_name_base = 'res' + str(stage) + block + '_branch'
        self._bn_name_base = 'bn' + str(stage) + block + '_branch'

        self._conv_layer_1 = Conv2D(self._filter_size_1, (1, 1),
                                    name=self._conv_name_base + '2a', use_bias=use_bias)
        self._conv_layer_2 = Conv2D(self._filter_size_2, (kernel_size, kernel_size), padding='same',
                                    name=self._conv_name_base + '2b', use_bias=use_bias)
        self._conv_layer_3 = Conv2D(self._filter_size_3, (1, 1),
                                    name=self._conv_name_base + '2c', use_bias=use_bias)

        self._batch_norm_1 = BatchNormalization(name=self._bn_name_base + '2a')
        self._batch_norm_2 = BatchNormalization(name=self._bn_name_base + '2b')
        self._batch_norm_3 = BatchNormalization(name=self._bn_name_base + '2c')

        self._act_layer_1 = Activation('relu')
        self._act_layer_2 = Activation('relu')
        self._act_layer_out = Activation('relu', name='res' + str(stage) + block + '_out')

    def call(self, x, train_bn=True):
        """
        # Arguments:
          x: the input tensor
          train_bn: boolean, whether to train the batch normalization layer or not
        """
        main = self._conv_layer_1(x)
        main = self._batch_norm_1(main, training=train_bn)
        main = self._act_layer_1(main)
        main = self._conv_layer_2(main)
        main = self._batch_norm_2(main, training=train_bn)
        main = self._act_layer_2(main)
        main = self._conv_layer_3(main)
        main = self._batch_norm_3(main, training=train_bn)
        main = main + x
        main = self._act_layer_out(main)

        return main


class ConvBlock(Layer):
    def __init__(self, kernel_size, filter_sizes, stage, block, strides=(2, 2),
                 use_bias=True):
        """conv_block is the block that has a conv layer at shortcut
        # Parameters
          kernel_size: default 3, the kernel size of middle conv layer at main path
          filters: list of integers, the nb_filters of 3 conv layer at main path
          stage: integer, current stage label, used for generating layer names
          block: 'a','b'..., current block label, used for generating layer names
          use_bias: Boolean. To use or not use a bias in conv layers.
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        super(ConvBlock, self).__init__()
        self._kernel_size = kernel_size
        self._filter_size1, self._filter_size2, self._filter_size3 = filter_sizes
        self._stage = stage
        self._block = block
        self._conv_name_base = 'res' + str(stage) + block + '_branch'
        self._bn_name_base = 'bn' + str(stage) + block + '_branch'

        self._conv_layer_1 = Conv2D(self._filter_size1, (1, 1), strides=strides,
                                    name=self._conv_name_base + '2a', use_bias=use_bias)
        self._conv_layer_2 = Conv2D(self._filter_size2, (kernel_size, kernel_size), padding='same',
                                    name=self._conv_name_base + '2b', use_bias=use_bias)
        self._conv_layer_3 = Conv2D(self._filter_size3, (1, 1),
                                    name=self._conv_name_base + '2c', use_bias=use_bias)
        self._conv_layer_sc = Conv2D(self._filter_size3, (1, 1), strides=strides,
                                     name=self._conv_name_base + '1', use_bias=use_bias)

        self._batch_norm_1 = BatchNormalization(name=self._bn_name_base + '2a')
        self._batch_norm_2 = BatchNormalization(name=self._bn_name_base + '2b')
        self._batch_norm_3 = BatchNormalization(name=self._bn_name_base + '2c')
        self._batch_norm_sc = BatchNormalization(name=self._bn_name_base + '1')

        self._act_layer_1 = Activation('relu')
        self._act_layer_2 = Activation('relu')
        self._act_layer_out = Activation('relu', name='res' + str(stage) + block + '_out')

    def call(self, x, train_bn=True):
        """
        # Arguments:
          x: the input tensor
          train_bn: boolean, whether to train the batch normalization layer or not
        """
        main = self._conv_layer_1(x)
        main = self._batch_norm_1(main, training=train_bn)
        main = self._act_layer_1(main)
        main = self._conv_layer_2(main)
        main = self._batch_norm_2(main, training=train_bn)
        main = self._act_layer_2(main)
        main = self._conv_layer_3(main)
        main = self._batch_norm_3(main, training=train_bn)

        shortcut = self._conv_layer_sc(x)
        shortcut = self._batch_norm_sc(shortcut)

        main = main + shortcut
        main = self._act_layer_out(main)

        return main


class ResNetLayer(Layer):
    def __init__(self, architecture, stage_5=False):
        """Builds a resnet
        # Parameters:
          architecture: Can be resnet50 or resnet101
          stage_5: Boolean. If False, stage 5 of the network is not included
        """
        assert architecture in ['resnet50', 'resnet101']
        super(ResNetLayer, self).__init__()

        self._stage_5 = stage_5

        self._zero_padding_layer = ZeroPadding2D((3, 3))
        self._conv_layer = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)
        self._batch_norm = BatchNormalization(name='bn_conv1')
        self._act_layer = Activation('relu')
        self._max_pool_layer = MaxPool2D((3, 3), strides=(2, 2), padding="same")

        self._conv_block_2a = ConvBlock(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        self._id_block_2b = IdentityBlock(3, [64, 64, 256], stage=2, block='b')
        self._id_block_2c = IdentityBlock(3, [64, 64, 256], stage=2, block='c')

        self._conv_block_3a = ConvBlock(3, [128, 128, 512], stage=3, block='a')
        self._id_block_3b = IdentityBlock(3, [128, 128, 512], stage=3, block='b')
        self._id_block_3c = IdentityBlock(3, [128, 128, 512], stage=3, block='c')
        self._id_block_3d = IdentityBlock(3, [128, 128, 512], stage=3, block='d')

        self._conv_block_4a = ConvBlock(3, [256, 256, 1024], stage=4, block='a')
        self._s4_block_count = {'resnet50': 5, "resnet101": 22}[architecture]
        self._s4_id_blocks = []
        for i in range(self._s4_block_count):
            self._s4_id_blocks.append(IdentityBlock(3, [256, 256, 1024], stage=4, block=chr(98 + i)))

        if stage_5:
            self._conv_block_5a = ConvBlock(3, [512, 512, 2048], stage=5, block='a')
            self._id_block_5b = IdentityBlock(3, [512, 512, 2048], stage=5, block='b')
            self._id_block_5c = IdentityBlock(3, [512, 512, 2048], stage=5, block='c')

    def call(self, x, train_bn=True):
        """
        # Arguments:
          x: the input tensor
          train_bn: boolean, whether to train the batch normalization layer or not
        """
        # Stage 1
        x = self._zero_padding_layer(x)
        x = self._conv_layer(x)
        x = self._batch_norm(x, training=train_bn)
        x = self._act_layer(x)
        C1 = x = self._max_pool_layer(x)
        # Stage 2
        x = self._conv_block_2a(x, train_bn=train_bn)
        x = self._id_block_2b(x, train_bn=train_bn)
        C2 = x = self._id_block_2c(x, train_bn=train_bn)
        # Stage 3
        x = self._conv_block_3a(x, train_bn=train_bn)
        x = self._id_block_3b(x, train_bn=train_bn)
        x = self._id_block_3c(x, train_bn=train_bn)
        C3 = x = self._id_block_3d(x, train_bn=train_bn)
        # Stage 4
        x = self._conv_block_4a(x, train_bn=train_bn)
        for i in range(self._s4_block_count):
            x = self._s4_id_blocks[i](x, train_bn=train_bn)
        C4 = x
        # Stage 5
        if self._stage_5:
            x = self._conv_block_5a(x, train_bn=train_bn)
            x = self._id_block_5b(x, train_bn=train_bn)
            C5 = x = self._id_block_5c(x, train_bn=train_bn)
        else:
            C5 = None

        return [C1, C2, C3, C4, C5]


class RegionProposalNetwork(Layer):
    def __init__(self, anchor_stride, anchors_per_location):
        super(RegionProposalNetwork).__init__()
        self._conv_shared = Conv2D(512, (3, 3), padding='same',
                                   activation='relu', strides=anchor_stride, name='rpn_conv_shared')

        self._conv_class = Conv2D(anchors_per_location * 2, (1, 1), padding='valid',
                                  activation='linear', name='rpn_class_raw')

        self._class_softmax = Activation('softmax', name='rpn_class_xxx')

        self._conv_bbox = Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
                                 activation='linear', name='rpn_bbox_pred')

    def call(self, x):
        """
        # Arguments:
          x: feature_map
        """
        x = self._conv_shared(x)
        # Anchor Score BG/FG. [batch, height, width, anchors per location * 2]
        class_logits = self._conv_class(x)
        # Reshape to [batch, anchors, 2]
        class_logits.set_shape(class_logits.get_shape()[0], -1, 2)
        # Softmax on last dimension of BG/FG
        class_probs = self._class_softmax(class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        bbox = self._conv_bbox(x)

        # Reshape to [batch, anchors, 4]
        bbox.set_shape(bbox.get_shape()[0], -1, 4)

        return [class_logits, class_probs, bbox]


class MaskRCNN(tf.keras.Model):
    """
    Model Class to implement MaskRCNN
    """
    def __init__(self, config, mode='training', rn_arch='resnet50', rn_stage_5=False):
        super(MaskRCNN, self).__init__()

        self._config = config
        self.mode = mode

        self._anchor_cache = {}

        self._resnet_layer = ResNetLayer(rn_arch, rn_stage_5)

        self._conv_layer_fpn_c5p5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')
        self._conv_layer_fpn_c4p4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')
        self._conv_layer_fpn_c3p3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')
        self._conv_layer_fpn_c2p2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')

        self._conv_layer_fpn_p2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), name='fpn_p2')
        self._conv_layer_fpn_p3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), name='fpn_p3')
        self._conv_layer_fpn_p4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), name='fpn_p4')
        self._conv_layer_fpn_p5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), name='fpn_p5')

        self._upsample_layer_fpn_p5 = UpSampling2D(size=(2, 2), name='fpn_p5upsampled')
        self._upsample_layer_fpn_p4 = UpSampling2D(size=(2, 2), name='fpn_p4upsampled')
        self._upsample_layer_fpn_p3 = UpSampling2D(size=(2, 2), name='fpn_p3upsampled')

        self._maxpool_layer = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')

        self._RPN = RegionProposalNetwork(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS))

    def _compute_backbone_shapes(self, image_shape):
        return np.array([[int(math.ceil(image_shape[0] / stride)),
                          int(math.ceil(image_shape[1] / stride))]
                         for stride in self._config.BACKBONE_STRIDES])

    def _get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = self._compute_backbone_shapes(image_shape)
        a = utils.generate_pyramid_anchors(self._config.RPN_ANCHOR_SCALES, self._config.RPN_ANCHOR_RATIOS,
                                           backbone_shapes, self._config.BACKBONE_STRIDES,
                                           self._config.RPN_ANCHOR_STRIDE)
        a = utils.norm_boxes(a, image_shape[:2])
        return a

    def _get_proposals(self, rpn_class_probs, rpn_bbox, anchors):
        """
        # Arguments:
          rpn_class_probs: [batch, num_anchors, (bg prob, fg prob)]
          rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
          anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates
        """

        # Box Scores. Use the foreground class confidence. [Batch, num_rois]
        scores = rpn_class_probs[:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = rpn_bbox * np.reshape(self._config.RPN_BBOX_STD_DEV, [1, 1, 4])

        pre_nms_limit = min(self._config.PRE_NMS_LIMIT, anchors.shape[1])

        # k = pre_nms_limit, ix = [Batch, k]
        ix = tf.math.top_k(scores, pre_nms_limit, sorted=True,
                           name='top_anchors').indices

        # scores.shape = [Batch, k]
        # deltas.shape = pre_nms_anchors.shape = [batch, k, 4]
        scores = tf.gather(scores, ix, axis=1, batch_dims=1)
        deltas = tf.gather(deltas, ix, axis=1, batch_dims=1)
        pre_nms_anchors = tf.gather(anchors, ix, axis=1, batch_dims=1)

        # boxes.shape = [batch, k, 4]
        boxes = utils.apply_box_deltas(pre_nms_anchors, deltas)
        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)  # TODO: Change to bfloat16?
        # boxes = [batch, k, 4]
        boxes = utils.clip_boxes(boxes, window)

        # Extend class dimension for cool tf combined nms function
        # boxes.shape = [batch, k, 1, 4], scores.shape = [batch, k, 1]
        boxes = tf.expand_dims(boxes, axis=2)
        scores = tf.expand_dims(scores, axis=2)
        # This function automatically zero pads to POST_NMS_ROIS_TRAINING boxes/batch
        proposals, _, _ = tf.image.combined_non_max_suppression(boxes, scores,
                                                                self._config.POST_NMS_ROIS_TRAINING,
                                                                self._config.POST_NMS_ROIS_TRAINING,
                                                                self._config.RPN_NMS_THRESHOLD)
        return proposals

    def _detection_targets(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        """Generates detection targets for one image. Subsamples proposals and
        generates target class IDs, bounding box deltas, and masks for each.
        Inputs:
        proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
        masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
               boundaries and resized to neural network output size.
        Note: Returned arrays might be zero padded if not enough target ROIs.
        """
        # Assertions
        asserts = [
            tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                      name="roi_assertion"),
        ]
        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals)

        # Remove zero padding
        proposals, _ = utils.trim_zeros(proposals, name="trim_proposals")
        gt_boxes, non_zeros = utils.trim_zeros(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                       name="trim_gt_class_ids")
        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                             name="trim_gt_masks")

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
        gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = utils.compute_overlaps(proposals, gt_boxes)

        # Compute overlaps with crowd boxes [proposals, crowd_boxes]
        crowd_overlaps = utils.compute_overlaps(proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)

        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(self._config.TRAIN_ROIS_PER_IMAGE *
                             self._config.ROI_POSITIVE_RATIO)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self._config.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
        )
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        # Compute bbox refinement for positive ROIs
        deltas = utils.box_refinement(positive_rois, roi_gt_boxes)
        deltas /= self._config.BBOX_STD_DEV

        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
        # Pick the right mask for each ROI
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        # Compute mask targets
        boxes = positive_rois
        if self._config.USE_MINI_MASK:
            # Transform ROI coordinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)
        box_ids = tf.range(0, tf.shape(roi_masks)[0])
        masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                         box_ids,
                                         self._config.MASK_SHAPE)
        # Remove the extra dimension from masks.
        masks = tf.squeeze(masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = tf.round(masks)

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(self._config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
        masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

        return rois, roi_gt_class_ids, deltas, masks

    def _get_detection_targets(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        """Generates detection targets for the batch. Subsamples proposals and
        generates target class IDs, bounding box deltas, and masks for each.
        Inputs:
          instances = max(instances/MAX_GT_INSTANCES) (per image)
        proposals: [batch, POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        Ragged Tensors:
        gt_class_ids: [batch, (instances)] int class IDs
        gt_boxes: [batch, (instances), (y1, x1, y2, x2)] in normalized coordinates.
        gt_masks: [batch, (instances), height, width] of boolean type.
        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        and masks.
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
        masks: [batch, TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
               boundaries and resized to neural network output size.
        Note: Returned arrays might be zero padded if not enough target ROIs.

        Unable to simplify operations to a batch level, using original implementation
        """
        batch_size = proposals.shape[0]
        names = ["rois", "target_class_ids", "target_deltas", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: self._detection_targets(w, x, y, z),
            batch_size, names=names)
        return outputs

    def _pyramid_roi_align(self, boxes, image_meta, feature_maps):
        """Implements ROI Pooling on multiple levels of the feature pyramid.

        Inputs:
        - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                 coordinates. Possibly padded with zeros if not enough
                 boxes to fill the array.
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - feature_maps: List of feature maps from different levels of the pyramid.
                        Each is [batch, height, width, channels]

        Output:
        Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
        """
        # Assign each ROI to a level in the pyramid based on the ROI area.
        # each tensor = [batch, num_boxes]
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size
        image_shape = utils.parse_image_meta(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32) #TODO: Change to bfloat16?
        roi_level = tf.math.log(tf.sqrt(h*w) / (224 / tf.sqrt(image_area))) / tf.math.log(2)
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32))) #TODO: Change to int16?
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize
            box_indices = tf.cast(ix[:, 0], tf.int32) #TODO: Change to bfloat16?

            #Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices,
                tuple([self._config.POOL_SIZE, self._config.POOL_SIZE]), method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)
        #TODO: Change to int16?

    def call(self, x, image_meta, train_bn=True, gt_class_ids=None, gt_boxes=None,
             gt_masks=None):
        """
        # Arguments:
          x: the image tensors, shape = [batch, height, width, channels]
          train_bn: Boolean. Train the batch normalization layers?
          Only used during training:
          gt_class_ids: ground truth classification of instance classes
            shape = [batch, (instances/image)] Ragged Tensor (RT)
          gt_boxes: ground truth bounding boxes [batch, (instances/image), 4] RT
          gt_masks: ground truth segmentation masks
            shape = [batch, (instances/image), height, width, ?]
        """

        shape = x.get_shape()
        image_shape = shape[1:3]

        _, C2, C3, C4, C5 = self._resnet_layer(x, train_bn)

        P5 = self._conv_layer_fpn_c5p5(C5)
        P4 = self._upsample_layer_fpn_p5(P5) + self._conv_layer_fpn_c4p4(C4)
        P3 = self._upsample_layer_fpn_p4(P4) + self._conv_layer_fpn_c3p3(C3)
        P2 = self._upsample_layer_fpn_p3(P3) + self._conv_layer_fpn_c2p2(C2)

        # Attach 3x3 conv to all P layers to get the final feature maps
        P2 = self._conv_layer_fpn_p2(P2)
        P3 = self._conv_layer_fpn_p3(P3)
        P4 = self._conv_layer_fpn_p4(P4)
        P5 = self._conv_layer_fpn_p5(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = self._maxpool_layer(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        if not tuple(image_shape) in self._anchor_cache:
            self._anchor_cache[tuple(image_shape)] = self._get_anchors(image_shape)
        anchors = self._anchor_cache[tuple(image_shape)]

        # Duplicate across the batch dimension
        anchors = np.broadcast_to(anchors, (self._config.BATCH_SIZE,) + anchors.shape)

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for PX in rpn_feature_maps:
            layer_outputs.append(self._RPN(PX))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class_probs", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [tf.concat(list(o), axis=1) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class_probs, rpn_bbox = outputs

        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]?
        proposals = self._get_proposals(rpn_class_probs, rpn_bbox, anchors)

        if self.mode == 'training':
            training_rois, target_class_ids, target_deltas, target_mask = \
                self._get_detection_targets(proposals, gt_class_ids, gt_boxes, gt_masks)

        mrcnn_class_logits, mrcnn_class, mrcnn_bbox