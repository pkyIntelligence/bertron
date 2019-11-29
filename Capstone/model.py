import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import math
import utils


# Config Globals for now...
TOP_DOWN_PYRAMID_SIZE = 256
BACKBONE_STRIDES = [4, 8, 16, 32, 64]
RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
RPN_ANCHOR_RATIOS = [0.5, 1, 2]
RPN_ANCHOR_STRIDE = 1
BATCH_SIZE = 2  # Hardcoded for now should be at most IMAGES_PER_GPU * GPU's (How does this change for TPUs?)
POST_NMS_ROIS_TRAINING = 2000
RPN_NMS_THRESHOLD = 0.7
RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
# ROIs kept after tf.nn.top_k and before non-maximum suppression
PRE_NMS_LIMIT = 6000
MAX_GT_INSTANCES = 100
TRAIN_ROIS_PER_IMAGE = 200
ROI_POSITIVE_RATIO = 0.33
RPN_ANCHOR_STRIDE = 1

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

    def __init__(self, mode='training', rn_arch='resnet50', rn_stage_5=False):
        super(MaskRCNN, self).__init__()

        self.mode = mode

        self._anchor_cache = {}

        self._resnet_layer = ResNetLayer(rn_arch, rn_stage_5)

        self._conv_layer_fpn_c5p5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')
        self._conv_layer_fpn_c4p4 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')
        self._conv_layer_fpn_c3p3 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')
        self._conv_layer_fpn_c2p2 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')

        self._conv_layer_fpn_p2 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), name='fpn_p2')
        self._conv_layer_fpn_p3 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), name='fpn_p3')
        self._conv_layer_fpn_p4 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), name='fpn_p4')
        self._conv_layer_fpn_p5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), name='fpn_p5')

        self._upsample_layer_fpn_p5 = UpSampling2D(size=(2, 2), name='fpn_p5upsampled')
        self._upsample_layer_fpn_p4 = UpSampling2D(size=(2, 2), name='fpn_p4upsampled')
        self._upsample_layer_fpn_p3 = UpSampling2D(size=(2, 2), name='fpn_p3upsampled')

        self._maxpool_layer = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')

        self._RPN = RegionProposalNetwork(RPN_ANCHOR_STRIDE, len(RPN_ANCHOR_RATIOS))

    def _compute_backbone_shapes(self, image_shape):
        return np.array([[int(math.ceil(image_shape[0] / stride)),
                          int(math.ceil(image_shape[1] / stride))]
                         for stride in BACKBONE_STRIDES])

    def _generate_anchors(self, scales, ratios, shape, feature_stride,
                          anchor_stride):
        """
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
        anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes

    def _generate_pyramid_anchors(self, scales, ratios, feature_shapes,
                                  feature_strides, anchor_stride):
        """Generate anchors at different levels of a feature pyramid. Each scale
        is associated with a level of the pyramid, but each ratio is used in
        all levels of the pyramid.
        Returns:
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
            with the same order of the given scales. So, anchors of scale[0] come
            first, then anchors of scale[1], and so on.
        """
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = []
        for i in range(len(scales)):
            anchors.append(self._generate_anchors(scales[i], ratios, feature_shapes[i],
                                                  feature_strides[i], anchor_stride))
        return np.concatenate(anchors, axis=0)

    def _norm_boxes(self, boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels
        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.
        Returns:
            [N, (y1, x1, y2, x2)] in normalized coordinates
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)

    def _get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = self._compute_backbone_shapes(image_shape)
        a = self._generate_pyramid_anchors(RPN_ANCHOR_SCALES, RPN_ANCHOR_RATIOS,
                                           backbone_shapes, BACKBONE_STRIDES,
                                           RPN_ANCHOR_STRIDE)
        a = self._norm_boxes(a, image_shape[:2])
        return a

    def _apply_box_deltas(boxes, deltas):
        """
        # Arguments:
          boxes: [batch, k, (y1, x1, y2, x2)]. boxes to update
          deltas: [batch, k, (dy, dx, log(dh), log(dw))] refinements to apply
        """

        # TODO: Convert to bfloat16 for TPU's???
        # boxes = boxes.astype(np.float32)
        # Convert to y, x, h, w
        height = boxes[:, :, 2] - boxes[:, :, 0]
        width = boxes[:, :, 3] - boxes[:, :, 1]
        center_y = boxes[:, :, 0] + 0.5 * height
        center_x = boxes[:, :, 1] + 0.5 * width
        # Apply deltas
        center_y += deltas[:, :, 0] * height
        center_x += deltas[:, :, 1] * width
        height *= tf.exp(deltas[:, :, 2])
        width *= tf.exp(deltas[:, :, 3])
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        result = tf.stack([y1, x1, y2, x2], axis=2, name="apply_box_deltas_out")
        return result

    def _clip_boxes(boxes, window):
        """
        # Arguments:
          boxes: [batch, k, (y1, x1, y2, x2)]
          window: (y1, x1, y2, x2)
        """
        # Split
        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        # Clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        clipped = tf.concat([y1, x1, y2, x2], axis=2, name='clipped_boxes')
        clipped.set_shape((clipped.shape[0], clipped.shape[1], 4))
        return clipped

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
        deltas = rpn_bbox * np.reshape(RPN_BBOX_STD_DEV, [1, 1, 4])

        pre_nms_limit = min(PRE_NMS_LIMIT, anchors.shape[1])

        # k = pre_nms_limit, ix = [Batch, k]
        ix = tf.math.top_k(scores, pre_nms_limit, sorted=True,
                           name='top_anchors').indices

        # scores.shape = [Batch, k]
        # deltas.shape = pre_nms_anchors.shape = [batch, k, 4]
        scores = tf.gather(scores, ix, axis=1, batch_dims=1)
        deltas = tf.gather(deltas, ix, axis=1, batch_dims=1)
        pre_nms_anchors = tf.gather(anchors, ix, axis=1, batch_dims=1)

        # boxes.shape = [batch, k, 4]
        boxes = self._apply_box_deltas(pre_nms_anchors, deltas)
        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)  # TODO: Change to bfloat16?
        # boxes = [batch, k, 4]
        boxes = self._clip_boxes(boxes, window)

        # Extend class dimension for cool tf combined nms function
        # boxes.shape = [batch, k, 1, 4], scores.shape = [batch, k, 1]
        boxes = tf.expand_dims(boxes, axis=2)
        scores = tf.expand_dims(scores, axis=2)
        # This function automatically zero pads to POST_NMS_ROIS_TRAINING boxes/batch
        proposals, _, _ = tf.image.combined_non_max_suppression(boxes, scores,
                                                                POST_NMS_ROIS_TRAINING, POST_NMS_ROIS_TRAINING,
                                                                RPN_NMS_THRESHOLD)

        return proposals

    def _get_detection_targets(proposals, gt_class_ids, gt_boxes, gt_masks):
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
        """
        # Remove Zero Padding
        proposals, _ = utils.trim_zeros(proposals, name='trim_proposals')
        gt_boxes, non_zeros = utils.trim_zeros(gt_boxes, name='trim_gt_boxes')
        gt_class_ids = tf.ragged.boolean_mask(gt_class_ids, non_zeros,
                                              name='trim_gt_class_ids')
        gt_masks = tf.ragged.boolean_mask(gt_masks, non_zeros,
                                          name='trim_gt_masks')

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_mask = (gt_class_ids < 0)
        non_crowd_mask = (gt_class_ids > 0)
        crowd_boxes = tf.ragged.boolean_mask(gt_boxes, crowd_mask)
        gt_class_ids = tf.ragged.boolean_mask(gt_class_ids, non_crowd_mask)
        gt_boxes = tf.ragged.boolean_mask(gt_boxes, non_crowd_mask)
        gt_masks = tf.ragged.boolean_mask(gt_masks, non_crowd_mask)

        batch_size = gt_boxes.shape[0]
        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = utils.batch_slice([proposals, gt_boxes],
                               lambda x, y: utils.compute_overlaps(x, y),
                               batch_size)

        # Compute overlaps with crowd boxes [proposals, crowd_boxes]
        crowd_overlaps = utils.batch_slice([proposals, crowd_boxes],
                                     lambda x, y: utils.compute_overlaps(x, y),
                                     batch_size)

        # overlap shape = [batch, proposals/image, gt or crowd boxes/image]
        # This is the max IOU for each proposal
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=2)
        no_crowd_mask = (crowd_iou_max < 0.001)

        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=2)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_mask = (roi_iou_max >= 0.5)
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_roi_mask = tf.logical_and(roi_iou_max < 0.5, no_crowd_mask)

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(TRAIN_ROIS_PER_IMAGE * ROI_POSITIVE_RATIO)
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

    def call(self, x, train_bn=True, gt_class_ids=None, gt_boxes=None,
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
        anchors = np.broadcast_to(anchors, (BATCH_SIZE,) + anchors.shape)

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
            training_rois, target_class_ids, target_bbox, target_mask = \
                self._get_detection_targets(proposals, gt_class_ids, gt_boxes, gt_masks)

