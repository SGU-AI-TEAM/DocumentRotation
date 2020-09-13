import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class DocRecognizer():

    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.__category_index = {
            1: {'id': 1, 'name': 'CMNDF'},
            2: {'id': 2, 'name': 'CMNDB'},
            3: {'id': 3, 'name': 'CCCDF'},
            4: {'id': 4, 'name': 'CCCDB'},
            5: {'id': 5, 'name': 'PP'},
            6: {'id': 6, 'name': 'Card'},
            7: {'id': 7, 'name': 'card'},
            8: {'id': 8, 'name': 'bill'},
            9: {'id': 9, 'name': 'paper'},
        }

    def predict_one(self, data) -> dict:
        image = data
        return self.__inference([image])

    def _load_model(self, model_path):
        """
        load saved model from path
        :param model_path: file path to frozen model
        :return:
        """
        model = tf.saved_model.load(str(model_path))
        # model = model.signatures['serving_default']

        return model

    def __inference(self, images):
        """
        Do inference in inputed images data
        :param images: list of image
        :return: a dict
        """
        tf_images = [tf.convert_to_tensor(image) for image in images]
        batch_image = tf.stack(tf_images)
        # Run inference
        output_dict = self.model(batch_image)
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[:num_detections].numpy()
                       for key, value in output_dict.items()}

        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
            output_dict['detection_masks_reframed'] = []
            for mask, box, image in zip(output_dict.get("detection_masks"), output_dict['detection_boxes'], images):
                # Reframe the the bbox mask to the image size.
                detection_masks_reframed = self.reframe_box_masks_to_image_masks(
                    mask, box,
                    image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                                   tf.uint8)
                output_dict['detection_masks_reframed'].append(detection_masks_reframed.numpy())

        return output_dict

    def reframe_box_masks_to_image_masks(self, box_masks, boxes, image_height,
                                         image_width):
        """Transforms the box masks back to full image masks.
        Embeds masks in bounding boxes of larger masks whose shapes correspond to
        image shape.
        Args:
          box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
          boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
                 corners. Row i contains [ymin, xmin, ymax, xmax] of the box
                 corresponding to mask i. Note that the box corners are in
                 normalized coordinates.
          image_height: Image height. The output mask will have the same height as
                        the image height.
          image_width: Image width. The output mask will have the same width as the
                       image width.
        Returns:
          A tf.float32 tensor of size [num_masks, image_height, image_width].
        """

        def reframe_box_masks_to_image_masks_default():
            """The default function when there are more than 0 box masks."""

            def transform_boxes_relative_to_boxes(boxes, reference_boxes):
                boxes = tf.reshape(boxes, [-1, 2, 2])
                min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
                max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
                transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
                return tf.reshape(transformed_boxes, [-1, 4])

            box_masks_expanded = tf.expand_dims(box_masks, axis=3)
            num_boxes = tf.shape(box_masks_expanded)[0]
            unit_boxes = tf.concat(
                [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
            reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
            return tf.image.crop_and_resize(
                image=box_masks_expanded,
                boxes=reverse_boxes,
                box_indices=tf.range(num_boxes),
                crop_size=[image_height, image_width],
                extrapolation_value=0.0)

        image_masks = tf.cond(
            tf.shape(box_masks)[0] > 0,
            reframe_box_masks_to_image_masks_default,
            lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
        return tf.squeeze(image_masks, axis=3)
