"""
Functions for explaining classifiers that use Multimodal data.
Developed initially for the hateful memes challenge in mmf
"""

import copy
from functools import partial
import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state
from tqdm.auto import tqdm

from lime.wrappers.scikit_image import SegmentationAlgorithm
from mmxai.interpretability.classification.lime.lime_base import LimeBase

from lime.lime_text import IndexedString, TextDomainMapper
from lime.exceptions import LimeError

from sklearn.calibration import CalibratedClassifierCV
from PIL import Image
from skimage.segmentation import mark_boundaries

# from object_detection import *


class MultiModalExplanation(object):
    def __init__(
        self,
        image,
        segments,
        domain_mapper,
        n_txt_features,
        n_img_features,
        n_detection_features,
        ratio_txt_img,
        detection_label,
        mode="classification",
        class_names=None,
        random_state=None,
    ):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
            domain_mapper: Maps text feature ids to words or word-positions
            n_txt_features:number of words to include in explanation
            n_img_features: number of superpixels to include in explanation
            n_detection_features:number of detected objects to include in explanation
            ratio_txt_img: weight ratio between text and image features
            detection_label: numpy array of names of detected objects
            mode: what kind of model for this object, classification default
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.image = image
        self.segments = segments
        self.random_state = random_state
        self.mode = mode
        self.domain_mapper = domain_mapper
        self.n_txt_features = n_txt_features
        self.n_img_features = n_img_features
        self.n_detection_features = n_detection_features
        self.ratio_txt_img = ratio_txt_img
        self.detection_label = detection_label
        self.local_exp = {}
        self.intercept = {}
        self.score = {}
        self.local_pred = {}
        self.unsorted_weights = {}

        # divide explanations of the two modalities
        self.local_exp_img = {}
        self.local_exp_txt = {}
        self.local_exp_det = {}

        if mode == "classification":
            self.class_names = class_names
            self.top_labels = None
            self.predict_proba = None

    def get_image_and_mask(
        self,
        label,
        positive_only=True,
        negative_only=False,
        hide_rest=False,
        num_features=5,
        min_weight=0.0,
    ):
        """

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation
        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """

        if label not in self.local_exp_img:
            raise KeyError("Label not in explanation")
        if positive_only & negative_only:
            raise ValueError(
                "Positive_only and negative_only cannot be true at the same time."
            )
        segments = self.segments
        image = self.image
        exp = self.local_exp_img[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp if x[1] < 0 and abs(x[1]) > min_weight][
                :num_features
            ]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) >= min_weight:
                    c = 0 if w < 0 else 1
                    mask[segments == f] = -1 if w < 0 else 1
                    temp[segments == f] = image[segments == f].copy()
                    temp[segments == f, c] = np.max(image)
            return temp, mask

    def as_list(self, label=1, **kwargs):
        """
        Returns the explanation as a list.
        Args:
            label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
                Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper
        Returns:
            list of tuples (representation, weight), where representation is
            given by domain_mapper. Weight is a float.
        """
        label_to_use = label
        ans = self.domain_mapper.map_exp_ids(self.local_exp_txt[label_to_use], **kwargs)
        ans = [(x[0], float(x[1])) for x in ans]
        return ans

    def get_explanation(self, label, num_features=10, which_exp="positive"):

        # the explanation to display:
        """
        :param label: label to explain
        :param num_features: how many top features to display
        :param which_exp: want features that encourage or discourage the dicision (label)
        :return:
            text_message: informative message to interpret text features
            img_message: informative message to interpret image features
            txt_exp_list: text part of the explanation, ready to display
            temp, mask: image part of the explanation, ready to display
        """
        this_exp = np.array(self.local_exp[label])

        if which_exp == "positive":
            positives = this_exp[this_exp[:, 1] >= 0]
        else:  # negative
            positives = this_exp[this_exp[:, 1] < 0]

        if positives.shape[0] < num_features:
            num_features = positives.shape[0]
        top_exp = positives[:num_features]
        top_exp_unique, top_idx = np.unique(top_exp[:, 0], return_index=True)

        txt_exp = top_exp[top_exp[:, 0] < self.n_txt_features]
        n_txt_exp = txt_exp.shape[0]
        txt_top_idx = []
        for txt_feature in txt_exp:
            txt_top_idx.append(top_idx[top_exp_unique == txt_feature[0]] + 1)

        img_exp = top_exp[self.n_txt_features <= top_exp[:, 0]]
        img_exp = img_exp[img_exp[:, 0] < (self.n_txt_features + self.n_img_features)]
        n_img_exp = img_exp.shape[0]
        img_top_idx = []
        for img_feature in img_exp:
            img_top_idx.append(top_idx[top_exp_unique == img_feature[0]] + 1)

        # detection features
        det_exp = top_exp[top_exp[:, 0] >= self.n_txt_features + self.n_img_features]
        n_det_exp = det_exp.shape[0]
        det_top_idx = []
        for det_feature in det_exp:
            det_top_idx.append(
                det_feature[0] - n_txt_exp - n_img_exp
            )  # index for retrieving the labels

        if n_det_exp != 0:
            readable_exp_det = (
                f" Also, we have detected {n_det_exp} types "
                f"of objects from the input image that can be the reason for the decision, they are:"
            )
            for i in det_top_idx:
                readable_exp_det += str(self.detection_label[i])
        else:
            readable_exp_det = (
                " No objects in the image contributed to the model decision"
            )

        # explanation of explanations
        readable_exp_txt = ""
        readable_exp_img = ""
        if n_txt_exp > 0:
            readable_exp_txt = f"{n_txt_exp} are from the text (the top"
            for i in txt_top_idx:
                readable_exp_txt += str(i)
            readable_exp_txt += "th), "
        else:
            readable_exp_txt += "none are from the words, "

        if n_img_exp > 0:
            readable_exp_img = f"{n_img_exp} are from the image (the top"
            for j in img_top_idx:
                readable_exp_img += str(j)
            readable_exp_img += (
                "th, some adjacent regions might merge into a larger area)."
            )
        else:
            readable_exp_img += "none are from the image pixel areas."

        txt_exp_list = np.array(self.as_list(label), dtype="object")

        # return explanations upon request
        if which_exp == "positive":
            # txt_list = txt_exp_list[txt_exp_list[:, 1] >= 0]
            temp, mask = self.get_image_and_mask(
                label, num_features=n_img_exp, positive_only=True
            )
        else:
            # txt_list = txt_exp_list[txt_exp_list[:, 1] < 0]
            temp, mask = self.get_image_and_mask(
                label, num_features=n_img_exp, positive_only=False, negative_only=True
            )
        # txt_list = txt_list[:n_txt_exp]

        # image and text hover display message
        label_decision = ""
        if label == 1:
            label_decision = "hateful"
        else:
            label_decision = "not hateful"

        txt_message = (
            f"For this result, the value associated with each word indicates how much it pushes "
            f"the model towards making a {label_decision} decision."
        )

        img_message = (
            f"Your image has been segmented into {self.n_img_features} small pixel areas, "
            f"the ones that most encourage (or discourage) your "
            f"model decision has been marked by the yellow boundaries."
        )

        top_message = (
            f"Each small pixel area in your image input and each distinct word "
            f"in your text input are called an interpretable feature. There are "
            f"{self.n_img_features + self.n_txt_features} features in total ({self.n_img_features} "
            f"pixel areas and {self.n_txt_features} words). Among the top 10 "
            f"such features that encourage (or discourage) your model decision, "
        )
        top_message = top_message + readable_exp_txt + readable_exp_img

        ratio_message = self.get_txt_img_ratio()

        # format new line as html
        img_message = "<p>" + img_message + "</p><p>" + top_message + "</p>"
        img_message = img_message + ratio_message
        return txt_message, img_message, txt_exp_list, temp, mask

    def get_txt_img_ratio(self):
        """
        Get informative message about the weight ratio between text and image features
        Return:
            words: informative string message to interpret the relative weight of text and image features
        """
        img_percentage = 1 / (1 + self.ratio_txt_img)
        txt_percentage = 1 - img_percentage
        words = (
            f"For this prediction, the relative importance of "
            f"text and image inputs to your model decision are respectively {round(100*txt_percentage, 2)}% "
            f"and {round(100*img_percentage, 2)}%"
        )
        return words


class LimeMultimodalExplainer(object):
    def __init__(
        self,
        image,
        text,
        model,
        kernel_width=0.25,
        kernel=None,
        feature_selection="auto",
        class_names=None,
    ):
        """
        Object to explain predictions on texts and images
        Args:
            image: input 3D numpy array
            text: input string in the meme
            model: multi-modal model to give predictions for given image and text
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
                class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
        """
        self.image = image
        self.text = text
        self.pred_model = model
        self.random_state = check_random_state(None)
        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.feature_selection = feature_selection
        self.class_names = class_names
        self.base = LimeBase(kernel_fn, verbose=False, random_state=self.random_state)

    def explain_instance(self, classifier_fn, n_samples, top_labels=2):

        """
        Generate explanations for a multi-modal input
        Arguments:
            classifier_fn: classification function to give predictions for given texts and images
            num_samples: size of the neighborhood to learn the linear model
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            Return:
                ret_exp: A MultiModalExplanation object with the corresponding
            explanations.
        """
        (
            data,
            labels,
            distances,
            n_txt_features,
            n_img_features,
            segments,
            domain_mapper,
            n_detection_features,
            detection_label,
            ratio_txt_img,
        ) = self.data_labels(n_samples, classifier_fn)
        num_features = data.shape[1]

        if self.class_names is None:
            self.class_names = [str(x) for x in range(labels[0].shape[0])]

        ret_exp = MultiModalExplanation(
            self.image,
            segments,
            domain_mapper=domain_mapper,
            n_txt_features=n_txt_features,
            n_img_features=n_img_features,
            n_detection_features=n_detection_features,
            ratio_txt_img=ratio_txt_img,
            detection_label=detection_label,
            class_names=self.class_names,
            random_state=self.random_state,
        )
        ret_exp.predict_proba = labels[0]

        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (
                ret_exp.intercept[label],
                ret_exp.unsorted_weights[label],
                ret_exp.local_exp[label],
                ret_exp.local_exp_txt[label],
                ret_exp.local_exp_img[label],
                ret_exp.local_exp_det[label],
                ret_exp.score[label],
                ret_exp.local_pred[label],
            ) = self.base.explain_instance_with_data(
                data,
                labels,
                distances,
                label,
                num_features,
                n_txt_features,
                n_img_features,
                n_detection_features,
                feature_selection=None,
            )

            # split local explanation into text and image features

        return ret_exp

    def data_labels(self, num_samples, classifier_fn, detection=False):
        """
        Steps of this function:
            1. generate perturbed text features and image features
            2. in a loop, 1) using these features to make instances of perturbed (text, image) pairs,
                          2) make predictions on these pairs, store labels into 'labels'
            3. concatenate text and image features, store into 'data',
                also append the original input and prediction of it
            4. calculate distances
            Arguments:
                classifier_fn: classification function to give predictions for given texts and images
                num_samples: size of the neighborhood to learn the linear model
                detection: Whether object detection method is invoked, default to be false
            Return:
            data: dense num_samples * num_superpixels
            labels: prediction probabilities matrix
            distances:distance including text/image distance ratio where
            text and image distance are cosine distances between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
            doc_size: number of words in indexed string, where indexed string is the string with various indexes
            n_img_features: number of superpixels to include in explanation
            segments:2d numpy array, with the output from skimage.segmentation
            domain_mapper:Maps text feature ids to words or word-positions
            num_object_detection:number of detected objects to include in explanation
            ori_label: numpy including deteced objects in the original image
            ratio_txt_img: weight ratio between text and image features
        """

        """ 1. make text features """
        indexed_string = IndexedString(
            self.text, bow=True, split_expression=r"\W+", mask_string=None
        )
        domain_mapper = TextDomainMapper(indexed_string)

        doc_size = indexed_string.num_words()
        sample = self.random_state.randint(
            1, doc_size + 1, num_samples
        )  # num_samples - 1
        data_txt = np.ones((num_samples, doc_size))
        # data[0] = np.ones(doc_size)
        features_range = range(doc_size)
        inverse_data_txt = []

        """ 1. make image features """
        random_seed = self.random_state.randint(0, high=1000)
        segmentation_fn = SegmentationAlgorithm(
            "quickshift",
            kernel_size=4,
            max_dist=200,
            ratio=0.2,
            random_seed=random_seed,
        )

        # segmentation_fn = SegmentationAlgorithm('felzenszwalb', scale=200, sigma=2, min_size=100)
        """segmentation_fn = SegmentationAlgorithm('slic', n_segments=60, compactness=10, sigma=1,
                     start_label=1)"""

        segments = segmentation_fn(self.image)  # get segmentation
        n_img_features = np.unique(segments).shape[0]  # get num of superpixel features
        data_img = self.random_state.randint(
            0, 2, n_img_features * num_samples
        ).reshape((num_samples, n_img_features))
        data_img_rows = tqdm(data_img)
        imgs = []

        """ 1. make object detection features 
        if detection:
            predictor, cfg = object_detection_predictor()
            ori_label = object_detection_obtain_label(predictor, cfg, self.image)
            num_object_detection = ori_label.shape[0]
            data_object_detection = np.zeros((num_samples,num_object_detection))"""

        # create fudged_image
        fudged_image = self.image.copy()
        for x in np.unique(segments):
            fudged_image[segments == x] = (
                np.mean(self.image[segments == x][:, 0]),
                np.mean(self.image[segments == x][:, 1]),
                np.mean(self.image[segments == x][:, 2]),
            )

        # img_features[0, :] = 1  # the first sample is the full image                                # num_samples

        """2. create data instances and make predictions"""
        labels = []
        for i, instance in enumerate(zip(sample, data_img_rows)):
            size_txt, row_img = instance

            # make text instance
            inactive = self.random_state.choice(features_range, size_txt, replace=False)
            data_txt[i, inactive] = 0
            inverse_data_txt.append(indexed_string.inverse_removing(inactive))

            # make image instance
            temp = copy.deepcopy(self.image)
            zeros = np.where(row_img == 0)[
                0
            ]  # get segment numbers that are turned off in this instance
            mask = np.zeros(segments.shape).astype(bool)
            for zero in zeros:
                mask[segments == zero] = True
            temp[mask] = fudged_image[mask]

            """if detection:
                label = object_detection_obtain_label(predictor, cfg, temp)
                label_diff = compare_labels(ori_label,label)
                data_object_detection[i] = label_diff"""
            imgs.append(temp)

            # make prediction and append result
            if len(imgs) == 10:
                preds = classifier_fn(self.pred_model, imgs, inverse_data_txt)
                labels.extend(preds)
                imgs = []
                inverse_data_txt = []

        if len(imgs) > 0:
            preds = classifier_fn(self.pred_model, imgs, inverse_data_txt)
            labels.extend(preds)

        """3. concatenate and append features"""
        data = np.concatenate((data_txt, data_img), axis=1)

        # append the original input to the last
        orig_img_f = np.ones((n_img_features,))
        orig_txt_f = np.ones(doc_size)

        """if detection:
            data = np.concatenate((data, data_object_detection),axis=1)
            orig_ot = np.ones(num_object_detection)
            data = np.vstack((data, np.concatenate((np.concatenate((orig_txt_f, orig_img_f)),orig_ot))))
        else:"""
        data = np.vstack((data, np.ones((data.shape[1]))))  ###

        labels.extend(classifier_fn(self.pred_model, [self.image], [self.text]))

        """4. compute distance# distances[:, :(doc_size-1)] *= 100
            use platt scaling t get relative importance of text and image modalities
        """

        labels = np.array(labels, dtype=float)

        # Modify MMF source code to zero out image / text attributes
        # dummy_label_image = np.array(classifier_fn([self.image], [self.text], zero_text=True))  # zero out text
        # dummy_label_text = np.array(classifier_fn([self.image], [self.text], zero_image=True))  # zero out image

        # perform calibration
        try:
            labels_for_calib = np.array(labels[:, 0] < 0.5, dtype=float)
            calibrated = CalibratedClassifierCV(cv=3)
            calibrated.fit(data[:, : doc_size + n_img_features], labels_for_calib)

            calib_data = np.ones((3, doc_size + n_img_features), dtype=float)
            calib_data[0][:doc_size] = 0  # zero out text
            calib_data[1][doc_size:] = 0  # zero out image
            calibrated_labels = calibrated.predict_proba(calib_data)

            delta_txt = abs(calibrated_labels[-1][0] - calibrated_labels[0][0])
            delta_img = abs(calibrated_labels[-1][0] - calibrated_labels[1][0])

            ratio_txt_img = max(min(100, delta_txt / delta_img), 0.01)
        except:
            dummy_text = ""
            dummy_image = np.zeros_like(self.image)
            label_text_out = np.array(
                classifier_fn(
                    self.pred_model, [self.image], [self.text], zero_text=True
                )
            )  # zero out text
            label_image_out = np.array(
                classifier_fn(
                    self.pred_model, [self.image], [self.text], zero_image=True
                )
            )  # zero out image

            delta_txt = abs(labels[-1][0] - label_text_out[0][0])
            delta_img = abs(labels[-1][0] - label_image_out[0][0])
            ratio_txt_img = max(min(10, delta_txt / delta_img), 0.1)

        # calculate distances
        distances_img = sklearn.metrics.pairwise_distances(
            data[:, doc_size:], data[-1, doc_size:].reshape(1, -1), metric="cosine"
        ).ravel()

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[-1], metric="cosine"
            ).ravel()

        distances_txt = distance_fn(sp.sparse.csr_matrix(data[:, :doc_size]))

        distances = (
            1 / (1 + ratio_txt_img) * distances_img
            + (1 - 1 / (1 + ratio_txt_img)) * distances_txt
        )

        # As required by lime_base, make the first element of data, labels, distances the original data point
        data[0] = data[-1]
        labels[0] = labels[-1]
        distances[0] = distances[-1]

        """if not detection:"""
        num_object_detection = 0
        ori_label = None

        return (
            data,
            labels,
            distances,
            doc_size,
            n_img_features,
            segments,
            domain_mapper,
            num_object_detection,
            ori_label,
            ratio_txt_img,
        )
