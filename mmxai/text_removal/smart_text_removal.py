from PIL import Image
import cv2 as cv
import numpy as np
import math

from mmxai.utils.image_loader import loadImage


class SmartTextRemover:
    def __init__(self, detector_path):
        """
        Function to instantiate SmartTextRemover object.

        INPUTS:
            detector_path - str: Path to the text detector.
        """
        self.__detector = cv.dnn.readNet(detector_path)

    @property
    def detector(self):
        return self.__detector

    def inpaint(self, image, dilation=0.02, method=cv.INPAINT_TELEA):
        """
        Function to inpaint the text in side an image

        INPUTS:
            image - PIL.Image or numpy.ndarray or str:
                The Image object or pixel array or path to the image.
            dilation - float: The amount of dilation to apply to the text mask.
            method: The inpainting algorithm to used. cv.INPAINT_TELEA or cv.INPAINT_NS.
        RETURNS:
            PIL.Image: The inpainted image.
        """

        image = loadImage(image).convert("RGB")
        image_array = np.array(image)

        vertices = self.getTextBoxes(image_array, debug_show_boxes=False)
        mask = self.generateTextMask(
            vertices,
            image_array.shape[0],
            image_array.shape[1],
            dilation=dilation,
        )

        impainted_image = cv.inpaint(image_array, mask, 15, method)

        # convert to PIL format
        impainted_image = Image.fromarray(impainted_image)

        return impainted_image

    def getTextBoxes(
        self,
        img_PIL,  # image array in RGB format
        conf_threshold=0.5,
        nms_threshold=0.4,
        inp_width=320,
        inp_height=320,
        debug_show_boxes=False,
    ):
        """
        Helper function to detect text inside an image return the text box vertices

        INPUTS:
            img_PIL - numpy.ndarray: Image pixel array in RGB format.
            conf_threshold - float: Confidence threshold.
            nms_threshold - float: Non-maximum suppression threshold.
            inp_width - float: Preprocess input image by resizing to a specific width.
            inp_height - float: Preprocess input image by resizing to a specific height.
            debug_show_boxes - bool: If true, will show the textboxes in a cv.window.
        
        RETURNS:
            numpy.ndarray - The text box vertices in shape of (:, 4, 2).
        """

        # Create a new named window if required
        if debug_show_boxes: # pragma: no cover
            kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
            cv.namedWindow(kWinName, cv.WINDOW_NORMAL)

        # Generate correct out_names
        out_names = []
        out_names.append("feature_fusion/Conv_7/Sigmoid")
        out_names.append("feature_fusion/concat_3")

        # Load the image from path
        img = self.convertRgbToBgr(img_PIL)  # convert PIL format to cv format

        # Get image height and width
        height_ = img.shape[0]
        width_ = img.shape[1]
        rW = width_ / float(inp_width)
        rH = height_ / float(inp_height)

        # Create a 4D blob from image.
        blob = cv.dnn.blobFromImage(
            img, 1.0, (inp_width, inp_height), (123.68, 116.78, 103.94), True, False
        )

        # Run the detection model
        self.detector.setInput(blob)
        outs = self.detector.forward(out_names)

        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = self.decodeBoundingBoxes(
            scores, geometry, conf_threshold
        )

        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(
            boxes, confidences, conf_threshold, nms_threshold
        )

        # Initialise a zero array for all the boxes' vertices
        vertices_all = np.zeros((len(indices), 4, 2), dtype=float)

        for n, i in enumerate(indices):
            # get 4 corners of the rotated rect
            vertices = cv.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH

            # save the vertices of this box to the vertices_all array
            vertices_all[n, :, :] = vertices

            # Add the boxes to current frame if required
            if debug_show_boxes: # pragma: no cover
                for j in range(4):
                    p1 = (vertices[j][0].astype(int), vertices[j][1].astype(int))
                    p2 = (
                        vertices[(j + 1) % 4][0].astype(int),
                        vertices[(j + 1) % 4][1].astype(int),
                    )
                    cv.line(img, p1, p2, (0, 255, 0), 1)

        # Display the image frame if required
        if debug_show_boxes: # pragma: no cover
            cv.imshow(kWinName, img)

            # waits for user to press any key
            cv.waitKey(0)

            # closing all open windows
            cv.destroyAllWindows()

        return vertices_all

    def generateTextMask(self, vertices, img_height, img_width, dilation=0.0):
        """
        Helper function to transform text box vertices to a mask

        INPUTS:
            vertices - numpy.ndarray:
                The text box vertices in shape of (:, 4, 2).
            img_height - int: Image height measured in number of pixels
            img_width - int: Image width measured in number of pixels
            dilation - float: The amount of dilation to apply to the text mask.

        RETURNS:
            numpy.ndarray: Text mask of shape (img_height, img_width).
        """

        # Generate the mesh grid
        x_range = np.arange(0, img_width)
        y_range = np.arange(0, img_height)

        x, y = np.meshgrid(x_range, y_range, sparse=True)

        # Enlarge all the text boxes for more coverage
        del_x = dilation * img_width
        del_y = dilation * img_height

        vertices[:, :2, 0] -= del_x
        vertices[:, 2:, 0] += del_x

        vertices[:, 1:3, 1] -= del_y
        vertices[:, [0, 3], 1] += del_y

        # Generate empty boolean array for storing the mask for all the boxes
        mask_all_boxes = np.zeros(
            (len(vertices), len(y_range), len(x_range)), dtype=bool
        )

        for n, vertex in enumerate(vertices):
            mask_this_box = np.zeros((4, len(y_range), len(x_range)), dtype=bool)

            # On the right of the left edge
            mask_this_box[0] = self.isOnRight(vertex[0], vertex[1], x, y)

            # Above the bottom edge
            mask_this_box[1] = self.isAbove(vertex[1], vertex[2], x, y)

            # On the left of the right edge
            mask_this_box[2] = np.logical_not(
                self.isOnRight(vertex[2], vertex[3], x, y)
            )

            # Below the top edge
            mask_this_box[3] = np.logical_not(self.isAbove(vertex[0], vertex[3], x, y))

            # Store the mask of this box into the boolean array
            mask_all_boxes[n] = mask_this_box.all(axis=0)

        # Generate the mask containing all the text boxes
        mask_bool = mask_all_boxes.any(axis=0)

        # Convert the bool mask to uint8 mask for cv2 standard
        mask = np.zeros((len(y_range), len(x_range)), dtype=np.uint8)
        mask[mask_bool] = 255

        return mask

    def findGradientAndYIntersec(self, p1, p2):
        """
        Helper function to find the gradient and the y-intersection of a straight line
        using the two-point formula

        INPUTS:
            p1 - numpy.ndarray: coordinate of point 1. p1 in shape of (2,).
            p2 - numpy.ndarray: coordinate of point 2. p2 in shape of (2,).
        
        RETURNS:
            float: gradient
            float: y intersection
        """
        gradient = (p2[1] - p1[1]) / (p2[0] - p1[0])
        y_intersec = -gradient * p1[0] + p1[1]

        return gradient, y_intersec

    def isOnRight(self, p1, p2, x, y):
        """
        Helper function to check if the points are on the right of a straight line
        defined by the two-point formula

        INPUTS:
            p1 - numpy.ndarray: coordinate of point 1. p1 in shape of (2,).
            p2 - numpy.ndarray: coordinate of point 2. p2 in shape of (2,).
            x - numpy.ndarray: x coordinates of the points to be checked.
            y - numpy.ndarray: y coordinates of the points to be checked.
            (x.shape must == y.shape)

        RETURNS:
            np.ndarray: array of bools
        """
        gradient, y_intersec = self.findGradientAndYIntersec(p1, p2)

        def findXFromY(y):
            return 1 / gradient * y - y_intersec / gradient

        x_line = findXFromY(y)

        return x > x_line

    def isAbove(self, p1, p2, x, y):
        """
        Helper function to check if the points are above a straight line
        defined by the two-point formula

        INPUTS:
            p1 - numpy.ndarray: coordinate of point 1. p1 in shape of (2,).
            p2 - numpy.ndarray: coordinate of point 2. p2 in shape of (2,).
            x - numpy.ndarray: x coordinates of the points to be checked.
            y - numpy.ndarray: y coordinates of the points to be checked.
            (x.shape must == y.shape)

        RETURNS:
            np.ndarray: array of bools
        """
        gradient, y_intersec = self.findGradientAndYIntersec(p1, p2)
        y_line = gradient * x + y_intersec

        return y > y_line

    def convertRgbToBgr(self, img_RGB: np.ndarray):
        """
        Helper function to convert image array from RGB format to BGR

        INPUT:
            img_RGB - numpy.ndarray: image pixel array in RBG format
        RETURNS:
            numpy.ndarray: image pixel array in BGR format
        """

        img_BGR = img_RGB[:, :, ::-1].copy()

        return img_BGR

    def decodeBoundingBoxes(self, scores, geometry, scoreThresh):
        """
        Helper function to decode bounding boxes.
        See https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
        for the example usage
        """
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert (
            scores.shape[2] == geometry.shape[2]
        ), "Invalid dimensions of scores and geometry"
        assert (
            scores.shape[3] == geometry.shape[3]
        ), "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if score < scoreThresh:
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = [
                    offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                    offsetY - sinA * x1_data[x] + cosA * x2_data[x],
                ]

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]


if __name__ == "__main__": # pragma: no cover
    remover = SmartTextRemover("mmxai/text_removal/frozen_east_text_detection.pb")
    img = remover.inpaint("https://www.iqmetrix.com/hubfs/Meme%2021.jpg")
    img.show()
