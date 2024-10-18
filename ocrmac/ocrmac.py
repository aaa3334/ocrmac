"""Main module."""

import io
import objc

from PIL import ImageFont, ImageDraw, Image
import math
import sys 

if sys.version_info < (3, 9):
    from typing import List, Dict, Set, Tuple
else:
    List, Tuple = list, tuple

import Vision

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def pil2buf(pil_image: Image.Image):
    """Convert PIL image to buffer"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def convert_coordinates_pyplot(bbox, im_width, im_height):
    """Convert vision coordinates to matplotlib coordinates"""
    x, y, w, h = bbox
    x1 = x * im_width
    y1 = (1 - y) * im_height

    x2 = w * im_width
    y2 = -h * im_height
    return x1, y1, x2, y2


def convert_coordinates_pil_other(bbox, im_width, im_height):
    """Convert vision coordinates to PIL coordinates"""
    x, y, w, h = bbox
    x1 = x * im_width
    y2 = (1 - y) * im_height

    x2 = x1 + w * im_width
    y1 = y2 - h * im_height

    return x1, y1, x2, y2

def convert_coordinates_pil(bbox, im_width, im_height):
    """Convert Vision coordinates to PIL coordinates for quadrilateral bounding boxes"""
    # bbox is now a list of four points: [top_left, top_right, bottom_right, bottom_left]
    x_coords = [point[0] * im_width for point in bbox]
    y_coords = [(1 - point[1]) * im_height for point in bbox]
    
    # Return the coordinates of the four corners directly
    return list(zip(x_coords, y_coords))




def calculate_angle(p1, p2):
    """Calculate the angle between two points in degrees."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def is_diagonal(text,bbox, tolerance=20):
    """Determine if the bounding box is diagonal.
    
    Args:
        bbox (list): List of four points representing the bounding box.
        tolerance (int, optional): Percentage tolerance for angle comparison. Defaults to 10.
    
    Returns:
        bool: True if the bounding box is diagonal, False otherwise.
    """
    angles = [
        calculate_angle(bbox[0], bbox[1]),
        calculate_angle(bbox[1], bbox[2]),
        calculate_angle(bbox[2], bbox[3]),
        calculate_angle(bbox[3], bbox[0])
    ]
    # print("text",text,"bbox", bbox, "angles", angles)
    
    # Check if angles match the specific values within the tolerance
    specific_angles = [0.0, -90.0, 180.0, 90.0]
    for angle, specific_angle in zip(angles, specific_angles):
        if specific_angle == 0.0:
            lower_bound = -tolerance
            upper_bound = tolerance
        else:
            lower_bound = specific_angle - abs(specific_angle * tolerance / 100)
            upper_bound = specific_angle + abs(specific_angle * tolerance / 100)
        
        # print("specific_angle", specific_angle, "lower_bound", lower_bound, "upper_bound", upper_bound)
            
        if not (lower_bound <= angle <= upper_bound):
                break
    else:
        return False
        
    return True
    

    # Comment out the rest of the logic for now
    # Check if any angle is within the specified bounds
    # for angle in angles:
    #     if lower_bound <= abs(angle) <= upper_bound or diagonal_lower_bound <= abs(angle) <= diagonal_upper_bound:
    #         return True


def text_from_image(
    image, recognition_level="accurate", language_preference=None, confidence_threshold=0.0
) -> List[Tuple[str, float, List[Tuple[float, float]], bool]]:
    """
    Helper function to call VNRecognizeTextRequest from Apple's vision framework.

    :param image: Path to image (str) or PIL Image.Image.
    :param recognition_level: Recognition level. Defaults to 'accurate'.
    :param language_preference: Language preference. Defaults to None.
    :param confidence_threshold: Confidence threshold. Defaults to 0.0.

    :returns: List of tuples containing the text, the confidence, the bounding box, and a watermark flag.
        Each tuple looks like (text, confidence, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], watermark)
        The bounding box is composed of four points, each represented by (x, y) coordinates.
    """

    if isinstance(image, str):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Invalid image format. Image must be a path or a PIL image.")

    if recognition_level not in {"accurate", "fast"}:
        raise ValueError(
            "Invalid recognition level. Recognition level must be 'accurate' or 'fast'."
        )

    if language_preference is not None and not isinstance(language_preference, list):
        raise ValueError(
            "Invalid language preference format. Language preference must be a list."
        )

    with objc.autorelease_pool():
        req = Vision.VNRecognizeTextRequest.alloc().init()

        if recognition_level == "fast":
            req.setRecognitionLevel_(1)
        else:
            req.setRecognitionLevel_(0)

        if language_preference is not None:
            available_languages = req.supportedRecognitionLanguagesAndReturnError_(None)[0]

            if not set(language_preference).issubset(set(available_languages)):
                raise ValueError(
                    f"Invalid language preference. Language preference must be a subset of {available_languages}."
                )
            req.setRecognitionLanguages_(language_preference)

        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
            pil2buf(image), None
        )

        success = handler.performRequests_error_([req], None)
        res = []

        if success:
            for result in req.results():
                top_candidate = result.topCandidates_(1)[0]
                if top_candidate.confidence() >= confidence_threshold:
                    text = top_candidate.string()
                    confidence = top_candidate.confidence()
                    bbox = [
                        (result.topLeft().x, result.topLeft().y),
                        (result.topRight().x, result.topRight().y),
                        (result.bottomRight().x, result.bottomRight().y),
                        (result.bottomLeft().x, result.bottomLeft().y)
                    ]
                    watermark = is_diagonal(text,bbox)
                    
                    res.append((text, confidence, bbox, watermark))
        # print(res)
        return res

class OCR:
    def __init__(self, image, recognition_level="accurate", language_preference=None, confidence_threshold=0.0):
        """OCR class to extract text from images.

        Args:
            image (str or PIL image): Path to image or PIL image.
            recognition_level (str, optional): Recognition level. Defaults to 'accurate'.
            language_preference (list, optional): Language preference. Defaults to None.
            param confidence_threshold: Confidence threshold. Defaults to 0.0.

        """

        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(
                "Invalid image format. Image must be a path or a PIL image."
            )

        self.image = image
        self.recognition_level = recognition_level
        self.language_preference = language_preference
        self.confidence_threshold = confidence_threshold

        self.res = None

    def recognize(
        self, px=False
    ) -> List[Tuple[str, float, List[Tuple[float, float]], bool]]:
        res = text_from_image(
            self.image, self.recognition_level, self.language_preference, self.confidence_threshold
        )
        self.res = res
        
        if px:
            return [(text, conf, convert_coordinates_pil(bbox, self.image.width, self.image.height), watermark) for text, conf, bbox, watermark in res]

        else:
            return res

    def annotate_matplotlib(
        self, figsize=(20, 20), color="red", alpha=0.5, fontsize=12
    ):
        """_summary_

        Args:
            figsize (tuple, optional): _description_. Defaults to (20,20).
            color (str, optional): _description_. Defaults to 'red'.
            alpha (float, optional): _description_. Defaults to 0.5.
            fontsize (int, optional): _description_. Defaults to 12.

        Returns:
            _type_: _description_

        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "Matplotlib is not available. Please install matplotlib to use this feature."
            )

        if self.res is None:
            self.recognize()

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.image, alpha=alpha)
        for _ in self.res:
            text, conf, bbox, watermark = _
            x1, y1, x2, y2 = convert_coordinates_pyplot(
                bbox, self.image.width, self.image.height
            )
            rect = patches.Rectangle(
                (x1, y1), x2, y2, linewidth=1, edgecolor=color, facecolor="none"
            )
            plt.text(x1, y1, text, fontsize=fontsize, color=color)
            ax.add_patch(rect)

        return fig

    
    
    def annotate_PIL(self, color="red", fontsize=12) -> Image.Image:
        """Annotate the image with bounding boxes and text.

        Args:
            color (str, optional): Default color of the bounding box and text. Defaults to 'red'.
            fontsize (int, optional): Font size of the text. Defaults to 12.

        Returns:
            Image.Image: Annotated image.
        """

        annotated_image = self.image.copy()

        if self.res is None:
            self.recognize()

        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("Arial Unicode.ttf", fontsize)

        for text, conf, bbox, watermark in self.res:
            points = convert_coordinates_pil(
                bbox, annotated_image.width, annotated_image.height
            )
            box_color = "blue" if watermark else color
            draw.polygon(points, outline=box_color)
            draw.text(points[3], text, font=font, align="left", fill=box_color)  # Adjust text position if needed

        return annotated_image