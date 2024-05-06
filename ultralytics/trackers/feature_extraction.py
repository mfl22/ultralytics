from pathlib import Path

import torch
from torchreid.utils import FeatureExtractor

# path to pretrained feature extractor (Re-ID) model
MODEL_PATH = Path.home().joinpath(
    'projekti/upwork/tracking/git/deep-person-reid/checkpoints/'
    'osnet_x1_0_imagenet.pth'
)


class FeatureExtractorClass:

    def __init__(self, model_name='osnet_x1_0', model_path=None):
        """
        Initialize feature extractor.

        Wraps underlying feature extractor that is used.
        In this case, torch-reid
            (https://github.com/KaiyangZhou/deep-person-reid)

        Inputs:
            model_name - name of the pretrained model to use (downloads
                automatically if not already available)
            model_path - path to model if using some other model
                (default: None = use default pretrained model)
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device,
        )

    def extract_features(self, image):
        """
        Call underlying feature extractor on given input image.

        Can work on batches too.
        """
        features = self.extractor(image)

        return features

    def inference(self, img, dets):
        """
        Compatibility API for yolov8.

        Inputs:
            img - input image (not used here)
            dets - detections (crops / bounding boxes of objects)
        """
        features = self.extract_features(dets)

        return features


# create extractor
MyFeatureExtractor = FeatureExtractorClass(model_path=MODEL_PATH)


if __name__ == '__main__':

    """
    Some simple test.
    """

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='a/b/c/model.pth.tar',
        device='cuda'
    )

    image_list = [
        'a/b/c/image001.jpg',
        'a/b/c/image002.jpg',
        'a/b/c/image003.jpg',
        'a/b/c/image004.jpg',
        'a/b/c/image005.jpg'
    ]

    features = extractor(image_list)
    print(features.shape)  # output (5, 512)
