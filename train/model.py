"model.py: contains the MusicXMLModel"

import torch
import torch.nn as nn
from transformers import SwinConfig, SwinModel, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTImageProcessor, RobertaConfig, RobertaModel
from transformers import CONFIG_MAPPING, MODEL_MAPPING
from decoder import XMLDecoderConfig, XMLDecoder

class MusicXMLModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Config of the model
        self.config_encoder = SwinConfig()
        # self.config_decoder = XMLDecoderConfig()
        self.config_decoder = RobertaConfig()
        # config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(self.config_encoder, self.config_decoder)

        # Image processor to resize and patch the input image if necessary
        self.image_processor = ViTImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")

        # Encoder, decoder and the model
        self.encoder = SwinModel(self.config_encoder)
        # self.decoder = XMLDecoder(self.config_decoder)
        self.decoder = RobertaModel(self.config_decoder)
        self.model = VisionEncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)

    def forward(self, X):
        X = self.image_processor(X, return_tensors='pt').pixel_values
        return self.model(X)


if __name__ == '__main__':
    # Add the decoder to the library
    # CONFIG_MAPPING.update({"xml_decoder": XMLDecoderConfig})
    # MODEL_MAPPING.update({"xml_decoder": XMLDecoder})
    model = MusicXMLModel()
    
    