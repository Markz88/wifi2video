import torch
import torch.nn as nn

def weights_init(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)

###### NGF and NDF notes #####
# 64 for 32x32 input image   #
# 32 for 64x64 input image   #
# 16 for 128x128 input image #
##############################

# Encoder network for signal features
class Encoder_sgnl(nn.Module):
    def __init__(self, features_size, hidden_size, latent_size, ngf, netlayers, ngpu):
        super(Encoder_sgnl, self).__init__()

        # Number of GPUs to use
        self.ngpu = ngpu

        # LTM layer
        model = [
            nn.LSTM(features_size, hidden_size, batch_first=True),
        ]

        # LSTM-based network
        self.lstm = nn.Sequential(*model)

        # Multiplayer to compute the channels of the last conv layer from video encoder
        self.mult = 2 ** (netlayers - 1)
        # LSTM output shape [N x z_size x 1 x 1 x 1] -> CNN output shape [N x C x D x H x W]
        cnn = [nn.ConvTranspose3d(latent_size, int(ngf * self.mult), [int(depth_size), int(conv_size), int(conv_size)], 1, 0, bias=False)]
        # 3D conv operation
        self.cnn = nn.Sequential(*cnn)
        
    def forward(self, input_data):
        """
        :param input_data: signal features [batch samples, packets per sample, features per sample]
        :return: latent vector, latent features
        """
        # latent vector, (final hidden state, final cell state)
        z, (h_n, c_n)  = self.lstm(input_data)
        # Get the latent vector of the last LSTM unit - shape [N x z_size]
        latent = z[:,-1,:]
        # Reshape latent vector - shape [N x z_size x 1 x 1 x 1]
        reshaped_latent = latent.unsqueeze(2)
        reshaped_latent = reshaped_latent.unsqueeze(2)
        reshaped_latent = reshaped_latent.unsqueeze(2)
        # 3D conv on latent features - shape [N x C x D x H x W]
        features = self.cnn(reshaped_latent)

        return features, latent

# Encoder network for video features
class Encoder_video(nn.Module):

    def __init__(self, img_size, img_ch, ngf, ngpu, z_size, nframes, downscaling):
        super(Encoder_video, self).__init__()

        # Number of GPUs to use
        self.ngpu = ngpu
        # Multiplayer to compute the channels of the current conv layer
        self.mult = 0

        # ARCHITECTURE DEFINITION
        # Keep track of 3D conv size for the signal encoder conv operation
        global conv_size
        conv_size = 0
        # Keep track of temporal depth (frames encoding) for the signal encoder conv operation
        global depth_size
        depth_size = 0

        # First CNN downscaling layer
        model = [nn.Conv3d(img_ch, ngf, 4, 2, 1, bias=False),
                 nn.LeakyReLU(0.2, inplace=True)]

        # Update conv size
        conv_size = img_size / 2
        # Update temporal depth
        depth_size = nframes / 2
        # Downscaling CNN layers
        for i in range(downscaling - 1):
            # CNN layer
            self.mult = 2 ** i
            model += [nn.Conv3d(ngf * self.mult, ngf * self.mult * 2, 4, 2, 1, bias=False),
                      nn.BatchNorm3d(ngf * self.mult * 2),
                      nn.LeakyReLU(0.2, inplace=True)]

            # Update conv size
            conv_size = conv_size / 2
            # Update temporal depth
            depth_size = depth_size / 2

        # 3D conv latent feature maps (video encoding) - shape [N x C x D x H x W]
        self.features = nn.Sequential(*model)

        # CNN output shape [N x C x D x H x W] -> Latent vector [N x z_size x 1 x 1 x 1]
        latent = [nn.Conv3d(ngf * self.mult * 2, z_size, [int(depth_size), int(conv_size), int(conv_size)], 1, 0, bias=False)]

        # Last 3D conv for latent vector
        self.latent = nn.Sequential(*latent)

    def forward(self, input_data):
        """
        :param input_data: real video frames [batch samples, frames per video, image channels, image height, image width]
        :return: latent feature maps, latent vector
        """
        
        # Feature maps - shape [N x C x D x H x W]
        features = self.features(input_data)
        # Latent vector [N x z_size] - not used
        latent = self.latent(features)
        latent = latent.view(-1, 1).squeeze(1)

        return features, latent

# Decoder network for videos
class Decoder(nn.Module):

    def __init__(self, img_ch, ndf, ngpu, z_size, upscaling):
        super(Decoder, self).__init__()

        # Number of GPUs to use
        self.ngpu = ngpu
        # Multiplayer to compute the channels of the last conv layer from video encoder
        self.mult = 2 ** (upscaling - 1)

        # ARCHITECTURE DEFINITION

        # Init the model
        model = []

        # First step for decoding from latent representazion z
        #model += [nn.ConvTranspose3d(z_size, int(ndf * self.mult), [int(frame_size),int(conv_size),int(conv_size)], 1, 0, bias=False),
        #          nn.BatchNorm3d(int(ndf * self.mult)),
        #          nn.ReLU(True)]

        # Upscaling CNN layers
        for i in range(upscaling - 1):
            # CNN layer
            self.mult = 2 ** ((upscaling - 1) - i)
            model += [nn.ConvTranspose3d(ndf * self.mult, int(ndf * self.mult / 2), 4, 2, 1, bias=False),
                      nn.BatchNorm3d(int(ndf * self.mult / 2)),
                      nn.ReLU(True)]

        # Final CNN layer to obtain video
        model += [nn.ConvTranspose3d(ndf, img_ch, 4, 2, 1, bias=False),
                  nn.Tanh()]

        # Video decoding
        self.model = nn.Sequential(*model)

    def forward(self, input_data):
        """
        :param input_data: latent feature maps
        :return: fake video frames
        """
        
        # fake video frames
        output = self.model(input_data)

        return output

# Disciminator
class Discriminator(nn.Module):

    def __init__(self, img_size, nframes, img_ch, ngf, ngpu, downscaling, latent_size = 1):
        super(Discriminator, self).__init__()

        # Number of GPUs to use
        self.ngpu = ngpu
        # Multiplayer to compute the channels of the current conv layer
        self.mult = 0

        # Keep track of 3D conv size
        self.conv_size = 0
        # Keep track of temporal depth (frames encoding)
        self.frame_size = 0

        # ARCHITECTURE DEFINITION

        # First downscaling layer
        model = [nn.Conv3d(img_ch, ngf, 4, 2, 1, bias=False),
                 nn.LeakyReLU(0.2, inplace=True)]

        # Update conv size
        self.conv_size = img_size / 2
        # Update temporal depth
        self.frame_size = nframes / 2
        # Downscaling CNN layers
        for i in range(downscaling - 1):
            # CNN layer
            self.mult = 2 ** i
            model += [nn.Conv3d(ngf * self.mult, ngf * self.mult * 2, 4, 2, 1, bias=False),
                      nn.BatchNorm3d(ngf * self.mult * 2),
                      nn.LeakyReLU(0.2, inplace=True)]

            # Update conv size
            self.conv_size = self.conv_size / 2
            # Update temporal depth
            self.frame_size = self.frame_size / 2

        # 3D conv latent feature maps - shape [N x C x D x H x W]
        self.features = nn.Sequential(*model)

        # CNN output shape [N x C x D x H x W] -> Latent vector [N x z_size x 1 x 1 x 1] + Classification
        classifier = [nn.Conv3d(ngf * self.mult * 2, latent_size, [int(self.frame_size), int(self.conv_size), int(self.conv_size)], 1, 0, bias=False)]
                  #nn.Sigmoid()]

        # Last 3D conv for latent vector + Classification
        self.classifier = nn.Sequential(*classifier)

    def forward(self, input_data):
        """
        :param input_data: video frames [batch samples, frames per video, image channels, image height, image width]
        :return: latent feature maps, classification (real/fake)
        """
       
        # Feature maps - shape [N x C x D x H x W]
        features = self.features(input_data)
        # Latent vector - shape [N x z_size] + Classification
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return features, classifier