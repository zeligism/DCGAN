
import torch.nn as nn


class DCGAN(nn.Module):
    """Deep Convolutional Generative Adversarial Network"""

    def __init__(self,
        num_latents=100,
        num_features=64,
        image_channels=3,
        image_size=64,
        gan_type="gan"):
        """
        Initializes DCGAN.

        Args:
            num_latents: Number of latent factors.
            num_features: Number of features in the convolutions.
            image_channels: Number of channels in the input image.
            image_size: Size (i.e. height or width) of image.
            gan_type: Type of GAN (e.g. "gan" or "wgan-gp").
        """
        super().__init__()

        self.num_latents = num_latents
        self.num_features = num_features
        self.image_channels = image_channels
        self.image_size = image_size
        self.gan_type = gan_type

        D_params = {
            "num_features": num_features,
            "image_channels": image_channels,
            "image_size": image_size,
            "gan_type": gan_type,
        }
        G_params = {
            "num_latents": num_latents,
            "num_features": num_features,
            "image_channels": image_channels,
            "image_size": image_size,
            "gan_type": gan_type,
        }

        self.D = DCGAN_Discriminator(**D_params)
        self.G = DCGAN_Generator(**G_params)


class DCGAN_Discriminator(nn.Module):
    """The discriminator of a DCGAN"""

    def __init__(self,
                 num_latents=1,
                 num_features=64,
                 image_channels=3,
                 image_size=64,
                 gan_type="gan",
                 max_features=512):
        super().__init__()

        using_gradient_penalty = gan_type == "wgan-gp"
        use_batchnorm = not using_gradient_penalty

        # Calculate intermediate image sizes
        image_sizes = [image_size]
        while image_sizes[-1] > 5:
            image_sizes.append(image_sizes[-1] // 2)
        latent_kernel = image_sizes[-1]  # should be either 3, 4, or 5
        num_layers = len(image_sizes) - 1

        # Calculate feature sizes
        features = [min(num_features * 2**i, max_features) for i in range(num_layers)]

        # Input layer
        modules = []
        modules += [DCGAN_DiscriminatorBlock(image_channels, features[0])]

        # Intermediate layers
        for in_features, out_features in zip(features, features[1:]):
            modules += [DCGAN_DiscriminatorBlock(in_features, out_features, use_batchnorm=use_batchnorm)]
        
        # Output layer (feature_size = 3, 4, or 5 -> 1)
        modules += [nn.Conv2d(features[-1], num_latents, latent_kernel, bias=False)]

        if gan_type == "gan":
            modules += [nn.Sigmoid()]

        self.main = nn.Sequential(*modules)

    def forward(self, inputs):
        # Remove H and W dimensions, infer channels dim (remove if 1)
        return self.main(inputs).view(inputs.size(0), -1).squeeze(1)


class DCGAN_DiscriminatorBlock(nn.Module):
    """
    A discriminator convolutional block.
    Default stride and padding half the size of features,
    e.g. if input is [in_channels, 64, 64], output will be [out_channels, 32, 32].
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=False):
        super().__init__()

        modules = []
        modules += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)]
        modules += [nn.BatchNorm2d(out_channels)] if use_batchnorm else []
        modules += [nn.LeakyReLU(0.2, inplace=True)]

        self.main = nn.Sequential(*modules)

    def forward(self, x):
        return self.main(x)


class DCGAN_Generator(nn.Module):
    """The generator of a DCGAN"""

    def __init__(self,
                 num_latents=100,
                 num_features=64,
                 image_channels=3,
                 image_size=64,
                 gan_type="gan",
                 max_features=512):
        super().__init__()

        # Calculate intermediate image sizes
        image_sizes = [image_size]
        while image_sizes[-1] > 5:
            image_sizes.append(image_sizes[-1] // 2)
        latent_kernel = image_sizes[-1]  # should be either 3, 4, or 5
        num_layers = len(image_sizes) - 1

        # Calculate feature sizes
        features = [min(num_features * 2**i, max_features) for i in range(num_layers)]

        # Reverse order of image sizes and features for generator
        image_sizes = image_sizes[::-1]
        features = features[::-1]

        # Input layer
        modules = []
        modules += [DCGAN_GeneratorBlock(num_latents, features[0],
                                            kernel_size=latent_kernel, stride=1, padding=0)]

        # Intermediate layers
        for in_features, out_features, expected_size in zip(features, features[1:], image_sizes[1:]):
            modules += [DCGAN_GeneratorBlock(in_features, out_features, kernel_size=4+(expected_size%2))]
        
        # Output layer
        modules += [nn.ConvTranspose2d(features[-1], image_channels, kernel_size=4+(image_size%2),
                                       stride=2, padding=1, bias=False)]
        modules += [nn.Tanh()]

        self.main = nn.Sequential(*modules)

    def forward(self, inputs):
        # Add H and W dimensions, infer channels dim (add if none)
        return self.main(inputs.view(inputs.size(0), -1, 1, 1))


class DCGAN_GeneratorBlock(nn.Module):
    """
    A generator convolutional block.
    Default stride and padding double the size of features,
    e.g. if input is [in_channels, 32, 32], output will be [out_channels, 64, 64].
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class FlexiDCGAN(DCGAN):
    """
    A DCGAN variant that accepts any square images with length larger than 6.
    It's the same as DCGAN above but I just wanted to give it a cool name.
    """

