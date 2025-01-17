import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.nn import functional as F


class SiTTModel(BaseModel):
    """
    This class implements the SiTT model, for learning image-to-image data

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    SiTT paper: https://arxiv.org/abs/2106.13804.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Output: I'_A, I'_B

        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.

        Adversarial loss: E[log D_B(I_B)] + E[1 - log D_B(I'_B)] + E[log D_A(I_A)] + E[1 - log D_A(I'_A)]
        Identity loss : lambda_idt * E[||I_BB - I_B||] + E[||I_AA - I_A||]
        Cycle-Consistency loss: lambda_rec * E[||I'_BA - I_A||] + E[||I'_AB - I_B||]
        Perceptual loss: lambda_f * L_f
        KL divergence loss: lambda_kl * L_kl
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_idt', type=float, default=1.0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_rec', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_kl', type=float, default=0.2, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_f', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_gram', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B',
                           'per_A', 'per_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_idt > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # G_A: A -> B, G_B: B -> A
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'sitt', opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_idt > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionPer = networks.VGGLoss().to(self.device)
            # self.criterionGram = networks.GramLoss().to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionKL = torch.nn.KLDivLoss(reduction='batchmean')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),
                                                lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        # real_A/real_B shape: [1, 3, 256, 256]
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A, self.real_B, 'b')  # G_A(A, B) - Content A + Texture B + Decoder B - I'B
        self.texture_B_1 = self.netG.module.texture(self.fake_B)  #
        self.texture_B_2 = self.netG.module.texture(self.real_B)  #

        self.idt_A = self.netG(self.real_A, self.real_A, 'a')  # IAA checked
        self.rec_A = self.netG(self.fake_B, self.real_A, 'a')  # G_B(G_A(A)) - Content I'B + Texture A - I'BA

        self.fake_A = self.netG(self.real_B, self.real_A, 'a')  # G_B(B) - Content B + Texture A - I'A
        self.texture_A_1 = self.netG.module.texture(self.fake_A)  #
        self.texture_A_2 = self.netG.module.texture(self.real_A)  #

        self.idt_B = self.netG(self.real_B, self.real_B, 'b')  # IAA
        self.rec_B = self.netG(self.fake_A, self.real_B, 'b')  # G_A(G_B(B)) - Content I'A + Texture B - I'AB

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_A)  # judge whether B is all right

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)  # judge whether A is all right

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_idt
        lambda_rec = self.opt.lambda_rec
        lambda_kl = self.opt.lambda_kl
        lambda_gram = self.opt.lambda_gram
        lambda_f = self.opt.lambda_f
        # Identity loss

        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A) * lambda_idt
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B) * lambda_idt

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_A), True)  # TODO: check this line
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_rec
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_rec

        # Perceptual Loss
        self.loss_per_A = self.criterionPer(self.fake_B, self.real_A) * lambda_f
        self.loss_per_B = self.criterionPer(self.fake_A, self.real_B) * lambda_f

        # TODO: set texture to some matrices larger than 1 * 1
        # print(self.texture_A_1.shape)
        # print(self.texture_A_2.shape)
        # self.loss_gram_A = self.criterionGram(self.texture_A_1, self.texture_A_2) * lambda_gram
        # self.loss_gram_B = self.criterionGram(self.texture_B_1, self.texture_B_2) * lambda_gram

        # self.loss_kl_A = self.criterionKL(F.log_softmax(self.texture_A_1.reshape(self.texture_A_1.shape[0], -1), -1),
        #                                   F.softmax(self.texture_A_2.reshape(self.texture_A_2.shape[0], -1), -1)) * lambda_kl
        # self.loss_kl_B = self.criterionKL(F.log_softmax(self.texture_B_1.reshape(self.texture_B_1.shape[0], -1), -1),
        #                                   F.softmax(self.texture_B_2.reshape(self.texture_B_2.shape[0], -1), -1)) * lambda_kl
        # combined loss and calculate gradients
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B +
                       self.loss_idt_A + self.loss_idt_B + self.loss_per_A + self.loss_per_B )

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
