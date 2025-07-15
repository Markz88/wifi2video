import os
import time
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

from nets.networks import Encoder_sgnl, Encoder_video, Decoder, Discriminator, weights_init
from imports.metrics import evaluation

import cv2
import imageio


class GAN(object):

    @staticmethod
    def name():
        return 'WiFi2VideoNet'

    def __init__(self, opt, usecase, test_set = None):
        super (GAN, self).__init__()

        # General variables
        self.opt = opt # CONFIG file parameters
        self.usecase = usecase # Usecase name

        # Set parameters for building test set
        test_prms = {'batch_size': self.opt['test']['batchsize'],
                     'drop_last': self.opt['test']['drop_last_batch'],
                     'shuffle': self.opt['test']['shuffle'],
                     'num_workers': self.opt['workers']}

        # Build the test set sample pairs (videos, signals, labels, filenames)
        self.testset = DataLoader(test_set, **test_prms)

        # Define the device to use (GPU or CPU)
        self.device = torch.device('cuda:0' if (self.opt['device'] == 'gpu' and torch.cuda.is_available() and self.opt['videoGAN'][
            'ngpu'] > 0) else 'cpu')

        # General variables
        self.real = None  # Real video frames
        self.fake = None  # Fake video frames produced by VideoGAN
        self.sgnl_synth_fake = None  # Fake video frames synthesized from signal features
        self.latent_vector_video = None # Video 1D latent representation (z_size) - not used
        self.latent_vector_sgnl = None # Video 1D latent representation (z_size)
        self.features_video = None # Video 3D feature maps
        self.features_sgnl = None # Signal 3D feature maps
        #self.error_gen_l1 = None # VideoGAN enc l1 loss (i.e., contextual loss)
        self.videogan_loss_gen = None # VideoGAN generator L2 loss (i.e., similarity loss among real and fake frames)
        self.videogan_loss = None # Two-branch network final objective function
        self.synth_loss = None # Video synthesis loss (i.e., similarity loss among fake and synth frames)
        self.latent_features_loss = None # VideoGAN and Signal encdoer features l2 loss

        self.videogan_discr_realpred = None # Predictions for real videos by VideoGAN discriminator
        self.videogan_discr_fakepred = None # Predictions for fake videos by VideoGAN discriminator
        self.videogan_discr_fakebce = None  # VideoGAN discriminator bce loss on fake frames
        self.videogan_discr_loss = None # VideoGAN discriminator total loss

        # VideoGAN discriminator parameters
        self.real_labels = torch.empty(size=(self.opt['test']['batchsize'],), dtype=torch.float32, device=self.device) # VideoGAN discriminator GT real frames
        self.fake_labels = torch.empty(size=(self.opt['test']['batchsize'],), dtype=torch.float32, device=self.device) # VideoGAN discriminator GT fake frames
        self.real_discr_label = 1.0 # Label used by VideoGAN discriminator for real frames
        self.fake_discr_label = 0.0 # Label used by VideoGAN discriminator for fake frames

        ## Networks definitions and weights init
        # VideoGAN encoder
        self.VideoENC = Encoder_video(opt['videoGAN']['img_size'], opt['videoGAN']['img_ch'], opt['videoGAN']['ngf'], opt['videoGAN']['ngpu'], opt['sgnlENC']['latent_size'], opt['videoGAN']['nframes'], opt['videoGAN']['netlayers']).to(self.device)
        # VideoGAN decoder
        self.VideoDEC = Decoder(opt['videoGAN']['img_ch'], opt['videoGAN']['ndf'], opt['videoGAN']['ngpu'], opt['sgnlENC']['latent_size'], opt['videoGAN']['netlayers']).to(self.device)
        # Signal encoder
        self.SgnlENC = Encoder_sgnl(opt['sgnlENC']['features_size'], opt['sgnlENC']['latent_size'], opt['sgnlENC']['latent_size'], opt['videoGAN']['ngf'], opt['videoGAN']['netlayers'], opt['videoGAN']['ngpu']).to(self.device)
        # VideoGAN discriminator
        self.VideoDISCR = Discriminator(opt['videoGAN']['img_size'], opt['videoGAN']['nframes'], opt['videoGAN']['img_ch'], opt['videoGAN']['ngf'], opt['videoGAN']['ngpu'], opt['videoGAN']['netlayers']).to(self.device) # z_size = 1
        # VideoGAN encoder weights init
        self.VideoENC.apply(weights_init)
        # VideoGAN decoder weights init
        self.VideoDEC.apply(weights_init)
        # VideoGAN discriminator weights init
        self.VideoDISCR.apply(weights_init)

        ## Input tensors
        # Real video frames - shape [N x C x D x H x W]
        self.real = torch.empty(size=(
        self.opt['test']['batchsize'], self.opt['videoGAN']['img_ch'], self.opt['videoGAN']['nframes'],
        self.opt['videoGAN']['img_size'], self.opt['videoGAN']['img_size']), dtype=torch.float32,
                                device=self.device)

        # Signal - shape [N x P x K] - P: packets K: features size
        self.signal = torch.empty(size=(self.opt['test']['batchsize'], self.opt['sgnlENC']['input_length'], self.opt['sgnlENC']['features_size']),
                                  dtype=torch.float32, device=self.device)

        # Fake video frames - shape [N x C x D x H x W]
        self.fake = torch.empty(size=(
        self.opt['test']['batchsize'], self.opt['videoGAN']['img_ch'], self.opt['videoGAN']['nframes'],
        self.opt['videoGAN']['img_size'], self.opt['videoGAN']['img_size']), dtype=torch.float32,
                                device=self.device)

        # Fake video frames synthesized from signal features - shape [N x C x D x H x W]
        self.sgnl_synth_fake = torch.empty(size=(
        self.opt['test']['batchsize'], self.opt['videoGAN']['img_ch'], self.opt['videoGAN']['nframes'],
        self.opt['videoGAN']['img_size'], self.opt['videoGAN']['img_size']), dtype=torch.float32,
                                       device=self.device)

    # Set network input (real video frames, signal)
    def set_input(self,data,signals):
        with torch.no_grad():
            # Batch of real video frames
            self.real.resize_(data.size()).copy_(data)
            # Batch of signals
            self.signal.resize_(signals.size()).copy_(signals)

    # Returns test current batch videos
    def get_test_imgs(self):
        reals = self.real.data # Batch of real videos
        sgnl_fakes = self.sgnl_synth_fake.data # Batch of synth videos

        # Reshape required for saving [N x C x D x H x W] -> [N x D x C x H x W]
        reals = reals.permute(0, 2, 1, 3, 4)
        sgnl_fakes = sgnl_fakes.permute(0, 2, 1, 3, 4)

        return reals, sgnl_fakes  

    # Test function
    def test(self):

        # Load trained Signal encoder and VideoGAN decoder weigths for test
        try:
            # Load from weights directory: './models/experiment/'
            self.VideoDEC.load_state_dict(torch.load(
                os.path.join('models', self.opt['experiment'], 'VideoDEC.pth'), map_location=self.device)['state_dict'])

            self.SgnlENC.load_state_dict(torch.load(
                os.path.join('models', self.opt['experiment'], 'SgnlENC.pth'), map_location=self.device)['state_dict'])
        except IOError:
            raise IOError("Weigths not found.")
        print('---> Loaded weights, starting test...')

        # Collect the metric
        self.similarity_scores = []

        # If testing, init time collection
        self.total_testing_time = []

        # network total steps per epoch
        self.total_steps = 0
        # current batch number
        batch_id = 0
            
        # Test the network
        print('>> Testing model: %s.' % self.name())

        # Set VideoGAN decoder in testing mode
        self.VideoDEC.eval()
        # Set Signal encoder in testing mode
        self.SgnlENC.eval()

        with torch.no_grad():
            # for each batch of videos and signals in the testing set
            for videos, signals, labels, names in tqdm(self.testset, leave = False, total=len(self.testset)):
                # Update total steps per epoch
                self.total_steps += self.opt['test']['batchsize']

                # Start test batch size
                start_batch_time = time.time()

                # Set network input
                self.set_input(videos,signals)

                # Signal 3D feature maps
                self.features_sgnl, _ = self.SgnlENC(self.signal)
                # Sythesized fake video frames by VideoGAN decoder from Signal 3D features
                self.sgnl_synth_fake = self.VideoDEC(self.features_sgnl)

                # If chosen, use SSIM as metric
                if self.opt['metric'] == 'SSIM':
                    # Compute SSIM
                    self.similarity_scores.append(evaluation(self.opt, self.real, self.sgnl_synth_fake,
                                                        metric=self.opt['metric']))
                    
                # Plot and save testing videos
                if self.total_steps % self.opt['test']['batchsize'] == 0:
                    reals, sgnl_fakes = self.get_test_imgs()
                    # Save real and synth videos, if enabled
                    if self.opt['save_test_video']:
                        # For each real video - shape [D x C x H x W]
                        for i in range(len(reals)):
                            frames = []

                            # For each frame - shape [C x H x W]
                            j = 0
                            for frame in reals[i]:
                                frame = frame.cpu()
                                min_val = torch.min(frame)
                                max_val = torch.max(frame)

                                if (max_val - min_val) > 1e-6:
                                    frame = (frame - min_val) / (max_val - min_val) # Normalize frame
                                else:
                                    frame = torch.zeros_like(frame) # Normalize frame to zero

                                current_frame = np.array(np.transpose(frame, (1, 2, 0))) # Reshape in [H x W x C] due to opencv
                                current_frame = cv2.cvtColor(np.uint8(255*current_frame), cv2.COLOR_BGR2RGB) # torchvision works in BGR color space
                                current_frame = cv2.resize(current_frame, (256, 256), interpolation=cv2.INTER_CUBIC)  # Resize the frame

                                #current_frame = cv2.putText(current_frame, 'Silh', (10, 120), font, .5, (255, 255, 255), 1, cv2.LINE_AA)

                                frames.append(current_frame) # collection of processed frames

                            # Testing video frames for demo
                            self.demo_dir = os.path.join(self.opt['output_dir'],self.opt['experiment'],'test','demo',self.usecase)
                            if not os.path.exists(self.demo_dir):
                                os.makedirs(self.demo_dir)
                            # Save the video in GIF format
                            imageio.mimsave(self.demo_dir + "/" + self.opt['experiment'] + '_gt.gif', frames, duration=100, loop=0)

                        # For each signal video - shape [D x C x H x W]
                        for i in range(len(sgnl_fakes)):
                            frames = []
                            

                            # For each frame - shape [C x H x W]
                            j = 0
                            for frame in sgnl_fakes[i]:
                                frame = frame.cpu()
                                min_val = torch.min(frame)
                                max_val = torch.max(frame)

                                if (max_val - min_val) > 1e-6:
                                    frame = (frame - min_val) / (max_val - min_val) # Normalize frame
                                else:
                                    frame = torch.zeros_like(frame) # Normalize frame to zero

                                current_frame = np.array(np.transpose(frame, (1, 2, 0))) # Reshape in [H x W x C] due to opencv
                                current_frame = cv2.cvtColor(np.uint8(255*current_frame), cv2.COLOR_BGR2RGB) # torchvision works in BGR color space
                                current_frame = cv2.resize(current_frame, (256, 256), interpolation=cv2.INTER_CUBIC)  # Resize the frame

                                #current_frame = cv2.putText(current_frame, 'experiment', (10, 120), font, .5, (255, 255, 255), 1, cv2.LINE_AA)

                                frames.append(current_frame) # collection of processed frames

                                
                            
                            # Testing video frames for demo
                            self.demo_dir = os.path.join(self.opt['output_dir'],self.opt['experiment'],'test','demo',self.usecase)
                            if not os.path.exists(self.demo_dir):
                                os.makedirs(self.demo_dir)
                            # Save the video in GIF format
                            imageio.mimsave(self.demo_dir + "/" + self.opt['experiment'] + '_pred.gif', frames, duration=100, loop=0)

                # Total batch testing time
                total_batch_time = time.time() - start_batch_time
                # Collect batch time
                self.total_testing_time.append(total_batch_time)

                # Update batch numnber
                batch_id += 1

            # Metric scores
            result = np.asarray(self.similarity_scores, dtype=np.float32)

            # SSIM mean
            if self.opt['metric'] == 'SSIM':
                result = np.mean(result)

                # Print metric
                print("SSIM: %.2f" %(result))

        
            # Compute testing and batch time
            self.total_testing_time = np.asarray(self.total_testing_time)
            # Compute average batch time
            avg_batch_time = np.mean(self.total_testing_time)
            # Compute total testing time
            self.total_testing_time = np.sum(self.total_testing_time)

            return result, self.total_testing_time
