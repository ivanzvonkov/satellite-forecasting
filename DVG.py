import warnings

from data.satellite import RGB_BANDS, ALL_BANDS, Normalization
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
import argparse
import random
from torch.utils.data import DataLoader
import utils
import pdb
import numpy as np
import gpytorch
import wandb
from collections import defaultdict
from models.gp_models import GPRegressionLayer1
from pytorch_ssim import SSIM
from typing import List
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import structural_similarity as ssim2

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs_final_version_gp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=1200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
#parser.add_argument('--epoch_size', type=int, default=300, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='kth', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=7, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=90, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--model_path', type=str, default='', help='model pth file with which to resume training')
parser.add_argument('--home_dir', type=str, default='.', help='Where to save gifs, models, etc')
parser.add_argument('--test', type=bool, default=False, help="whether to train or test the model")
parser.add_argument('--run_name', type=str, default='', help='name of run')
parser.add_argument('--loss', default="mse", help='loss function to use (mse | l1 | ssim | ssim_mse_point1 | ssim_mse_point01 | ssim_l1_point1 | ssim_l1_point01 )')
parser.add_argument('--normalization', type=str, default="z", help='normalization to use (z | minmax | skip | clip5_minmax | clip4_minmax | clip3_minmax)')
parser.add_argument('--components', type=str, default="all", help='components to train (encoder | encoder_lstm | all)')
parser.add_argument('--interval_for_gp_layer', type=int, default=10, help='interval at which gaussian process is triggered')
parser.add_argument('--patch_size', type=int, default=64, help='size of patch')

opt = parser.parse_args()
if not opt.test:
    wandb.init(project="satellite-forecasting", entity="izvonkov")
    wandb.config.update(opt)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
home_dir = Path(opt.home_dir)

torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

if opt.model_path:
    model = torch.load(opt.model_path)
    opt_dict = vars(model['opt'])
    for opt_key in ["data_root", "dataset", "n_past", "n_future", "n_eval", "image_width", "channels", "components", "normalization", "batch_size", "patch_size"]:
        if opt_key in opt_dict:
            setattr(opt, opt_key, opt_dict[opt_key])

print("Arguments:")
print(opt)
assert opt.components in ["encoder", "encoder_lstm", "lstm", "all"]
assert opt.image_width % opt.patch_size == 0

# --------- load a dataset ------------------------------------
print('Loading data...')
train_data, test_data = utils.load_dataset(opt, bands_to_keep=ALL_BANDS, normalization=Normalization(opt.normalization))
num_workers = opt.data_threads
if opt.dataset == "satellite":
    print("Satellite dataset only works with num_workers=0")
    num_workers = 0

train_loader = DataLoader(train_data,
                          num_workers=num_workers,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=num_workers,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)

# --------- Figure out which components to enable ------------------------------------
encoder_enabled = opt.components in ["encoder", "encoder_lstm", "all"]     
lstm_enabled =   opt.components in ["encoder_lstm", "lstm", "all"] 
gp_enabled =    opt.components in ["all"]
if lstm_enabled and not encoder_enabled:
    opt.g_dim = opt.channels * opt.patch_size * opt.patch_size

# ---------------- load the models  ----------------
lr = opt.lr
loss_type = opt.loss

if opt.model_path and model:
    # ---------------- load the trained model -------------
    if encoder_enabled:
        encoder = model['encoder']
        decoder = model['decoder']
    if lstm_enabled:
        frame_predictor = model['frame_predictor']

else:
    if opt.test:
        raise ValueError('Must specify model path if testing')

    # ---------------- initialize the new model -------------
    if encoder_enabled:
        print('Initializing encoder...')
        from models import dcgan_64, vgg_64, dcgan_32, dcgan_16, dcgan_8
        if opt.model == 'dcgan':
            if opt.patch_size == 64:
                encoder = dcgan_64.encoder(opt.g_dim, opt.channels)
                decoder = dcgan_64.decoder(opt.g_dim, opt.channels)
            elif opt.patch_size == 32:
                encoder = dcgan_32.encoder(opt.g_dim, opt.channels)
                decoder = dcgan_32.decoder(opt.g_dim, opt.channels)
            elif opt.patch_size == 16:
                encoder = dcgan_16.encoder(opt.g_dim, opt.channels)
                decoder = dcgan_16.decoder(opt.g_dim, opt.channels)
            elif opt.patch_size == 8:
                encoder = dcgan_8.encoder(opt.g_dim, opt.channels)
                decoder = dcgan_8.decoder(opt.g_dim, opt.channels)
            else:
                raise ValueError('Invalid patch size')
        else:
            encoder = vgg_64.encoder(opt.g_dim, opt.channels)
            decoder = vgg_64.decoder(opt.g_dim, opt.channels)

        encoder.apply(utils.init_weights)
        decoder.apply(utils.init_weights)

    if lstm_enabled:
        print('Importing LSTM models')
        import models.lstm as lstm_models
        print('Initializing LSTM...')
        frame_predictor = lstm_models.lstm(opt.g_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
        print('Applying LSTM weights...')
        frame_predictor.apply(utils.init_weights)

# ---------------- models tranferred to GPU ----------------
print("Moving models to CUDA")
if encoder_enabled:
    encoder.cuda()
    decoder.cuda()
if lstm_enabled:
    frame_predictor.cuda()

# ---------------- optimizers ----------------
if encoder_enabled:
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr = lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr = lr)
if lstm_enabled:
    frame_predictor_optimizer = torch.optim.Adam(frame_predictor.parameters(), lr = lr)


if gp_enabled:
    # ---------------- GP initialization ----------------------
    from models.gp_models import GPRegressionLayer1
    gp_layer = GPRegressionLayer1().cuda()#inputs
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=opt.g_dim).cuda()

    if opt.model_path:
        likelihood.load_state_dict(model['likelihood'])
        gp_layer.load_state_dict(model['gp_layer'])


    # ---------------- GP optimizer initialization ----------------------
    optimizer = torch.optim.Adam([{'params': gp_layer.parameters()}, {'params': likelihood.parameters()},], lr=0.002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5], gamma=0.1)

    # Our loss for GP object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_layer, num_data=opt.batch_size, combine_terms=True)

# --------- loss functions ------------------------------------

ssim = SSIM().cuda()
mse = nn.MSELoss().cuda()
l1 = nn.L1Loss().cuda()

if loss_type == "ssim":
    reconstruction_loss_func = lambda pred, gt: 1 - ssim(pred, gt)
elif loss_type == "l1":
    reconstruction_loss_func = l1
elif loss_type == "mse":
    reconstruction_loss_func = mse
elif loss_type.startswith("ssim_mse"):
    if loss_type.endswith("point01"):
        mse_weight = 0.01
    elif loss_type.endswith("point1"):
        mse_weight = 0.1
    elif loss_type.endswith("point001"):
        mse_weight = 0.001
    else:
        mse_weight = 1
    reconstruction_loss_func = lambda pred, gt:  mse_weight*mse(pred, gt) + 1 - ssim(pred, gt)
elif loss_type.startswith("ssim_l1"):
    if loss_type.endswith("point01"):
        l1_weight = 0.01
    elif loss_type.endswith("point1"):
        l1_weight = 0.1
    elif loss_type.endswith("point001"):
        l1_weight = 0.001
    else:
        l1_weight = 1
    reconstruction_loss_func = lambda pred, gt:  l1_weight*l1(pred, gt) + 1 - ssim(pred, gt)

else:
    raise ValueError(f"Loss type {loss_type} not recognized")

latent_loss_func = nn.MSELoss()
latent_loss_func.cuda()

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt.dataset, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

# training_batch_generator = torch.nn.DataParallel(training_batch_generator)

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt.dataset, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()

# # --------- training/evaluation funtions ------------------------------------
def run_batch(x, train: bool = True):  

    if encoder_enabled and train:
        encoder.zero_grad()
        decoder.zero_grad()
    
    if lstm_enabled:
        if train:
            frame_predictor.zero_grad()
        # initialize the hidden state.
        frame_predictor.hidden = frame_predictor.init_hidden()

    lstm_loss = 0
    latent_loss = 0
    gp_loss = 0
    max_ll = 0
    ae_loss = 0
    x_in = x[0]
    if not train:
        gen_seq = [x[0].detach()]
    for i in range(1, opt.n_past+opt.n_future):
        assert x_in.shape == (opt.batch_size, opt.channels, opt.patch_size, opt.patch_size), f"x_in.shape = {x_in.shape}"

        if encoder_enabled:
            # Encode past frame
            if opt.last_frame_skip or i < opt.n_past:   
                h, skip = encoder(x_in)
            else:
                h, _ = encoder(x_in)
        
        if encoder_enabled and lstm_enabled:
            h_pred = frame_predictor(h)  
            x_pred = decoder([h_pred, skip])  
        elif lstm_enabled:
            x_pred = frame_predictor(x_in)
            if x_pred.shape != x_in.shape:
                x_pred = x_pred.view(x_in.shape)                  

        if encoder_enabled and gp_enabled:
            # Make latent prediction with GP
            gp_pred = gp_layer(h.transpose(0,1).view(90,opt.batch_size,1)) #likelihood(gp_layer(h.transpose(0,1).view(90,opt.batch_size,1)))#
            x_pred_gp = decoder([gp_pred.mean.transpose(0,1), skip])


        # Add timestep to generated sequence
        if i < opt.n_past:
            x_in = x[i]
        else:
            x_in = x_pred

        if not train:
            gen_seq.append(x_in.detach())   

        # --------- loss functions ------------------------------------
        if encoder_enabled:
            h_target = encoder(x[i])[0]

        if encoder_enabled and lstm_enabled:
            latent_loss  += latent_loss_func(h_pred, h_target) # LSTM loss - how well LSTM predicts next encoding
            x_target_pred = decoder([h_target, skip])           # Decoded target encoding
            ae_loss += reconstruction_loss_func(x_target_pred, x[i])  # Encoder loss - how well the encoder encodes
        elif encoder_enabled:
            decoded = decoder([h_pred, skip])
            assert x[i-1].shape == decoded.shape
            ae_loss += reconstruction_loss_func(decoded, x[i-1])

        if lstm_enabled:
            lstm_loss += reconstruction_loss_func(x_pred, x[i])  # Encoder + LSTM loss - how well the encoder+LSTM predicts the next frame
            
        
        if encoder_enabled and gp_enabled:
            max_ll -= mll(gp_pred, h_target.transpose(0,1)).sum()      # GP Loss - how well GP predicts next encoding
            gp_loss += reconstruction_loss_func(x_pred_gp, x[i])             # Encoder + GP loss - how well the encoder+GP predicts the next frame
        
        torch.cuda.empty_cache()

    # --------- loss functions ------------------------------------
    encoder_weight = 100
    alpha = 1
    beta = 0.1
    loss = 0
    if encoder_enabled:
        loss += encoder_weight*ae_loss
    if lstm_enabled:
        loss += alpha*lstm_loss
    if encoder_enabled and lstm_enabled:
        loss += alpha*latent_loss
    if gp_enabled:
        loss += beta*gp_loss + beta*max_ll #+ kld*opt.beta

    if train:
        loss.backward()
        if encoder_enabled:
            encoder_optimizer.step()
            decoder_optimizer.step()
        if lstm_enabled:
            frame_predictor_optimizer.step()
        if gp_enabled:
            optimizer.step()

    to_log = {"Total": loss}
    if encoder_enabled:
        to_log[f"Encoder {loss_type}"] = ae_loss
    if lstm_enabled:
        to_log["LSTM"] = lstm_loss
    if encoder_enabled and lstm_enabled:
        to_log["Latent LSTM"] = latent_loss
    if gp_enabled:
        to_log["GP"] = gp_loss
        to_log["Latent GP"] = max_ll

    if train:
        return to_log

    assert len(x) == len(gen_seq), f"{len(x)} != {len(gen_seq)}"
    assert x[0].shape == gen_seq[0].shape
    return to_log, gen_seq


# --------- patching functions ------------------------------------
def generate_patches(x: List[torch.Tensor]) -> List[List[torch.Tensor]]:
    assert x[0].shape == (opt.batch_size, opt.channels, opt.image_width, opt.image_width), x.shape
    if opt.patch_size == opt.image_width:
        return [x]
    x_patches = []
    for j in range(0, opt.image_width, opt.patch_size):
        for k in range(0, opt.image_width, opt.patch_size):
            x_patch = [x_timestep[:, :, j:j+opt.patch_size, k:k+opt.patch_size] for x_timestep in x]
            assert x_patch[0].shape == (opt.batch_size, opt.channels, opt.patch_size, opt.patch_size), x_patch.shape
            assert len(x_patch) == len(x)
            x_patches.append(x_patch)
    return x_patches

def merge_patches(x_patches: List[torch.Tensor]) -> torch.Tensor:
    if len(x_patches) == 1:
        return x_patches[0]
    reconstructed_x = []
    for i in range(opt.n_past+opt.n_future):
        reconstructed_x_timestep = torch.zeros(opt.batch_size, opt.channels, opt.image_width, opt.image_width)
        for j in range(0, opt.image_width, opt.patch_size):
            for k in range(0, opt.image_width, opt.patch_size):
                patch_idx = k//opt.patch_size + j//opt.patch_size*(opt.image_width//opt.patch_size)
                reconstructed_x_timestep[:, :, j:j+opt.patch_size, k:k+opt.patch_size]  = x_patches[patch_idx][i]
        reconstructed_x.append(reconstructed_x_timestep)
    return torch.stack(reconstructed_x)


# --------- plotting funtions ------------------------------------
def plot(x, epoch, gen_seq):
    nsample = 5
    gt_seq = [x[i] for i in range(len(x))]

    assert len(gt_seq) == len(gen_seq[0]), f"{len(gt_seq)} != {len(gen_seq[0])}"
    assert gt_seq[0].shape == gen_seq[0][0].shape, f"{gt_seq[0].shape} != {gen_seq[0][0].shape}"

    # -------------- creating the GIFs ---------------------------
    to_plot = []
    gifs = [ [] for _ in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [gt_seq[t][i] for t in range(opt.n_eval)] 
        to_plot.append(row)

        if len(gen_seq) == 1:
            s_list = [0]
        else:
            # Finds best sequence (lowest loss)
            min_mse = 1e7
            for s in range(nsample):
                mse = 0
                for t in range(opt.n_eval):
                    mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
                if mse < min_mse:
                    min_mse = mse
                    min_idx = s

            s_list = [min_idx, 
                    np.random.randint(nsample), 
                    np.random.randint(nsample), 
                    np.random.randint(nsample), 
                    np.random.randint(nsample)]
        
        # Add to images
        for s in s_list:
            row = [gen_seq[s][t][i] for t in range(opt.n_eval)]
            to_plot.append(row)
        
        # Add to gifs
        for t in range(opt.n_eval):
            row = [gt_seq[t][i]]
            row += [gen_seq[s][t][i] for s in s_list]
            gifs[t].append(row)
    
    if opt.dataset == 'satellite':
        for i, row in enumerate(to_plot):
            for j, img in enumerate(row):
                img_np = img.cpu().numpy()
                img_viewable = train_data.for_viewing(img_np)
                img_tensor = torch.from_numpy(img_viewable)
                to_plot[i][j] = img_tensor
        
        for i, gif in enumerate(gifs):
            for j, row in enumerate(gif):
                for k, img in enumerate(row):
                    img_np = img.cpu().numpy()
                    img_viewable = train_data.for_viewing(img_np)
                    img_tensor = torch.from_numpy(img_viewable)
                    gifs[i][j][k] = img_tensor


    if opt.run_name:
        file_name = opt.run_name
    else:
        file_name = f"end2end_gp_ctrl_sample_{epoch}"

    if encoder_enabled and not lstm_enabled and not gp_enabled:
        file_name += "_autoencoders"

    img_path = home_dir / f'imgs/{opt.dataset}/{file_name}.png'
    img_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_of_images = utils.save_tensors_image(str(img_path), to_plot)
    print(f"Saving image to: {img_path}")
    return tensor_of_images

    # gif_path = home_dir / f'gifs/{opt.dataset}/{file_name}.gif'
    # gif_path.parent.mkdir(parents=True, exist_ok=True)
    # utils.save_gif(str(gif_path), gifs)
    # print(f"Saving images as gif: {gif_path}")
    

# --------- testing loop ------------------------------------
def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    metric_for_example = {}
    metric_for_example["mse"] = ((Y_true - Y_pred)**2).mean()
    metric_for_example["l1"] = np.abs(Y_true - Y_pred).mean()
    # Mean over batch, timesteps, width, height
    metric_for_example["mse_per_band"] = ((Y_true - Y_pred)**2).mean(axis=(0,1,3,4))

    if opt.dataset == 'satellite':
        rgb_index = train_data.rgb_index
        Y_true_img = np.moveaxis(Y_true[:,:, rgb_index], -3, -1).reshape(-1,opt.image_width,opt.image_width,opt.channels) 
        Y_pred_img = np.moveaxis(Y_pred[:,:, rgb_index], -3, -1).reshape(-1,opt.image_width,opt.image_width,opt.channels) 
    else:
        Y_true_img = Y_true
        Y_pred_img = Y_pred
    metric_for_example["ssim2"] = np.array([ssim2(
        Y_true_img[i],
        Y_pred_img[i],
        multichannel=True, 
        data_range=1, 
        gaussian_weights=True, 
        win_size=3, 
        sigma=1.5, 
        use_sample_covariance=False) for i in range(Y_true.shape[0])]).mean()

    return metric_for_example


def get_metrics_for_example(gt_seq: torch.Tensor, gen_seq: List[List], nsample=1):
    # Finds best sequence (lowest loss)
    min_mse = None
    for s in range(nsample):

        if opt.dataset == "satellite":
            Y_pred = train_data._unnormalize(gen_seq[s][opt.n_past: opt.n_eval].data.cpu().numpy())
            Y_true = train_data._unnormalize(gt_seq[opt.n_past: opt.n_eval].data.cpu().numpy())
        else:
            Y_pred = gen_seq[s][opt.n_past: opt.n_eval].data.cpu().numpy()
            Y_true = gt_seq[opt.n_past: opt.n_eval].data.cpu().numpy()

        metrics_for_example = compute_metrics(Y_true, Y_pred)
        
        if min_mse is None or  min_mse > metrics_for_example["mse"]:
            min_mse = metrics_for_example["mse"]
            lowest_metrics_for_example = metrics_for_example

    return lowest_metrics_for_example


# --------- training loop ------------------------------------
with gpytorch.settings.max_cg_iterations(45):

    for epoch in range(opt.niter):

        if not opt.test:
            if lstm_enabled:
                frame_predictor.train()
            if encoder_enabled:
                encoder.train()
                decoder.train()
            if gp_enabled:
                gp_layer.train()
                likelihood.train()
                scheduler.step()
            epoch_loss = 0
            for i in tqdm(range(len(train_loader)), leave=False, desc=f"Training epoch: {epoch}"):
                x = next(training_batch_generator)
                x_patches = generate_patches(x)
                step_loss_dict = defaultdict(lambda: 0)
                step_loss_dict["Epoch"] = epoch
                for x_patch in x_patches:
                    step_patch_loss_dict = run_batch(x_patch, train=True)#train(x_patch, epoch) 
                    for k,v in step_patch_loss_dict.items():
                        step_loss_dict[f"train_{k}"] += v
                    
                wandb.log(step_loss_dict)

        if lstm_enabled:
            frame_predictor.eval()
        if gp_enabled:
            gp_layer.eval()
            likelihood.eval()
        if encoder_enabled:
            encoder.eval()
            decoder.eval()
        
        test_size = len(test_loader)

        with torch.no_grad():
            log_dict = defaultdict(lambda: 0)
            log_dict["Epoch"] = epoch
            metrics_for_each_batch = []
            for sequence in tqdm(test_loader, leave=False, desc=f"Testing epoch {epoch}"):
                x = utils.normalize_data(opt.dataset, dtype, sequence)
                x_patches = generate_patches(x)
                gen_seq_per_patch = []
                for x_patch in x_patches:
                    step_patch_loss_dict, gen_seq_for_patch = run_batch(x_patch, train=False)
                    gen_seq_per_patch.append(torch.stack(gen_seq_for_patch))
                    for k,v in step_patch_loss_dict.items():
                        log_dict[f"test_{k}"] += v

                gen_seq = [merge_patches(gen_seq_per_patch)]
                assert len(gen_seq[0]) == len(x)
                assert gen_seq[0][0].shape == x[0].shape

                lowest_metrics_per_batch = get_metrics_for_example(sequence, gen_seq, nsample=1)
                metrics_for_each_batch.append(lowest_metrics_per_batch)

                # This is porbably not necessary
                torch.cuda.empty_cache()
            
            for k,v in log_dict.items():
                if k != "Epoch":
                    log_dict[k] /= test_size

            if not opt.test:
                print("--------------------------------------------------------------")
                print(f"Epoch: {epoch}")
                print("--------------------------------------------------------------")
            for metric_type in metrics_for_each_batch[0].keys():
                mean_metric = np.array([m[metric_type] for m in metrics_for_each_batch]).mean(axis=0)
                log_dict[metric_type] = mean_metric if type(mean_metric) is not np.ndarray else list(mean_metric)
                print(f"{metric_type}: {log_dict[metric_type]}")

            if opt.test:
                break
            else:
               wandb.log(log_dict, commit=False) 
            
            if epoch % 4 == 0:
                tensor_of_images = plot(x, epoch, gen_seq=gen_seq)
                wandb.log({"test_image": wandb.Image(tensor_of_images), "epoch": epoch}, commit=False)
                
                # save the model
                if opt.run_name:
                    model_name = opt.run_name
                else:
                    model_name = f'{opt.components}_{opt.dataset}_model_epoch_{epoch}'
                
                if encoder_enabled and not lstm_enabled and not gp_enabled:
                    model_name += '_autoencoder'
                model_path = home_dir / f'model_dump/{opt.dataset}/{model_name}.pth'
                model_path.parent.mkdir(parents=True, exist_ok=True)

                save_dict = {'opt': opt}
                if encoder_enabled:
                    save_dict['encoder'] = encoder
                    save_dict['decoder'] = decoder
                if lstm_enabled:
                    save_dict['frame_predictor'] = frame_predictor
                if gp_enabled:
                    save_dict['gp_layer'] = gp_layer
                    save_dict['likelihood'] = likelihood

                torch.save(save_dict, str(model_path))
                print(f"Saving model: {opt.dataset}/{model_name}.pth'")
                #artifact.add_file(str(model_path), name=f'{model_name}.pth')
                #run.log_artifact(artifact)

