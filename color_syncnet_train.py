from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio
import io
import torch
from torch import nn
from torch import optim
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from opencv_transforms import transforms
from torchvision import transforms as torch_transforms
from glob import glob
import os, random, cv2, argparse
from hparams import hparams, get_image_list

from multiprocessing import Manager



parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split, shared_dict, writer):
        print(args.data_root)
        self.all_videos = get_image_list(args.data_root, split)
        self.shared_dict = shared_dict
        self.writer = writer
        self.desc = split
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, str(frame_id).zfill(7) + ".jpg")
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            #print(img_names)
            if len(img_names) <= 3 * syncnet_T:
               
                continue
            img_name = random.choice(img_names)
            i = img_names.index(img_name)
            
            if len(img_names) > 30:
                if(random.uniform(0, 1) > 0.5):
                    j = random.choice([ele for ele in range(i, i+100) if ele != i])
                else:
                    j = random.choice([ele for ele in range(i-100, i) if ele != i])
            else:
                if(random.uniform(0, 1) > 0.5):
                    j = random.choice([ele for ele in range(i, i+5) if ele != i])
                else:
                    j = random.choice([ele for ele in range(i-5, i) if ele != i])
                    
            if j >= len(img_names):
                j=len(img_names) -1
                
            if j < 0:
                if(i == 0):
                    j = i+5
                else:
                    j = 0
                
            #print("wrong_img_idx:{}, image_name_idx:{}".format(j,i))
            
                    
            wrong_img_name = img_names[j]
            
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                
                continue

            window = []
            all_read = True
            augmentation_transform = transforms.Compose([
                 transforms.RandomHorizontalFlip(p=1.0)
                 #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
             ])

            should_flip = False;
            if(random.random() <=0.50):
                should_flip = True;

            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                if(should_flip == True):
                    img = augmentation_transform(img)

                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    
                    break

                window.append(img)
                
            #for i, w in enumerate(window):
                
                #self.writer.add_image(self.desc+'-'+str(i), cv2.imencode('.jpg', w)[1].tostring())

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")
                if wavpath not in self.shared_dict:
                    wav = audio.load_wav(wavpath, hparams.sample_rate)
                    orig_mel = audio.melspectrogram(wav).T
                    self.shared_dict[wavpath] = orig_mel
                else:
                    orig_mel = self.shared_dict[wavpath]
               
            except Exception as e:
              
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            
           
            return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,scheduler,writer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    
    print('global_step: {}'.format(global_step))
    print('global_epoch: {}'.format(global_epoch))
    print('nepochs: {}'.format(nepochs))
    
    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()
            global_step += 1
            print('global_step_change: {}'.format(global_step))
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir, writer)
                #scheduler.step()

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
            
            writer.add_scalar('running Loss: ', (running_loss / (step + 1)), global_step)

        global_epoch += 1
        
        
        print('Step:{0} | lr: {1} | Epochs:{2}'.format(step + 1, optimizer.param_groups[0]['lr'], global_epoch))
        writer.add_scalar('Step_LR: ',(optimizer.param_groups[0]['lr']), global_step)
        
        
        

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, writer):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)
        writer.add_scalar('averaged loss: ', (averaged_loss), global_step)

        return

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('experiment15/SynNet_logs')
    
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    manager = Manager()
    shared_dict = manager.dict()
    # Dataset and Dataloader setup
    train_dataset = Dataset('train', shared_dict, writer)
    test_dataset = Dataset('val', shared_dict, writer)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr, betas=(0.5, 0.999))
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,scheduler,writer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
    writer.flush()
    writer.close()