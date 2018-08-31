import os
import argparse
import preprocess_text
import train_text

def preprocess_config():
    config = argparse.Namespace()
    config.trainfile = '/remote/bones/user/dspokoyn/vae-mode-collapse/yahoo_data/yahoo.train.txt'
    config.valfile = '/remote/bones/user/dspokoyn/vae-mode-collapse/yahoo_data/yahoo.valid.txt'
    config.testfile = '/remote/bones/user/dspokoyn/vae-mode-collapse/yahoo_data/yahoo.test.txt'
    config.outputfile = 'data/yahoo/yahoo'
    config.vocabsize = 70000
    # config.vocabsize = 11
    config.vocabminfreq = -1
    config.batchsize = 32
    # config.batchsize = 256
    config.seqlength = 200
    # config.seqlength = 20
    config.vocabfile = ''
    config.shuffle = 1
    return config
    #     python preprocess_text.py --trainfile data/yahoo/train.txt --valfile data/yahoo/val.txt
    # --testfile data/yahoo/test.txt --outputfile data/yahoo/yahoo

def train_config():
    config = argparse.Namespace()
    config.train_file = 'data/yahoo/yahoo-train.hdf5'
    config.val_file = 'data/yahoo/yahoo-val.hdf5'
    config.test_file = 'data/yahoo/yahoo-test.hdf5'
    config.train_from = ''
    config.gpu = 1
    # checkpoint_path model-path

    # # Model options
    config.latent_dim =32
    config.enc_word_dim =512
    config.enc_h_dim =1024
    config.enc_num_layers =1
    config.dec_word_dim =512
    config.dec_h_dim =1024
    config.dec_num_layers =1
    config.dec_dropout =0.5
    config.train_n2n =1
    # config.train_kl =1 only for savae

    # # Optimization options
    config.checkpoint_path ='baseline.pt'
    config.slurm =0
    config.warmup =10
    config.kl_start = 0.1
    # config.warmup =0
    config.num_epochs =100
    # config.min_epochs =15
    # config.num_epochs =3
    # config.min_epochs =1
    config.start_epoch =0
    config.svi_steps =20
    config.svi_lr1 =1
    config.svi_lr2 =1
    config.eps =1e-5
    # config.decay =0
    config.lr_decay = 0.5
    config.momentum =0.5
    config.lr =1
    config.optim = 'SGD'
    config.max_grad_norm =5
    config.svi_max_grad_norm =5
    config.seed =3435
    config.print_every=100
    config.test = 0

    return config

def autoreg_default(config):
    config.model = 'autoreg'
    config.warmup = 0
    config.kl_start = 0
    return config

def vae_default(config):
    config.model = 'vae'
    return config

def svi_default(config):
    config.model = 'svi'
    config.svi_steps = 20
    config.train_n2n = 0
    return config

def vae_svi_default(config):
    config.model = 'savae'
    config.svi_steps = 20
    config.train_n2n = 0
    config.train_kl = 0
    return config

def vae_svi_kl_default(config):
    config.model = 'savae'
    config.svi_steps = 20
    config.train_n2n = 0
    config.train_kl = 1
    return config

def sa_vae_default(config):
    config.model = 'savae'
    config.svi_steps = 20
    # config.svi_steps = 2 #small
    config.train_n2n = 1
    return config


# Autoregressive (i.e. language model): --model autoreg
# VAE: --model vae

# SVI: --model svi --svi_steps 20 --train_n2n 0

# VAE+SVI: --model savae --svi_steps 20 --train_n2n 0 --train_kl 0
# VAE+SVI+KL: --model savae --svi_steps 20 --train_n2n 0 --train_kl 1
# SA-VAE: --model savae --svi_steps 20 --train_n2n 1

# 100 epochs
# no min epochs
# no min lr ?
# save the best, and keep the logs
# IW for Vae
# Nothing for LM


if __name__ == '__main__':
    TRAIN = True
    prep_config = preprocess_config()
    preprocess_text.get_data(prep_config)
    args = train_config()
    args.dataset = 'yahoo'
    args.mode = 'autoreg'
    # args.mode = 'vae'
    # args.mode = 'svi'
    # args.mode = 'vae_svi'
    # args.mode = 'vae_svi_kl'
    # args.mode = 'sa_vae'
    if args.mode == 'autoreg':
        args =  autoreg_default(args)
    elif args.mode == 'vae':
        args =  vae_default(args)
    elif args.mode == 'svi':
        args =  svi_default(args)
    elif args.mode == 'vae_svi':
        args =  vae_svi_default(args)
    elif args.mode == 'vae_svi_kl':
        args =  vae_svi_kl_default(args)
    elif args.mode == 'sa_vae':
        args =  sa_vae_default(args)

    save_dir = "models/%s" % args.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "%s_%s_optim%s_kls%.1f_warm%d_%d" % \
            (args.mode, args.dataset, args.optim,
             args.kl_start, args.warmup, args.seed)

    save_path = os.path.join(save_dir, id_)

    args.checkpoint_path = save_path
    if TRAIN:
        train_text.main(args)
    else:
        args.test = 1
        args.train_from = args.checkpoint_path +'_' + '206.11166362842684.pt'
        train_text.main(args)
