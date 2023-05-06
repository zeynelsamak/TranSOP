from ast import arg
import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .layers import fuse_img_clinic, normalisation, activation, weights_init, ChannelSpatialSELayer3D2


class Predictor(torch.nn.Module):
    def __init__(self, indim, layer=4, meddim=None, outdim=None, usebn=True, lastusebn=False, usenoise=False, classification=None, drop=0.3):
        super(Predictor, self).__init__()
        self.usenoise = usenoise
        self.classification = classification
        if usenoise:
            indim = indim + indim // 2
        assert layer >= 1
        if meddim is None:
            meddim = indim
        if outdim is None:
            outdim = indim
        models = []
        for _ in range(layer - 1):
            models.append(nn.Linear(indim, meddim))
            if usebn:
                models.append(nn.BatchNorm1d(meddim))
            models.append(nn.ReLU(inplace=True))
            indim = meddim
        models.append(nn.Dropout(drop))
        models.append(nn.Linear(meddim, outdim))
        if lastusebn:
            models.append(nn.BatchNorm1d(outdim))
        self.model = nn.Sequential(*models)
        if self.classification:
            self.cls_layer = nn.Linear(outdim, self.classification)
    def forward(self, x):
        if self.usenoise:
            x = torch.cat(
                [x, torch.randn(x.size(0), x.size(1) // 2).to(x.device)], dim=1)
        x = self.model(x)
        if self.classification:
            out = self.cls_layer(x)
            return out, x
        return x
class Generator(nn.Module):
    def __init__(self, in_channel: int = 512, clin_size=0):
        super(Generator, self).__init__()
        base_feat = in_channel // 8
        self.in_channel = in_channel
        self.d = nn.Linear(1024 + clin_size, base_feat * 8 * 2 * 12 * 8)
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose3d(in_channel, base_feat * 4, 5, stride=2, padding=2, output_padding=1),
                                      normalisation(base_feat * 4),
                                      activation())
        self.tp_conv2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                      nn.Conv3d(base_feat * 4, base_feat * 2, 3, padding=1, stride=1),
                                      normalisation(base_feat * 2),
                                      activation())
        self.tp_conv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                      nn.Conv3d(base_feat * 2, base_feat, 3, padding=1, stride=1),
                                      normalisation(base_feat),
                                      activation())
        self.tp_conv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                      nn.Conv3d(base_feat, base_feat, 3, padding=1, stride=1),
                                      nn.Conv3d(base_feat, 1, 1),
                                      nn.Tanh()
                                      )
        for m in self.modules():
            weights_init(m)
    def forward(self, noise):
        if noise.ndim != 5:
            noise = self.d(noise)
            noise = noise.view(-1, self.in_channel, 2, 12, 8)
        h = self.tp_conv1(noise)
        h = self.tp_conv2(h)
        h = self.tp_conv3(h)
        h = self.tp_conv5(h)
        return h



class MultiSwinTrans(nn.Module):
    def __init__(self, args, channel=512, out_class=2, clin_size=2, dis_type='rec', drop=0.3, attention=False, follow=0, class_mode='cat'):
        super().__init__()

        #follow 0 only baseline, follow 1 baseline + f24h, follow 2 bl+f1w, follow 3 bl+f24h+f1w

        self.follow = follow
        self.clin_size = clin_size
        self.attention = attention
        in_ch = [768]

        patch_size = ensure_tuple_rep(args.patch_size, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.pret_model = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims
            )


        clin_feats = 768
        self.clin_fc = nn.Linear(self.clin_size, clin_feats)

        dim = 768
        in_ch.append(clin_feats)
        print(in_ch)
        
        self.classify = fuse_img_clinic(in_ch, out_ch=512, out_class=2, dropout=0.3, mode=class_mode)
        
        self.follow = False
        if args.follow_time==2 or args.follow_time==1:
            self.follow = True
            self.clin2dec_size = 2
            if 'clin2dec' in args.trial:
                self.clin2dec_size = clin_feats
            self.pred_1W = Predictor(dim+self.clin2dec_size, layer=3, meddim=1024, outdim=1024)
            self.decoder = Generator(in_channel= channel, clin_size=0)

    def forward(self, x, c):
        feat_bl = self.pret_model(x)[4]
        feat_bl = rearrange(feat_bl, "n c h w d -> n 1 (c h w d)")
        feat_clin = self.clin_fc(c[:, :self.clin_size])
        out = self.classify([feat_bl.squeeze(1), feat_clin])
        
        if self.follow:
            feat_1w = self.pred_1W(torch.cat([feat_bl.squeeze(1), feat_clin if self.clin2dec_size > 3 else c[:, :self.clin2dec_size]],1))
            vol_1w = self.decoder(feat_1w)

            return vol_1w, out

        return out

class MultiViTrans(nn.Module):
    def __init__(self, args, channel=512, out_class=2, clin_size=2, dis_type='rec', drop=0.3, attention=False, follow=0, class_mode='cat'):
        super().__init__()

        #follow 0 only baseline, follow 1 baseline + f24h, follow 2 bl+f1w, follow 3 bl+f24h+f1w

        self.follow = follow
        self.clin_size = clin_size
        self.attention = attention
        in_ch = [768]

        patch_size = ensure_tuple_rep(args.patch_size, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.pret_model = ViT(
            in_channels=args.in_channels,
            img_size= args.img_size,
            patch_size= patch_size,
            hidden_size= 768,
            mlp_dim= 3072,
            num_layers= 12,
            num_heads= 12,
            pos_embed= "conv",
            classification= True,
            num_classes= 768,
            dropout_rate = args.dropout_path_rate,
            spatial_dims = args.spatial_dims,
        )


        clin_feats = 768
        self.clin_fc = nn.Linear(self.clin_size, clin_feats)

        dim = 768
        in_ch.append(clin_feats)
        print(in_ch)
        self.classify = fuse_img_clinic(in_ch, out_ch=512, out_class=2, dropout=0.3, mode=class_mode)

        
        self.follow = False
        if args.follow_time==2 or args.follow_time==1:
            self.follow = True
            self.clin2dec_size = 2
            if 'clin2dec' in args.trial:
                self.clin2dec_size = clin_feats
            self.pred_1W = Predictor(dim+self.clin2dec_size, layer=3, meddim=1024, outdim=1024)
            self.decoder = Generator(in_channel= channel, clin_size=0)

    def forward(self, x, c):
        feat_bl, hidden_out = self.pret_model(x)
        feat_clin = self.clin_fc(c[:, :self.clin_size])
        out = self.classify([feat_bl, feat_clin])

        if self.follow:
            feat_1w = self.pred_1W(torch.cat([feat_bl, feat_clin if self.clin2dec_size > 3 else c[:, :self.clin2dec_size]],1))
            vol_1w = self.decoder(feat_1w)

            return vol_1w, out

        return out

class MultiViTransConv(nn.Module):
    def __init__(self, args, channel=512, out_class=2, clin_size=2, dis_type='rec', drop=0.3, attention=False, follow=0, class_mode='cat'):
        super().__init__()

        #follow 0 only baseline, follow 1 baseline + f24h, follow 2 bl+f1w, follow 3 bl+f24h+f1w

        self.follow = follow
        self.clin_size = clin_size
        self.attention = attention
        in_ch = [768]
        base_feat = channel//4
        self.conv = nn.Sequential(
                    nn.Conv3d(1, base_feat, 3, padding=1, stride=1),
                    activation('lrelu'),
                    nn.Conv3d(base_feat, base_feat*2, 3, padding=1, stride=2),
                    normalisation(base_feat*2),
                    activation('lrelu'),
                    nn.Conv3d(base_feat*2, base_feat*4, 3, padding=1, stride=2),
                    normalisation(base_feat*4),
                    activation('lrelu')
                    )

        if self.attention == 1:
            self.csSE = ChannelSpatialSELayer3D2(channel, channel)

        patch_size = ensure_tuple_rep(args.patch_size, args.spatial_dims)
        self.pret_model = ViT(
            in_channels= channel , #args.in_channels,
            img_size=  (8, 48, 32) , #args.img_size,
            patch_size= patch_size,
            hidden_size= 384,
            mlp_dim= 1536,
            num_layers= 12,
            num_heads= 12,
            pos_embed= "conv",
            classification= True,
            num_classes= 768,
            dropout_rate = args.dropout_path_rate,
            spatial_dims = args.spatial_dims,
        )

        clin_feats = 768
        self.clin_fc = nn.Linear(self.clin_size, clin_feats)

        dim = 768
        in_ch.append(clin_feats)
        print(in_ch)
        self.classify = fuse_img_clinic(in_ch, out_ch=512, out_class=2, dropout=0.3, mode=class_mode)

        
        self.follow = False
        if args.follow_time==2:
            self.follow = True
            self.clin2dec_size = 2
            if 'clin2dec' in args.trial:
                self.clin2dec_size = clin_feats
            self.pred_1W = Predictor(dim+self.clin2dec_size, layer=3, meddim=1024, outdim=1024)
            self.decoder = Generator(in_channel= channel, clin_size=0)

    def forward(self, x, c):

        x = self.conv(x)
        if self.attention:
            x = self.csSE(x)
        feat_bl, hidden_out = self.pret_model(x)
        feat_clin = self.clin_fc(c[:, :self.clin_size])
        out = self.classify([feat_bl, feat_clin])

        if self.follow:
            feat_1w = self.pred_1W(torch.cat([feat_bl, feat_clin.detach() if self.clin2dec_size > 3 else c[:, :self.clin2dec_size]],1))
            vol_1w = self.decoder(feat_1w)

            return vol_1w, out

        return out
