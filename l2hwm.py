from torch import nn
import torch
import networks
import tools
import numpy as np
import einops
import torch.nn.functional as F
from torch import distributions as torchd
from torchview import draw_graph
import graphviz
import math

# graphviz.set_jupyter_format('png')
from torchviz import make_dot, make_dot_from_trace

to_np = lambda x: x.detach().cpu().numpy()


class L2HWM(nn.Module):

    def __init__(self, configs):
        super(L2HWM, self).__init__()
        self.step = 0
        self.layers = nn.ModuleList()
        self.debug = configs.debug
        self.bottom_factor = configs.tmp_abs_factor ** (configs.levels - 1)
        self.configs = configs
        self._use_amp = True if configs.precision == 16 else False

        for level in range(configs.levels):
            if level == 0:
                isBottom = True
                lowerLayer = None
            else:
                isBottom = False
                lowerLayer = layer

            if level == configs.levels - 1:
                isTop = True
            else:
                isTop = False

            layer = LocalLayer(
                name=f"{level}",
                isTop=isTop,
                isBottom=isBottom,
                lower=lowerLayer,
                configs=configs,
            )

            self.layers.append(layer)

            if isBottom:
                self.bottomLayer = layer
            if isTop:
                self.topLayer = layer

    def init_layers(self, x):
        videoFeeder = VideoFeeder(x)
        self.bottomLayer.lowerLayer = videoFeeder

        B, T, _, _, _ = x.shape
        top_trim = math.floor(T / self.bottom_factor)

        for layer in self.layers:
            layer.init_layer(B)
        return top_trim

    def pred(self, x, top_ctx):
        top_trim = self.init_layers(x)

        B, T, _, _, _ = x.shape
        empty_context = torch.empty(B, 0).to(self.configs.device)
        for i in range(top_ctx):
            self.topLayer.single_run(context=empty_context)

        self.bottomLayer.isImaginary = True

        for i in range(self.configs.open_loop_ctx, top_trim):
            self.topLayer.single_run(context=empty_context)

        videos = self.decode_videos()
        new_videos = []
        for video in videos:
            trim_video = video[:, :T]
            new_videos.append(trim_video)
        return new_videos

    def video_pred(self, x):
        videos = self.pred(x, self.configs.open_loop_ctx)
        num_initial = self.configs.open_loop_ctx * self.bottom_factor
        _, T, _, _, _ = videos[0].shape
        gt = x[:, :T, :, :, :]
        recon_loss_list = []
        for level in range(self.configs.levels):
            mse = F.mse_loss(videos[level], gt)
            mse = to_np(mse)
            recon_loss_list.append(mse)

        num_gifs = 6
        videos = torch.cat(videos, 2)
        initial_decode = videos[:, :num_initial]
        open_loop_decode = videos[:, num_initial:]
        initial_decode = 1 - initial_decode
        model_video = torch.cat([initial_decode, open_loop_decode], 1)[:num_gifs]
        return_video = torch.cat([gt[:num_gifs], model_video], 2)
        return to_np(return_video), recon_loss_list

    def decode_videos(self):
        videos = []
        for layer in self.layers:
            states = torch.stack(layer.states, dim=1)
            recon = layer.decode_video(states)
            videos.append(recon)
        return videos

    def local_train(self, x):
        top_trim = self.init_layers(x)
        metrics = {}
        B, _, _, _, _ = x.shape
        empty_context = torch.empty(B, 0).to(self.configs.device)
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                for _ in range(top_trim):
                    self.topLayer.single_run(context=empty_context)
            for layer in self.layers:
                met = layer.update()
                metrics.update(met)
        return metrics


class LocalLayer(nn.Module):

    def __init__(
        self, name="layer", isTop=False, isBottom=False, lower=None, configs=None
    ):
        super(LocalLayer, self).__init__()
        self.configs = configs
        self.name = name
        self._use_amp = True if configs.precision == 16 else False
        self.lowerLayer = lower
        if configs.dyn_discrete:
            state_size = (
                configs.dyn_stoch_size * configs.dyn_discrete + configs.dyn_deter_size
            )
        else:
            state_size = configs.dyn_stoch_size + configs.dyn_deter_size
        enc_input_size = state_size * configs.tmp_abs_factor
        self.isBottom = isBottom
        self.step = 0

        if isTop:
            context_size = 0
        else:
            context_size = state_size

        self.dynamics = networks.RSSM(
            stoch=configs.dyn_stoch_size,
            deter=configs.dyn_deter_size,
            hidden=configs.dyn_hidden_size,
            context=context_size,
            layers_input=configs.dyn_input_layers,
            layers_output=configs.dyn_output_layers,
            discrete=configs.dyn_discrete,
            act=configs.act,
            mean_act=configs.dyn_mean_act,
            std_act=configs.dyn_std_act,
            min_std=configs.dyn_min_stddev,
            unimix_ratio=configs.unimix_ratio,
            initial=configs.initial,
            num_actions=0,
            embed=configs.enc_emb_size,
            device=configs.device,
        )

        if isBottom:
            self.encoder = networks.ConvEncoder(
                configs.enc_emb_size,
                channels=configs.channels,
                depth=configs.cnn_depth,
                act=getattr(nn, configs.act),
                kernels=configs.encoder_kernels,
            )
            self.decoder = networks.ConvDecoder(
                state_size,
                depth=configs.cnn_depth,
                act=getattr(nn, configs.decoder_act),
                shape=(configs.channels, *configs.img_size),
                kernels=configs.decoder_kernels,
                thin=configs.decoder_thin,
            )
        else:
            self.encoder = networks.MLPEncoder(
                enc_input_size,
                configs.enc_emb_size,
                n_layers=3,
                hidden_size=configs.enc_emb_size,
                activation="elu",
                batch_norm=True,
            )
            self.decoder = networks.MLPDecoder(
                state_size,
                enc_input_size,
                n_layers=3,
                hidden_size=state_size,
                activation="elu",
                batch_norm=True,
            )
        self.optimizer = tools.Optimizer(
            name,
            self.parameters(),
            configs.lr,
            eps=configs.eps,
            clip=configs.clip_grad_norm_by,
            wd=configs.weight_decay,
            opt=configs.optimizer,
            use_amp=self._use_amp,
        )

    def init_layer(self, batch_size):
        self.obs = []
        self.states = []
        self.recons = []
        self.priors = []
        self.posteriors = []
        self.prev_state = self.dynamics.initial(batch_size)
        self.isImaginary = False

    # Feed isolates between layer
    # It gets context from current layer.
    # Inside the function, the context is detached.
    # It returns obs which is already detached.
    # If imagine is true, it will return prior state without observing
    # Else, it wrill return posterior state based on observation
    def feed(self, context=None):
        states = []
        for i in range(self.configs.tmp_abs_factor):
            state = self.single_run(context)
            states.append(state)
        states = torch.concat(states, dim=1).detach()
        return states

    def single_run(self, context=None):
        if context is not None:
            context = context.detach()

        B, _ = context.shape
        prev_action = torch.empty(B, 0).to(self.configs.device)

        prior = self.dynamics.img_step(self.prev_state, context, prev_action)
        # prior_state = torch.cat((prior["deter"], prior["stoch"]), dim=1)
        prior_state = self.dynamics.get_feat(prior)
        obs = self.lowerLayer.feed(context=prior_state)
        emb = self.encoder(obs)
        if self.isImaginary:
            post = prior
        else:
            post, _ = self.dynamics.obs_step(self.prev_state, context, prev_action, emb)
        self.priors.append(prior)
        self.posteriors.append(post)
        state = self.dynamics.get_feat(post)
        self.obs.append(obs)
        self.states.append(state)
        self.prev_state = post
        return state

    def list_to_dict(self, x):
        out_dict = {}
        for key in x[0]:
            values = [item[key] for item in x]
            out_dict[key] = torch.stack(values, dim=1)
        return out_dict

    def update(self):
        kl_free = self.configs.kl_free
        dyn_scale = self.configs.kl_dyn_scale
        rep_scale = self.configs.kl_rep_scale

        post = self.list_to_dict(self.posteriors)
        prior = self.list_to_dict(self.priors)

        with torch.cuda.amp.autocast(self._use_amp):
            kl_loss, kl_value = self.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )
            kl_loss = torch.mean(kl_loss)
            states = torch.stack(self.states, dim=1).to(self.configs.device)
            recon = self.decoder(states)
            obs = torch.stack(self.obs, dim=1).to(self.configs.device)
            nll = -recon.log_prob(obs)
            recon_loss = nll.mean()
            loss = recon_loss + kl_loss

        metrics = {}
        metrics[f"recon_loss_{self.name}"] = to_np(recon_loss)
        metrics[f"kl_loss_{self.name}"] = to_np(kl_loss)
        metrics[f"loss_{self.name}"] = to_np(loss)
        metrics[f"grad_norm_{self.name}"] = self.optimizer(loss, self.parameters())
        metrics[f"kl_{self.name}"] = to_np(torch.mean(kl_value))
        sum_entropy = torch.sum(self.dynamics.get_dist(prior).entropy(), dim=1)
        mean_entropy = torch.mean(sum_entropy)
        metrics[f"prior_ent_{self.name}"] = to_np(mean_entropy)
        sum_entropy = torch.sum(self.dynamics.get_dist(post).entropy(), dim=1)
        mean_entropy = torch.mean(sum_entropy)
        metrics[f"posterior_ent_{self.name}"] = to_np(mean_entropy)
        self.step += 1
        return metrics

    def decode_video(self, states):
        recon = self.decoder(states).mode()
        layer = self
        while layer.isBottom is not True:
            recon = einops.rearrange(
                recon, "b t (f c) -> b (t f) c", f=self.configs.tmp_abs_factor
            )
            recon = layer.lowerLayer.decoder(recon).mode()
            layer = layer.lowerLayer
        recon = layer.lowerLayer.decoder(recon)
        return recon


class VideoFeeder:

    def __init__(self, data):
        self.count = 0
        self.data = data
        self.data_size = data.shape[1]

    def feed(self, context=None):
        frame = self.data[:, self.count]
        frame = self.encode(frame)
        self.count += 1
        self.count = self.count % self.data_size
        return frame

    def encode(self, obs):
        # obs = obs.clone()
        obs = obs - 0.5
        obs = obs * 2.0
        return obs

    def decoder(self, obs):
        obs = obs / 2.0 + 0.5
        # obs = obs + 0.5
        return obs
