from torch import nn
import einops
import networks
import models
import torch
import tools

to_np = lambda x: x.detach().cpu().numpy()


class Feeder:
    """Feeder that feeds sensor signals into the local module."""

    def __init__(self, key, feed_size):
        """Feeder has a key for envBuffer

        Args:
            envBuffer (EnvBuffer): EnvBuffer that will contain the observations from environments
            key (str): key for EnvBuffer. ex) key = "central"
        """
        self.key = key
        self.feed_size = feed_size
        self.buf = []
        self.isLeaf = True
        self.name = f"{key}-feeder"

    def feed(self, context=None):
        assert (
            len(self.buf) == 1
        ), f"The Feeder for [{self.key}] got {len(self.buf)} elements. It should be one element when feeding."
        return self.buf.pop()

    def put(self, value):
        assert (
            len(self.buf) == 0
        ), f"The Feeder for [{self.key}] got {len(self.buf)} elements. It should be zero element when putting."
        self.buf.append(value)


class LocalModule(nn.Module):

    def __init__(self, name="module", lowers=[], context_size=0, configs=None):
        """Initialize module

        Args:
            name (str, optional): name. Defaults to "module".
            lowers (list[LocalModule]): lower LocalModules. Defaults to [].
            context_size (int, optional): size of context input vector. If top module, it will be action size, otherwise 0, it will be state size.
            configs (dict, optional): configuration for module. Defaults to None.
        """
        super(LocalModule, self).__init__()
        self.configs = configs
        self.name = name
        self.isLeaf = False
        self._use_amp = True if configs.precision == 16 else False
        self.lowerModules = lowers
        if configs.dyn_discrete:
            state_size = configs.dyn_stoch * configs.dyn_discrete + configs.dyn_deter
        else:
            state_size = configs.dyn_stoch + configs.dyn_deter

        if context_size == 0:
            self.context_size = state_size
        else:
            self.context_size = context_size

        self.feed_size = state_size

        enc_input_size = 0
        for lower in lowers:
            enc_input_size += lower.feed_size
        enc_input_size *= configs.tmp_abs_factor
        self.step = 0

        self.encoder = networks.MLP(
            enc_input_size,
            None,
            configs.enc_num_layers,
            configs.enc_emb_size,
            configs.enc_act,
            configs.enc_norm,
            symlog_inputs=configs.enc_symlog_inputs,
            name=f"{self.name}-enc",
            device=configs.device,
        )

        self.mlp_shapes = {}
        for lower in lowers:
            self.mlp_shapes[lower.name] = lower.feed_size

        self.decoder = networks.MLP(
            state_size,
            self.mlp_shapes,
            configs.dec_num_layers,
            configs.dec_emb_size,
            configs.dec_act,
            configs.dec_norm,
            dist=configs.dec_dist,
            outscale=configs.dec_outscale,
            name=f"{self.name}-dec",
            device=configs.device,
        )

        self.dynamics = networks.RSSM(
            stoch=configs.dyn_stoch,
            deter=configs.dyn_deter,
            hidden=configs.dyn_hidden,
            rec_depth=configs.dyn_rec_depth,
            discrete=configs.dyn_discrete,
            act=configs.dyn_act,
            norm=configs.dyn_norm,
            mean_act=configs.dyn_mean_act,
            std_act=configs.dyn_std_act,
            min_std=configs.dyn_min_std,
            unimix_ratio=configs.unimix_ratio,
            initial=configs.initial,
            num_actions=self.context_size,
            embed=configs.enc_emb_size,
            device=configs.device,
        )

        self.optimizer = tools.Optimizer(
            self.name,
            self.parameters(),
            configs.lr,
            eps=configs.opt_eps,
            clip=configs.grad_clip,
            wd=configs.weight_decay,
            opt=configs.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )

    def init_layer(self):
        self.obs = []
        self.states = []
        self.recons = []
        self.priors = []
        self.posteriors = []
        batch_size = self.configs.batch_size
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
            state = self.run(context)
            states.append(state)
        states = torch.concat(states, dim=1).detach()
        return states

    def run(self, context=None):
        if context is not None:
            context = context.detach()

        if not hasattr(self, "prev_state"):
            self.init_layer()

        B, _ = context.shape

        prior = self.dynamics.img_step(self.prev_state, context)
        prior_state = self.dynamics.get_feat(prior)
        obs = {}
        for lower_module in self.lowerModules:
            obs[lower_module.name] = lower_module.feed(context=prior_state)
        flat_obs = torch.concat([obs[k] for k in self.mlp_shapes], dim=1)
        emb = self.encoder(flat_obs)
        if self.isImaginary:
            post = prior
        else:
            post = self.dynamics.obs_step(prior, emb)
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
        dyn_scale = self.configs.dyn_scale
        rep_scale = self.configs.rep_scale

        post = self.list_to_dict(self.posteriors)
        prior = self.list_to_dict(self.priors)
        obs = self.list_to_dict(self.obs)

        with torch.cuda.amp.autocast(self._use_amp):
            kl_loss, kl_value, _, _ = self.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )
            kl_loss = torch.mean(kl_loss)
            states = torch.stack(self.states, dim=1).to(self.configs.device)
            recon = self.decoder(states)
            # obs = torch.stack(self.obs, dim=1).to(self.configs.device)
            recon_loss = 0
            nll = {}
            for name, pred in recon.items():
                nll[name] = -pred.log_prob(obs[name])
                recon_loss += nll[name].mean()
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


class HierarchicalWorldModel(nn.Module):

    def __init__(self, configs):
        super(HierarchicalWorldModel, self).__init__()
        modules = {}
        self.configs = configs
        self._use_amp = True if configs.precision == 16 else False
        self.feeder_keys = configs.feeder_keys

        # Build feeders
        self.feeders = {}
        for key, size in self.feeder_keys.items():
            self.feeders[key] = Feeder(key, size)
            name = f"{key}_module"
            localModule = LocalModule(name, lowers=[self.feeders[key]], configs=configs)
            modules[name] = localModule

        modules["assoc_module"] = LocalModule(
            "assoc_module",
            lowers=[modules["central_module"], modules["loc_module"]],
            configs=configs,
        )

        modules["top_module"] = LocalModule(
            "top_module",
            lowers=[modules["peripheral_module"], modules["assoc_module"]],
            context_size=configs.num_actions,
            configs=configs,
        )
        self.local_modules = nn.ModuleDict(modules)
        print(self.local_modules.keys())

    def get_init_state(self):
        return self.local_modules["top_module"].init_layer()

    def get_feat(self, state):
        feat = self.local_modules["top_module"].dynamics.get_feat(state)
        detached_feat = feat.detach().clone()
        return detached_feat

    def load_feeders(self, obs):
        for key, feeder in self.feeders.items():
            feeder.put(obs[key])

    def step(self, action, obs):
        self.load_feeders(obs)
        return self.local_modules["top_module"].feed(context=action)

    def train(self):
        metrics = {}
        for name, module in self.local_modules.items():
            met = module.update()
            metrics.update(met)
        return metrics

    def init(self):
        for name, module in self.local_modules.items():
            module.init_layer()
        return self.local_modules["top_module"].prev_state


class CognitiveArchitecture(nn.Module):
    def __init__(self, configs):
        super(CognitiveArchitecture, self).__init__()
        self.configs = configs
        self.wm = HierarchicalWorldModel(configs)
        self.behavior = models.ACBehavior(configs)

    def get_action(self, feat):
        actor = self.behavior.actor(feat)
        action = actor.sample()
        logprob = actor.log_prob(action)
        actor_ent = actor.entropy()
        return action, logprob, actor_ent

    def wm_step(self, action, obs):
        return self.wm.step(action, obs)

    def train(self, batch):
        # Train Hierarchical Worldmodel
        return self.wm.train()
