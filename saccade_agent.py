import torch
import tools
from torch import nn
import models
import os


class SaccadeAgent(nn.Module):
    def __init__(self, obs_space, act_space, config, logger):
        super(SaccadeAgent, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._metrics = {}
        # this is update step
        self._step = logger.step
        self._update_count = 0
        self._wm = models.WorldModel(obs_space, act_space, self._step, config).to(
            config.device
        )
        if config.behavior == 'ac':
            self.behavior = models.ACBehavior(config).to(config.device)
        elif config.behavior == 'random':
            self.behavior = models.SaccadeRandomBehavior(config, act_space).to(config.device)
        else:
            raise NotImplementedError(config.behavior)

        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self.behavior = torch.compile(self.behavior)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl, mse = self._wm.saccade_video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                    self._logger.scalar("Sac_MSE", mse)
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def get_init_wm_state(self):
        init_state = self._wm.dynamics.initial(self._config.envs)
        return init_state
    
    def saccade_evaluation(self, batch):
        return self._wm.saccade_video_evaluation(batch)


    def get_action(self, state):
        feat = state['feat'].detach()
        actor = self.behavior.actor(feat)
        action = actor.sample()
        logprob = actor.log_prob(action)
        actor_ent = actor.entropy()
        return action, logprob, actor_ent

    def estimate_state(self, state, action, obs):
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        post, prior = self._wm.dynamics.obs_step(state, action, embed, obs["is_first"])
        post['feat'] = self._wm.dynamics.get_feat(post)
        prior['feat'] = self._wm.dynamics.get_feat(prior)
        return post, prior

    def calculate_reward(self, batch, method="prediction"):
        recon = self._wm.heads["decoder"](batch["prior"]['feat'])[
            "central"
        ].mode()
        reward = 0.5 * (recon - batch["obs"]["central"]) ** 2
        return torch.mean(reward, 2)




    def saccade_train(self, batch):
        metrics = {}
        mets = self._wm.saccade_train(batch)
        metrics.update(mets)
        mets = self.behavior.saccade_train(batch)
        metrics.update(mets)
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

