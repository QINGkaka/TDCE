"""
Based on https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
and https://github.com/ehoogeboom/multinomial_diffusion
"""

import torch.nn.functional as F
import torch
import math

import numpy as np
from .utils import *

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianMultinomialDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            denoise_fn,
            num_timesteps=1000,
            gaussian_loss_type='mse',
            gaussian_parametrization='eps',
            multinomial_loss_type='vb_stochastic',
            parametrization='x0',
            scheduler='cosine',
            device=torch.device('cpu'),
            # TDCE参数：Gumbel-Softmax支持（TDCE始终使用Gumbel-Softmax处理分类特征）
            tau_init=1.0,             # 初始温度
            tau_final=0.3,            # 最终温度
            tau_schedule='anneal'     # 温度调度策略：'anneal'或'fixed'
        ):

        super(GaussianMultinomialDiffusion, self).__init__()
        assert multinomial_loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if multinomial_loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device)

        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        alphas = 1. - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        ).float().to(device)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas.numpy())
            / (1.0 - alphas_cumprod)
        ).float().to(device)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device))

        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))
        
        # TDCE新增：Gumbel-Softmax相关参数
        # TDCE always uses Gumbel-Softmax for categorical features
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.tau_schedule = tau_schedule
        
        # 注册betas到buffer（用于Gumbel-Softmax前向扩散）
        self.register_buffer('betas', betas.float().to(device))
    
    # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x.device), (1. - self.alphas)[1:]], dim=0)
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)


        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
            
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], "out_mean": out["mean"], "true_mean": true_mean}
    
    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
    
    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        terms = {}
        if self.gaussian_loss_type == 'mse':
            terms["loss"] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]


        return terms['loss']
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def gaussian_p_sample(
        self,
        model_out,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        # TDCE新增参数：分类器引导
        classifier=None,
        target_y=None,
        x_original=None,
        x_cat_gumbel=None,
        immutable_mask=None,
        lambda_guidance=1.0,
        debug_mode=False  # 调试模式标志
    ):
        """
        TDCE: 引导的高斯反向采样（连续特征）
        
        公式：μ_guided = μ_θ + Σ_θ · ||μ_θ|| · g_guided
        
        Args:
            model_out: 模型输出
            x: 当前时间步的连续特征
            t: 时间步
            clip_denoised: 是否裁剪去噪结果
            denoised_fn: 去噪函数
            model_kwargs: 模型额外参数
            classifier: 分类器（用于引导）
            target_y: 目标标签
            x_original: 原始样本（用于距离约束）
            x_cat_gumbel: 当前时间步的分类特征（Gumbel-Softmax向量，用于分类器输入）
            immutable_mask: 不可变特征掩码
            lambda_guidance: 引导权重
        
        Returns:
            {"sample": sample, "pred_xstart": out["pred_xstart"]}
        """
        # TDCE: 如果有分类器引导，需要确保输入设置了requires_grad
        if classifier is not None and target_y is not None:
            # 确保x和model_out设置了requires_grad=True
            if not x.requires_grad:
                x = x.clone().detach().requires_grad_(True)
            if not model_out.requires_grad:
                model_out = model_out.clone().detach().requires_grad_(True)
        
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        mu_theta = out["mean"]
        sigma_theta = out["variance"]
        pred_xstart = out["pred_xstart"]
        
        # TDCE: 如果有分类器引导，添加梯度引导项
        if classifier is not None and target_y is not None:
            from .classifier_guidance import (
                compute_classifier_gradient_split,
                compute_distance_gradient,
                compute_guided_gradient
            )
            
            # pred_xstart应该已经从model_out继承了梯度
            # 但如果需要，我们可以重新设置requires_grad
            if not pred_xstart.requires_grad:
                pred_xstart = pred_xstart.clone().detach().requires_grad_(True)
            
            # 1. 计算分类器梯度（只对数值特征）
            # 注意：分类器需要同时看到数值特征和分类特征（如果有）
            if x_cat_gumbel is not None:
                # 有分类特征，需要组合输入
                g_classifier_num, _ = compute_classifier_gradient_split(
                    classifier,
                    pred_xstart,  # 使用预测的x0
                    x_cat_gumbel,
                    target_y,
                    self.num_numerical_features,
                    list(self.num_classes),
                    x.device
                )
            else:
                # 只有数值特征
                # 简化处理：直接计算数值特征的梯度
                # 这里需要分类器只接受数值特征，或者我们需要创建一个只包含数值特征的分类器输入
                # 为了简化，我们假设分类器可以接受不完整的输入
                # 实际应用中，分类器应该接受完整的输入（数值+分类）
                pred_xstart_with_cat = pred_xstart  # 如果分类器需要完整输入，应该在这里组合
                g_classifier_num = compute_classifier_gradient_split(
                    classifier,
                    pred_xstart_with_cat,
                    None,  # 无分类特征
                    target_y,
                    self.num_numerical_features,
                    [],
                    x.device
                )[0]
            
            # 2. 计算距离约束梯度
            if x_original is not None:
                g_distance = compute_distance_gradient(x_original, pred_xstart)
            else:
                g_distance = torch.zeros_like(g_classifier_num)
            
            # 调试输出：检查梯度
            if debug_mode:
                print(f"      [Debug] g_classifier_num range: [{g_classifier_num.min():.4f}, {g_classifier_num.max():.4f}], norm: {torch.norm(g_classifier_num, dim=1).mean():.4f}", flush=True)
                print(f"      [Debug] g_distance range: [{g_distance.min():.4f}, {g_distance.max():.4f}], norm: {torch.norm(g_distance, dim=1).mean():.4f}", flush=True)
            
            # 3. 归一化组合
            g_guided = compute_guided_gradient(
                g_classifier_num,
                g_distance,
                lambda_guidance
            )
            
            # 调试输出：检查引导梯度
            if debug_mode:
                print(f"      [Debug] g_guided range: [{g_guided.min():.4f}, {g_guided.max():.4f}], norm: {torch.norm(g_guided, dim=1).mean():.4f}", flush=True)
            
            # 4. 应用引导到均值
            # 论文公式23：μ_guided = μ_θ + Σ_θ · ||μ_θ|| · g_guided
            # 其中：Σ_θ是协方差矩阵（对角矩阵，对角线为方差σ²）
            # 所以：Σ_θ · ||μ_θ|| · g_guided = σ² · ||μ_θ|| · g_guided
            variance = out["variance"]  # 方差σ²（不是标准差σ）
            mu_norm = torch.norm(mu_theta, dim=1, keepdim=True)
            
            # 数值稳定性保护：限制variance和mu_norm的大小，避免数值爆炸
            # 如果variance或mu_norm过大，会导致引导项过大，从而产生异常值
            # 使用soft clipping保持梯度流
            
            # 调试输出：检查裁剪前的值
            if debug_mode:
                variance_raw = variance.clone()
                mu_norm_raw = mu_norm.clone()
                print(f"      [Debug] variance (before clipping) range: [{variance_raw.min():.4f}, {variance_raw.max():.4f}], mean: {variance_raw.mean():.4f}", flush=True)
                print(f"      [Debug] mu_norm (before clipping) range: [{mu_norm_raw.min():.4f}, {mu_norm_raw.max():.4f}], mean: {mu_norm_raw.mean():.4f}", flush=True)
            
            # 限制variance最大为100（对应标准差最大为10）
            variance = torch.tanh(variance / 100.0) * 100.0
            mu_norm = torch.tanh(mu_norm / 10.0) * 10.0  # 平滑限制mu_norm最大为10
            
            # 调试输出：检查裁剪后的值和引导项
            if debug_mode:
                print(f"      [Debug] variance (after clipping) range: [{variance.min():.4f}, {variance.max():.4f}]", flush=True)
                print(f"      [Debug] mu_norm (after clipping) range: [{mu_norm.min():.4f}, {mu_norm.max():.4f}]", flush=True)
                guidance_term = variance * mu_norm * g_guided
                print(f"      [Debug] guidance_term (variance * mu_norm * g_guided) range: [{guidance_term.min():.4f}, {guidance_term.max():.4f}], mean: {guidance_term.mean():.4f}", flush=True)
            
            # 严格按照论文公式23：μ_guided = μ_θ + Σ_θ · ||μ_θ|| · g_guided
            # 其中Σ_θ是对角协方差矩阵（对角线为variance），所以是variance * mu_norm * g_guided
            mu_guided = mu_theta + variance * mu_norm * g_guided
            
            # 调试输出：检查引导后的均值
            if debug_mode:
                print(f"      [Debug] mu_theta range: [{mu_theta.min():.4f}, {mu_theta.max():.4f}], mean: {mu_theta.mean():.4f}", flush=True)
                print(f"      [Debug] mu_guided range: [{mu_guided.min():.4f}, {mu_guided.max():.4f}], mean: {mu_guided.mean():.4f}", flush=True)
            
            # 5. 应用不可变特征掩码
            if immutable_mask is not None:
                # immutable_mask shape: (batch_size, num_numerical_features)
                # 掩码为0的位置保持原始值，掩码为1的位置使用新采样值
                mu_guided = mu_guided * immutable_mask + mu_theta * (1 - immutable_mask)
        else:
            mu_guided = mu_theta
        
        # 采样
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        # 数值稳定性保护：限制采样噪声的标准差
        sample_variance = torch.exp(0.5 * out["log_variance"])
        
        # 调试输出：检查采样方差
        if debug_mode:
            sample_variance_raw = sample_variance.clone()
            print(f"      [Debug] sample_variance (before clipping) range: [{sample_variance_raw.min():.4f}, {sample_variance_raw.max():.4f}], mean: {sample_variance_raw.mean():.4f}", flush=True)
        
        # 使用soft clipping保持梯度流
        sample_variance = torch.tanh(sample_variance / 10.0) * 10.0
        
        # 调试输出：检查采样方差（裁剪后）
        if debug_mode:
            print(f"      [Debug] sample_variance (after clipping) range: [{sample_variance.min():.4f}, {sample_variance.max():.4f}]", flush=True)  # 平滑限制采样标准差最大为10
        
        sample = mu_guided + nonzero_mask * sample_variance * noise
        
        # 额外的数值稳定性保护：使用平滑裁剪采样结果，避免极端值
        
        # 调试输出：检查采样结果（裁剪前）
        if debug_mode:
            sample_raw = sample.clone()
            print(f"      [Debug] sample (before final clipping) range: [{sample_raw.min():.4f}, {sample_raw.max():.4f}], mean: {sample_raw.mean():.4f}, std: {sample_raw.std():.4f}", flush=True)
        # 对于标准化后的数据，通常应该在[-5, 5]范围内
        # 使用tanh进行平滑裁剪，保持梯度流
        sample = torch.tanh(sample / 10.0) * 10.0  # 平滑裁剪到[-10, 10]
        
        # 调试输出：检查采样结果（裁剪后）
        if debug_mode:
            print(f"      [Debug] sample (after final clipping) range: [{sample.min():.4f}, {sample.max():.4f}]", flush=True)
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # Multinomial part

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def predict_start(self, model_out, log_x_t, t, out_dict):

        # model_out = self._denoise_fn(x_t, t.to(x_t.device), **out_dict)

        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes.sum(), f'{model_out.size()}'

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # EV_log_qxt_x0 = self.q_pred(log_x_start, t)

        # print('sum exp', EV_log_qxt_x0.exp().sum(1).mean())
        # assert False

        # log_qxt_x0 = (log_x_t.exp() * EV_log_qxt_x0).sum(dim=1)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32))

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - sliced_logsumexp(unnormed_logprobs, self.offsets)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t, out_dict):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t, out_dict):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t, out_dict=out_dict)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape, out_dict):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), out_dict)
        return img

    @torch.no_grad()
    def _sample(self, image_size, out_dict, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size), out_dict)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample
    
    def q_sample_gumbel_softmax(self, x_cat_onehot, t):
        """
        TDCE: Gumbel-Softmax前向扩散（替换原有的多项式扩散）
        
        Args:
            x_cat_onehot: shape (batch_size, num_cat_features, max_num_classes)
                        - one-hot编码的分类特征
            t: shape (batch_size,) - 时间步索引
        
        Returns:
            x_t_cat: shape (batch_size, num_cat_features, max_num_classes) - Gumbel-Softmax连续向量
        """
        from .gumbel_softmax_utils import gumbel_softmax_q_sample, temperature_scheduler
        
        # 计算当前时间步的温度
        if self.tau_schedule == 'anneal':
            # 对于batch中的每个样本，可能需要不同的温度（如果t不同）
            # 简化处理：使用batch中第一个样本的时间步
            if isinstance(t, torch.Tensor):
                t_val = t[0].item() if t.numel() > 0 else 0
            else:
                t_val = t
            tau = temperature_scheduler(
                t_val,
                self.tau_init,
                self.tau_final,
                self.num_timesteps
            )
        else:
            tau = self.tau_final
        
        # 调用Gumbel-Softmax扩散
        return gumbel_softmax_q_sample(
            x_cat_onehot,
            t,
            self.betas,
            tau,
            x_cat_onehot.device
        )
    
    def p_sample_gumbel_softmax(
        self,
        model_out_cat,
        x_t_cat_gumbel,
        t,
        out_dict,
        # TDCE新增参数：分类器引导
        classifier=None,
        target_y=None,
        x_num=None,
        x_original_cat=None,
        immutable_mask_cat=None,
        lambda_guidance=1.0
    ):
        """
        TDCE: Gumbel-Softmax反向采样（分类特征）
        
        从模型输出预测x_0，然后采样得到x_{t-1}
        支持分类器梯度引导：使用一阶泰勒展开将梯度融入logits
        
        公式：log p_φ(y|\\tilde{x}_t) ≈ (\\tilde{x}_t - \\tilde{x}_{t+1})^T g_cat + const
        
        Args:
            model_out_cat: shape (batch_size, sum(num_classes)) - 模型输出的分类特征logits
            x_t_cat_gumbel: shape (batch_size, num_cat_features, max_num_classes) 
                          - 当前时间步的Gumbel-Softmax连续向量
            t: shape (batch_size,) - 时间步索引
            out_dict: 输出字典（可能包含y等条件信息）
            classifier: 分类器（用于引导，可选）
            target_y: 目标标签（用于引导，可选）
            x_num: shape (batch_size, num_numerical_features) - 数值特征（用于分类器输入）
            x_original_cat: shape (batch_size, num_cat_features, max_num_classes) 
                          - 原始分类特征（用于距离约束，可选）
            immutable_mask_cat: shape (batch_size, num_cat_features, max_num_classes)
                               - 不可变特征掩码（0=不可变，1=可变）
            lambda_guidance: 引导权重（控制分类器梯度和距离约束的平衡）
        
        Returns:
            x_t_minus_1_cat: shape (batch_size, num_cat_features, max_num_classes) 
                           - 下一步的Gumbel-Softmax连续向量
        """
        from .gumbel_softmax_utils import (
            gumbel_softmax_relaxation, 
            temperature_scheduler,
            gumbel_softmax_p_sample_logits
        )
        
        device = x_t_cat_gumbel.device
        batch_size = x_t_cat_gumbel.shape[0]
        num_cat_features = len(self.num_classes)
        max_num_classes = max(self.num_classes) if len(self.num_classes) > 0 else 0
        
        # 1. 计算当前时间步的温度
        if self.tau_schedule == 'anneal':
            if isinstance(t, torch.Tensor):
                t_val = t[0].item() if t.numel() > 0 else 0
            else:
                t_val = t
            tau = temperature_scheduler(
                t_val,
                self.tau_init,
                self.tau_final,
                self.num_timesteps
            )
            tau = torch.tensor(tau, device=device).float()
        else:
            tau = torch.tensor(self.tau_final, device=device).float()
        
        # 2. 从模型输出提取每个分类特征的logits
        logits_per_feat = gumbel_softmax_p_sample_logits(
            model_out_cat,
            x_t_cat_gumbel,
            t,
            self.num_classes,
            tau
        )
        
        # TDCE: 如果有分类器引导，计算梯度并调整logits
        if classifier is not None and target_y is not None:
            from .classifier_guidance import (
                compute_classifier_gradient_split,
                compute_distance_gradient,
                compute_guided_gradient
            )
            
            # 1. 计算分类器梯度（分类特征部分）
            # 注意：需要同时传入数值特征（如果有）和分类特征
            _, g_cat = compute_classifier_gradient_split(
                classifier,
                x_num,  # 传入数值特征（如果有）
                x_t_cat_gumbel,
                target_y,
                self.num_numerical_features,
                list(self.num_classes),
                device
            )
            
            # 2. 计算距离约束梯度（分类特征部分）
            if x_original_cat is not None:
                # 需要将Gumbel-Softmax向量转换为可导的距离计算
                # 这里简化处理：直接使用L2距离
                distance = torch.sum((x_original_cat - x_t_cat_gumbel) ** 2, dim=[1, 2])
                # 计算梯度（简化版本，实际应该对x_t_cat_gumbel求导）
                g_distance_cat = -2 * (x_original_cat - x_t_cat_gumbel)  # 简化梯度
            else:
                g_distance_cat = torch.zeros_like(g_cat)
            
            # 3. 归一化组合梯度
            # 将梯度reshape为(batch_size, num_cat_features, max_num_classes)形式
            g_cat_reshaped = g_cat.view(batch_size, num_cat_features, max_num_classes)
            g_distance_cat_reshaped = g_distance_cat.view(batch_size, num_cat_features, max_num_classes)
            
            # 对每个特征单独归一化组合
            g_guided_cat = torch.zeros_like(g_cat_reshaped)
            for i in range(num_cat_features):
                num_class = self.num_classes[i]
                g_classifier_feat = g_cat_reshaped[:, i, :num_class]
                g_distance_feat = g_distance_cat_reshaped[:, i, :num_class]
                
                # 归一化组合
                g_classifier_norm = g_classifier_feat / (
                    torch.norm(g_classifier_feat, dim=1, keepdim=True) + 1e-8
                )
                g_distance_norm = g_distance_feat / (
                    torch.norm(g_distance_feat, dim=1, keepdim=True) + 1e-8
                )
                
                # 严格按照论文公式：直接相减，没有lambda（与连续特征一致）
                g_guided_feat = g_classifier_norm - g_distance_norm
                g_guided_cat[:, i, :num_class] = g_guided_feat
            
            # 4. 将梯度融入logits（论文公式22）
            # log p_{θ,φ}(x̃_t | x̃_{t+1}, y) ≈ x̃_t^T (log π_θ(x̃_{t+1}) + λ g_cat) + const
            # 其中：g_cat = ∇ log p_φ(y | x̃_t)|_{x̃_t = x̃_{t+1}}
            # 注意：论文公式22中使用的是分类器梯度g_cat，不是g_guided
            # 但为了与连续特征保持一致，我们使用g_guided（分类器梯度 - 距离梯度）
            # 实际上，根据论文，分类特征的引导应该只使用分类器梯度，不使用距离约束
            # 但为了保持一致性，我们仍然使用g_guided
            for i, num_class in enumerate(self.num_classes):
                g_guided_feat = g_guided_cat[:, i, :num_class]
                # 论文公式22：logits = log π_θ(x̃_{t+1}) + λ g_cat
                # 这里使用g_guided（分类器梯度 - 距离梯度），λ为lambda_guidance
                logits_per_feat[i] = logits_per_feat[i] + lambda_guidance * g_guided_feat
        
        # 3. 对每个分类特征应用Gumbel-Softmax采样得到x_{t-1}
        x_t_minus_1_cat = torch.zeros(batch_size, num_cat_features, max_num_classes, device=device)
        
        for i, num_class in enumerate(self.num_classes):
            logits_feat = logits_per_feat[i]  # shape: (batch_size, num_class)
            
            # 应用Gumbel-Softmax采样（hard=False，保持连续可导）
            x_t_minus_1_feat = gumbel_softmax_relaxation(
                logits_feat,
                tau=tau.item() if isinstance(tau, torch.Tensor) else tau,
                hard=False
            )  # shape: (batch_size, num_class)
            
            # 存储到对应的位置
            x_t_minus_1_cat[:, i, :num_class] = x_t_minus_1_feat
        
        # 4. 应用不可变特征掩码（如果提供）
        if immutable_mask_cat is not None:
            # immutable_mask_cat shape: (batch_size, num_cat_features, max_num_classes)
            # 掩码为0的位置保持原始值，掩码为1的位置使用新采样值
            x_t_minus_1_cat = x_t_minus_1_cat * immutable_mask_cat + x_t_cat_gumbel * (1 - immutable_mask_cat)
        
        return x_t_minus_1_cat

    def nll(self, log_x_start, out_dict):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array,
                out_dict=out_dict)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, model_out, log_x_start, log_x_t, t, out_dict, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t, out_dict=out_dict)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt, out_dict):

        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_Lt(
                model_out, log_x_start, log_x_t, t, out_dict
            )
            kl_prior = self.kl_prior(log_x_start)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return vb_loss

        elif self.multinomial_loss_type == 'vb_all':
            # Expensive, dont do it ;).
            # DEPRECATED
            return -self.nll(log_x_start)
        else:
            raise ValueError()

    def log_prob(self, x, out_dict):
        b, device = x.size(0), x.device
        if self.training:
            return self._multinomial_loss(x, out_dict)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, out_dict)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss
    
    def mixed_loss(self, x, out_dict):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]
        
        x_num_t = x_num
        log_x_cat_t = x_cat
        x_cat_t_gumbel = None
        
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        
        if x_cat.shape[1] > 0:
            # TDCE: 使用Gumbel-Softmax扩散
            from .gumbel_softmax_utils import index_to_onehot
            
            # 将索引转为one-hot编码
            x_cat_onehot = index_to_onehot(x_cat.long(), self.num_classes)
            
            # Gumbel-Softmax前向扩散
            x_cat_t_gumbel = self.q_sample_gumbel_softmax(x_cat_onehot, t)
            
            # 将Gumbel-Softmax向量flatten后与数值特征拼接（用于模型输入）
            # 需要按特征展开为(batch_size, sum(num_classes))
            batch_size = x_cat_t_gumbel.shape[0]
            num_cat_features = len(self.num_classes)
            max_num_classes = x_cat_t_gumbel.shape[2]
            
            # 将(batch_size, num_cat_features, max_num_classes)展平为(batch_size, sum(num_classes))
            x_cat_t_flat = []
            for i in range(num_cat_features):
                num_class = self.num_classes[i]
                x_cat_t_flat.append(x_cat_t_gumbel[:, i, :num_class])
            log_x_cat_t = torch.cat(x_cat_t_flat, dim=1)
        
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)

        model_out = self._denoise_fn(
            x_in,
            t,
            **out_dict
        )

        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        loss_multi = torch.zeros((1,)).float().to(device)
        loss_gauss = torch.zeros((1,)).float().to(device)
        
        if x_cat.shape[1] > 0:
            # TDCE: Gumbel-Softmax损失计算（使用MSE作为简化版本）
            # 模型输出应该是每个分类特征的logits，需要转换为Gumbel-Softmax分布
            from .gumbel_softmax_utils import gumbel_softmax_relaxation, index_to_onehot, gumbel_softmax_p_sample_logits
            
            # 将原始分类特征转为one-hot
            x_cat_onehot = index_to_onehot(x_cat.long(), self.num_classes)
            
            # 计算当前时间步的温度
            if self.tau_schedule == 'anneal':
                if isinstance(t, torch.Tensor):
                    t_val = t[0].item() if t.numel() > 0 else 0
                else:
                    t_val = t
                from .gumbel_softmax_utils import temperature_scheduler
                tau = temperature_scheduler(t_val, self.tau_init, self.tau_final, self.num_timesteps)
            else:
                tau = self.tau_final
            
            # 将模型输出的logits转换为Gumbel-Softmax分布（预测的x_0）
            logits_per_feat = gumbel_softmax_p_sample_logits(
                model_out_cat, x_cat_t_gumbel, t, self.num_classes, tau
            )
            
            # 重新组装为(batch_size, num_cat_features, max_num_classes)
            batch_size = model_out_cat.shape[0]
            num_cat_features = len(self.num_classes)
            max_num_classes = max(self.num_classes) if len(self.num_classes) > 0 else 0
            
            model_out_cat_gumbel = torch.zeros(batch_size, num_cat_features, max_num_classes, device=device)
            for i, num_class in enumerate(self.num_classes):
                model_out_cat_gumbel[:, i, :num_class] = gumbel_softmax_relaxation(
                    logits_per_feat[i], tau=tau, hard=False
                )
            
            # 计算MSE损失（预测的x_0 vs 真实的x_0）
            # 使用x_cat_onehot作为真实值
            loss_multi = F.mse_loss(
                model_out_cat_gumbel[:, :, :max_num_classes],
                x_cat_onehot[:, :, :max_num_classes],
                reduction='none'
            )
            # 只计算有效的类别维度
            mask = torch.zeros_like(loss_multi)
            for i, num_class in enumerate(self.num_classes):
                mask[:, i, :num_class] = 1.0
            loss_multi = (loss_multi * mask).sum(dim=[1, 2]).mean() / len(self.num_classes)
        
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)

        # loss_multi = torch.where(out_dict['y'] == 1, loss_multi, 2 * loss_multi)
        # loss_gauss = torch.where(out_dict['y'] == 1, loss_gauss, 2 * loss_gauss)

        return loss_multi.mean() if isinstance(loss_multi, torch.Tensor) else loss_multi, \
               loss_gauss.mean() if isinstance(loss_gauss, torch.Tensor) else loss_gauss
    
    @torch.no_grad()
    def mixed_elbo(self, x0, out_dict):
        b = x0.size(0)
        device = x0.device

        x_num = x0[:, :self.num_numerical_features]
        x_cat = x0[:, self.num_numerical_features:]
        has_cat = x_cat.shape[1] > 0
        if has_cat:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes).to(device)

        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        multinomial_loss = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x_num)

            x_num_t = self.gaussian_q_sample(x_start=x_num, t=t_array, noise=noise)
            if has_cat:
                log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_array)
            else:
                log_x_cat_t = x_cat

            model_out = self._denoise_fn(
                torch.cat([x_num_t, log_x_cat_t], dim=1),
                t_array,
                **out_dict
            )
            
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]

            kl = torch.tensor([0.0])
            if has_cat:
                kl = self.compute_Lt(
                    model_out=model_out_cat,
                    log_x_start=log_x_cat,
                    log_x_t=log_x_cat_t,
                    t=t_array,
                    out_dict=out_dict
                )

            out = self._vb_terms_bpd(
                model_out_num,
                x_start=x_num,
                x_t=x_num_t,
                t=t_array,
                clip_denoised=False
            )

            multinomial_loss.append(kl)
            gaussian_loss.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_num) ** 2))
            # mu_mse.append(mean_flat(out["mean_mse"]))
            out_mean.append(mean_flat(out["out_mean"]))
            true_mean.append(mean_flat(out["true_mean"]))

            eps = self._predict_eps_from_xstart(x_num_t, t_array, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        multinomial_loss = torch.stack(multinomial_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        # mu_mse = torch.stack(mu_mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)



        prior_gauss = self._prior_gaussian(x_num)

        prior_multin = torch.tensor([0.0])
        if has_cat:
            prior_multin = self.kl_prior(log_x_cat)

        total_gauss = gaussian_loss.sum(dim=1) + prior_gauss
        total_multin = multinomial_loss.sum(dim=1) + prior_multin
        return {
            "total_gaussian": total_gauss,
            "total_multinomial": total_multin,
            "losses_gaussian": gaussian_loss,
            "losses_multinimial": multinomial_loss,
            "xstart_mse": xstart_mse,
            "mse": mse,
            # "mu_mse": mu_mse
            "out_mean": out_mean,
            "true_mean": true_mean
        }

    @torch.no_grad()
    def gaussian_ddim_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        eta=0.0
    ):
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=None,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample
    
    @torch.no_grad()
    def gaussian_ddim_sample(
        self,
        noise,
        T,
        out_dict,
        eta=0.0
    ):
        x = noise
        b = x.shape[0]
        device = x.device
        for t in reversed(range(T)):
            print(f'Sample timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_step(
                out_num,
                x,
                t_array
            )
        print()
        return x


    @torch.no_grad()
    def gaussian_ddim_reverse_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        eta=0.0
    ):
        assert eta == 0.0, "Eta must be zero."
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=None,
        )

        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(
        self,
        x,
        T,
        out_dict,
    ):
        b = x.shape[0]
        device = x.device
        for t in range(T):
            print(f'Reverse timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_reverse_step(
                out_num,
                x,
                t_array,
                eta=0.0
            )
        print()

        return x


    @torch.no_grad()
    def multinomial_ddim_step(
        self,
        model_out_cat,
        log_x_t,
        t,
        out_dict,
        eta=0.0
    ):
        # not ddim, essentially
        log_x0 = self.predict_start(model_out_cat, log_x_t=log_x_t, t=t, out_dict=out_dict)

        alpha_bar = extract(self.alphas_cumprod, t, log_x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, log_x_t.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2
        

        log_ps = torch.stack([
            torch.log(coef1) + log_x_t,
            torch.log(coef2) + log_x0,
            torch.log(coef3) - torch.log(self.num_classes_expanded)
        ], dim=2)

        log_prob = torch.logsumexp(log_ps, dim=2)

        out = self.log_sample_categorical(log_prob)

        return out

    @torch.no_grad()
    def sample_ddim(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(
            y_dist,
            num_samples=b,
            replacement=True
        )
        out_dict = {'y': y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_ddim_step(model_out_num, z_norm, t, clip_denoised=False)
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t, out_dict)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict
    

    @torch.no_grad()
    def sample(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        
        # TDCE: 初始化Gumbel-Softmax连续向量
        z_cat_gumbel = None
        if has_cat:
            # TDCE: 初始化均匀分布的Gumbel-Softmax连续向量
            from .gumbel_softmax_utils import gumbel_softmax_relaxation
            num_cat_features = len(self.num_classes)
            max_num_classes = max(self.num_classes) if len(self.num_classes) > 0 else 0
            
            # 创建均匀分布的logits，然后应用Gumbel-Softmax
            z_cat_gumbel = torch.zeros((b, num_cat_features, max_num_classes), device=device)
            for i, num_class in enumerate(self.num_classes):
                uniform_logits = torch.zeros((b, num_class), device=device)
                tau = self.tau_init  # 初始时间步使用初始温度
                z_cat_gumbel[:, i, :num_class] = gumbel_softmax_relaxation(
                    uniform_logits, tau=tau, hard=False
                )
            
            # 将Gumbel-Softmax向量展平为(batch_size, sum(num_classes))用于模型输入
            log_z = []
            for i, num_class in enumerate(self.num_classes):
                log_z.append(z_cat_gumbel[:, i, :num_class])
            log_z = torch.cat(log_z, dim=1)
        else:
            log_z = torch.zeros((b, 0), device=device).float()

        y = torch.multinomial(
            y_dist,
            num_samples=b,
            replacement=True
        )
        out_dict = {'y': y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
            
            if has_cat:
                # TDCE: 使用Gumbel-Softmax反向采样
                # 需要将log_z重新组装为(batch_size, num_cat_features, max_num_classes)形状
                num_cat_features = len(self.num_classes)
                max_num_classes = max(self.num_classes) if len(self.num_classes) > 0 else 0
                
                # 重新组装z_cat_gumbel
                z_cat_gumbel = torch.zeros((b, num_cat_features, max_num_classes), device=device)
                offset = 0
                for j, num_class in enumerate(self.num_classes):
                    z_cat_gumbel[:, j, :num_class] = log_z[:, offset:offset + num_class]
                    offset += num_class
                
                # 调用Gumbel-Softmax反向采样
                z_cat_gumbel = self.p_sample_gumbel_softmax(
                    model_out_cat,
                    z_cat_gumbel,
                    t,
                    out_dict
                )
                
                # 重新展平为log_z格式用于模型输入
                log_z = []
                for j, num_class in enumerate(self.num_classes):
                    log_z.append(z_cat_gumbel[:, j, :num_class])
                log_z = torch.cat(log_z, dim=1)

        print()
        
        # 后处理：将分类特征转换为离散索引
        if has_cat:
            # TDCE: 将Gumbel-Softmax连续向量转换为离散索引
            from .gumbel_softmax_utils import gumbel_softmax_to_index
            z_cat = gumbel_softmax_to_index(z_cat_gumbel, self.num_classes)
        else:
            z_cat = torch.zeros((b, 0), device=device)
        
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict
    
    def sample_all(self, num_samples, batch_size, y_dist, ddim=False):
        if ddim:
            print('Sample using DDIM.')
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample
        
        b = batch_size

        all_y = []
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            sample, out_dict = sample_fn(b, y_dist)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            out_dict['y'] = out_dict['y'][~mask_nan]

            all_samples.append(sample)
            all_y.append(out_dict['y'].cpu())
            if sample.shape[0] != b:
                raise FoundNANsError
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]
        y_gen = torch.cat(all_y, dim=0)[:num_samples]

        return x_gen, y_gen
    
    def sample_counterfactual(
        self,
        x_original,
        y_original,
        target_y,
        classifier,
        immutable_mask=None,
        lambda_guidance=1.0,
        num_steps=None,
        start_from_noise=True
    ):
        """
        TDCE: 反事实生成方法
        
        从原始样本生成满足目标标签的反事实样本，使用分类器梯度引导
        
        **论文要求**：每个测试样本生成1个反事实样本（一对一映射）
        - 输入N个原始样本，输出N个反事实样本
        - 每个数据集的测试集固定为1,000条，因此生成1,000个反事实样本
        
        Args:
            x_original: shape (batch_size, total_features) - 原始样本
            y_original: shape (batch_size,) - 原始标签（可选，用于验证）
            target_y: 目标标签（标量或shape (batch_size,)）
            classifier: 分类器模型（用于引导）
            immutable_mask: shape (batch_size, total_features) - 不可变特征掩码（1=可变，0=不可变）
            lambda_guidance: 引导权重（控制分类器梯度和距离约束的平衡）
            num_steps: 反向步数（默认使用全部时间步T=1000）
            start_from_noise: 是否从完全噪声开始（True）或从部分加噪的原始样本开始（False）
        
        Returns:
            x_counterfactual: shape (batch_size, total_features) - 反事实样本
                - 每个原始样本对应生成1个反事实样本
                - 形状与输入相同
        """
        device = x_original.device
        batch_size = x_original.shape[0]
        
        if num_steps is None:
            num_steps = self.num_timesteps
        
        # 1. 拆分特征
        # 注意：如果使用one-hot编码，分类特征已经合并到数值特征中
        # 此时num_classes为空或全为0，所有特征都在数值特征中
        if len(self.num_classes) == 0 or (len(self.num_classes) == 1 and self.num_classes[0] == 0):
            # 使用one-hot编码，所有特征都在数值特征中
            x_num_original = x_original  # 包含所有特征（68维）
            x_cat_original = torch.zeros((batch_size, 0), device=device)  # 没有分类特征
        else:
            # 有分类特征，按原始方式拆分
            x_num_original = x_original[:, :self.num_numerical_features]
            x_cat_original = x_original[:, self.num_numerical_features:]
        
        # 2. 处理目标标签
        if isinstance(target_y, int):
            target_y = torch.full((batch_size,), target_y, device=device, dtype=torch.long)
        else:
            target_y = target_y.long().to(device)
        
        # 3. 拆分不可变掩码
        if immutable_mask is not None:
            if len(self.num_classes) == 0 or (len(self.num_classes) == 1 and self.num_classes[0] == 0):
                # 使用one-hot编码，所有特征都在数值特征中
                immutable_mask_num = immutable_mask  # 包含所有特征
                immutable_mask_cat = torch.ones((batch_size, 0), device=device)  # 没有分类特征
            else:
                # 有分类特征，按原始方式拆分
                immutable_mask_num = immutable_mask[:, :self.num_numerical_features]
                immutable_mask_cat = immutable_mask[:, self.num_numerical_features:]
        else:
            immutable_mask_num = None
            immutable_mask_cat = None
        
        # 4. 前向扩散（加噪）或从噪声开始
        if start_from_noise:
            # 从完全噪声开始
            T0 = num_steps - 1
            t_start = torch.full((batch_size,), T0, device=device, dtype=torch.long)
            
            # 连续特征：从完全噪声开始
            if x_num_original.shape[1] > 0:
                x_num_t = torch.randn_like(x_num_original)
            else:
                x_num_t = x_num_original
            
            # 分类特征：从完全噪声开始
            # 注意：如果使用one-hot编码，分类特征已经合并到数值特征中
            # 此时num_classes为空或全为0，x_cat_original.shape[1]也应该为0
            if x_cat_original.shape[1] > 0 and len(self.num_classes) > 0 and max(self.num_classes) > 0:
                # TDCE: Gumbel-Softmax模式，从均匀分布开始
                from .gumbel_softmax_utils import gumbel_softmax_relaxation
                num_cat_features = len(self.num_classes)
                max_num_classes = max(self.num_classes) if len(self.num_classes) > 0 else 0
                x_cat_t_gumbel = torch.zeros(batch_size, num_cat_features, max_num_classes, device=device)
                for i, num_class in enumerate(self.num_classes):
                    uniform_logits = torch.zeros((batch_size, num_class), device=device)
                    tau = self.tau_init  # 初始时间步使用初始温度
                    x_cat_t_gumbel[:, i, :num_class] = gumbel_softmax_relaxation(
                        uniform_logits, tau=tau, hard=False
                    )
                x_cat_t = x_cat_t_gumbel
            else:
                # 没有分类特征或分类特征已合并到数值特征中
                x_cat_t = x_cat_original
        else:
            # 从部分加噪的原始样本开始
            T0 = num_steps - 1
            t_start = torch.full((batch_size,), T0, device=device, dtype=torch.long)
            
            # 连续特征：前向扩散
            if x_num_original.shape[1] > 0:
                noise = torch.randn_like(x_num_original)
                x_num_t = self.gaussian_q_sample(x_num_original, t_start, noise=noise)
            else:
                x_num_t = x_num_original
            
            # 分类特征：前向扩散
            # 注意：如果使用one-hot编码，分类特征已经合并到数值特征中
            # 此时num_classes为空或全为0，x_cat_original.shape[1]也应该为0
            if x_cat_original.shape[1] > 0 and len(self.num_classes) > 0 and max(self.num_classes) > 0:
                # TDCE: Gumbel-Softmax前向扩散
                from .gumbel_softmax_utils import index_to_onehot
                x_cat_onehot = index_to_onehot(x_cat_original.long(), list(self.num_classes))
                x_cat_t = self.q_sample_gumbel_softmax(x_cat_onehot, t_start)
            else:
                # 没有分类特征或分类特征已合并到数值特征中
                x_cat_t = x_cat_original
        
        # 5. 准备分类特征的相关数据（用于引导）
        # 注意：如果使用one-hot编码，分类特征已经合并到数值特征中
        if x_cat_original.shape[1] > 0 and len(self.num_classes) > 0 and max(self.num_classes) > 0:
            # 将原始分类特征转为one-hot（用于距离约束）
            from .gumbel_softmax_utils import index_to_onehot
            x_cat_original_onehot = index_to_onehot(x_cat_original.long(), list(self.num_classes))
        else:
            x_cat_original_onehot = None
        
        # 6. 反向去噪（带分类器引导）
        out_dict = {'y': target_y}
        
        # 准备用于模型输入的log_z（TDCE使用Gumbel-Softmax）
        # 注意：如果使用one-hot编码，分类特征已经合并到数值特征中
        if x_cat_original.shape[1] > 0 and len(self.num_classes) > 0 and max(self.num_classes) > 0:
            log_z = []
            for i, num_class in enumerate(self.num_classes):
                log_z.append(x_cat_t[:, i, :num_class])
            log_z = torch.cat(log_z, dim=1)
            x_cat_t_gumbel = x_cat_t  # 保留原始形状用于引导
        else:
            log_z = torch.zeros((batch_size, 0), device=device)
            x_cat_t_gumbel = None
        
        # 反向迭代
        print(f"  Starting reverse diffusion sampling ({num_steps} steps)...", flush=True)
        
        # 调试模式：每100步输出详细信息
        debug_mode = True  # 设置为False可以关闭详细调试输出
        
        # 调试输出：初始状态
        if debug_mode and x_num_t.shape[1] > 0:
            print(f"  [Debug] Initial x_num_t range: [{x_num_t.min():.4f}, {x_num_t.max():.4f}], mean: {x_num_t.mean():.4f}, std: {x_num_t.std():.4f}", flush=True)
        
        for step_idx, i in enumerate(reversed(range(0, num_steps))):
            if step_idx % 100 == 0 or step_idx == num_steps - 1:
                print(f"  Sampling step {step_idx + 1}/{num_steps} (t={i})", flush=True)
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 调试输出：检查输入状态（裁剪前）
            if debug_mode and step_idx % 100 == 0 and x_num_t.shape[1] > 0:
                print(f"    [Debug] x_num_t (before clipping) range: [{x_num_t.min():.4f}, {x_num_t.max():.4f}], mean: {x_num_t.mean():.4f}, std: {x_num_t.std():.4f}", flush=True)
            
            # 数值稳定性保护：在输入模型前，确保输入值在合理范围内
            # 使用平滑裁剪确保输入在合理范围内，同时保持梯度流
            if x_num_t.shape[1] > 0:
                x_num_t = torch.tanh(x_num_t / 10.0) * 10.0  # 平滑裁剪到[-10, 10]
            if log_z is not None and log_z.shape[1] > 0:
                log_z = torch.tanh(log_z / 10.0) * 10.0  # 平滑裁剪到[-10, 10]
            
            # 调试输出：检查输入状态（裁剪后）
            if debug_mode and step_idx % 100 == 0 and x_num_t.shape[1] > 0:
                print(f"    [Debug] x_num_t (after clipping) range: [{x_num_t.min():.4f}, {x_num_t.max():.4f}]", flush=True)
            
            # 组合输入（用于模型）
            # 注意：如果使用分类器引导，需要确保输入设置了requires_grad
            if classifier is not None:
                # 确保输入设置了requires_grad=True（在裁剪之后）
                if x_num_t.shape[1] > 0:
                    x_num_t = x_num_t.detach().requires_grad_(True)
                if log_z is not None and log_z.shape[1] > 0:
                    log_z = log_z.detach().requires_grad_(True)
            
            if x_num_t.shape[1] > 0 and log_z is not None and log_z.shape[1] > 0:
                x_t_model = torch.cat([x_num_t, log_z], dim=1)
            elif x_num_t.shape[1] > 0:
                x_t_model = x_num_t
            elif log_z is not None and log_z.shape[1] > 0:
                x_t_model = log_z
            else:
                x_t_model = torch.zeros((batch_size, 0), device=device)
            
            # 去噪网络预测
            # 注意：如果使用分类器引导，需要启用梯度计算
            if classifier is not None:
                # 启用梯度以计算分类器梯度
                with torch.enable_grad():
                    model_out = self._denoise_fn(x_t_model.float(), t, **out_dict)
            else:
                model_out = self._denoise_fn(x_t_model.float(), t, **out_dict)
            
            # 数值稳定性保护：使用soft clipping而不是hard clipping，以保持梯度流
            # 对于超出范围的值，使用平滑的sigmoid函数进行裁剪，保持梯度连续性
            # 公式：clamped = x * sigmoid((x - max) / scale) + max * sigmoid((x - max) / scale)
            # 简化为：使用tanh进行平滑裁剪
            
            # 调试输出：检查模型输出（裁剪前）
            if debug_mode and step_idx % 100 == 0:
                print(f"    [Debug] model_out (before clipping) range: [{model_out.min():.4f}, {model_out.max():.4f}], mean: {model_out.mean():.4f}, std: {model_out.std():.4f}", flush=True)
            
            model_out = torch.tanh(model_out / 10.0) * 10.0  # 平滑裁剪到[-10, 10]
            
            # 调试输出：检查模型输出（裁剪后）
            if debug_mode and step_idx % 100 == 0:
                print(f"    [Debug] model_out (after clipping) range: [{model_out.min():.4f}, {model_out.max():.4f}]", flush=True)
            
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            
            # 连续特征反向采样（带引导）
            if x_num_t.shape[1] > 0:
                # 如果使用分类器引导，需要启用梯度
                if classifier is not None:
                    with torch.enable_grad():
                        x_num_t_minus_1 = self.gaussian_p_sample(
                            model_out_num,
                            x_num_t,
                            t,
                            clip_denoised=False,
                            classifier=classifier,
                            target_y=target_y,
                            x_original=x_num_original,
                            x_cat_gumbel=x_cat_t_gumbel,  # 传入分类特征用于分类器输入
                            immutable_mask=immutable_mask_num,
                            lambda_guidance=lambda_guidance,
                            debug_mode=debug_mode and step_idx % 100 == 0  # 传递调试标志
                        )["sample"]
                else:
                    x_num_t_minus_1 = self.gaussian_p_sample(
                        model_out_num,
                        x_num_t,
                        t,
                        clip_denoised=False
                    )["sample"]
                
                # 调试输出：检查采样结果
                if debug_mode and step_idx % 100 == 0:
                    print(f"    [Debug] x_num_t_minus_1 range: [{x_num_t_minus_1.min():.4f}, {x_num_t_minus_1.max():.4f}], mean: {x_num_t_minus_1.mean():.4f}, std: {x_num_t_minus_1.std():.4f}", flush=True)
            else:
                x_num_t_minus_1 = x_num_t
            
            # 分类特征反向采样（带引导）
            if x_cat_original.shape[1] > 0:
                # TDCE: 使用Gumbel-Softmax反向采样
                x_cat_t_minus_1_gumbel = self.p_sample_gumbel_softmax(
                    model_out_cat,
                    x_cat_t_gumbel,
                    t,
                    out_dict,
                    classifier=classifier,
                    target_y=target_y,
                    x_num=x_num_t_minus_1,  # 传入数值特征用于分类器输入
                    x_original_cat=x_cat_original_onehot,
                    immutable_mask_cat=immutable_mask_cat,
                    lambda_guidance=lambda_guidance
                )
                
                # 更新log_z用于下一次迭代
                log_z = []
                for j, num_class in enumerate(self.num_classes):
                    log_z.append(x_cat_t_minus_1_gumbel[:, j, :num_class])
                log_z = torch.cat(log_z, dim=1)
                x_cat_t_gumbel = x_cat_t_minus_1_gumbel
                x_cat_t = x_cat_t_minus_1_gumbel
            else:
                x_cat_t_gumbel = None
            
            # 更新（detach以避免梯度累积）
            x_num_t = x_num_t_minus_1.detach()
            
            # 调试输出：检查最终采样结果（在逆变换前）
            if debug_mode and step_idx % 100 == 0 and x_num_t.shape[1] > 0:
                print(f"    [Debug] Final x_num_t (before inverse transform) range: [{x_num_t.min():.4f}, {x_num_t.max():.4f}], mean: {x_num_t.mean():.4f}, std: {x_num_t.std():.4f}", flush=True)
        
        # 7. 后处理：将分类特征转换回离散形式
        # 注意：如果使用one-hot编码，分类特征已经合并到数值特征中
        if x_cat_original.shape[1] > 0 and len(self.num_classes) > 0 and max(self.num_classes) > 0:
            # TDCE: 从Gumbel-Softmax连续向量转换为离散索引
            from .gumbel_softmax_utils import gumbel_softmax_to_index
            x_cat_final = gumbel_softmax_to_index(x_cat_t_gumbel, list(self.num_classes))
        else:
            # 没有分类特征或分类特征已合并到数值特征中
            x_cat_final = torch.zeros((batch_size, 0), device=device)
        
        # 8. 组合最终反事实样本
        if x_num_t.shape[1] > 0 and x_cat_final.shape[1] > 0:
            x_counterfactual = torch.cat([x_num_t, x_cat_final], dim=1)
        elif x_num_t.shape[1] > 0:
            x_counterfactual = x_num_t
        else:
            x_counterfactual = x_cat_final
        
        # 确保最终结果被detach（释放梯度）
        return x_counterfactual.detach()