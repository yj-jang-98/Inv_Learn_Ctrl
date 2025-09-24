from sklearn.cluster import DBSCAN
import torch
import gpytorch
import matplotlib.pyplot as plt

def trainGPmdl(train_x,train_y, training_iter):

    # --- Define GP Model class ---
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # --- Initialize likelihood and model ---
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # --- Training mode ---
    model.train()
    likelihood.train()

    # --- Use Adam for hyperparameter optimization ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- Compute marginal log likelihood loss ---
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # --- Optimization ---
    for i in range(training_iter):
        optimizer.zero_grad()   
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if i % 100 == 0 or i == training_iter-1:
            print(f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}')
        optimizer.step()

    
    return model, likelihood


def trainGPmdlARD(train_x,train_y, training_iter, ard_num_dims):

    # --- Define ARD GP Model class ---
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # --- Initialize likelihood and model ---
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # --- Training mode ---
    model.train()
    likelihood.train()

    # --- Use Adam for hyperparameter optimization ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- Compute marginal log likelihood loss ---
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # --- Optimization ---
    for i in range(training_iter):
        optimizer.zero_grad()
        with gpytorch.settings.cholesky_jitter(1e-4):
            out  = model(train_x)
            loss = -mll(out, train_y)   
        loss.backward()
        if i % 100 == 0 or i == training_iter-1:
            print(f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}')
        optimizer.step()
    return model, likelihood

def compute_rkhs_norm(model, train_x, train_y, noise_variance=None):
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # --- Compute gram matrix K ---
        K = model.covar_module(train_x).evaluate()

        # --- Add jittering if noise_variance provided for numerical stability ---
        if noise_variance is None:
            noise_variance = model.likelihood.noise.item()

        K_noise = K + noise_variance * torch.eye(train_x.size(0), device=train_x.device)

        # --- Solve for alpha = (K + σ² I)^{-1} y ---
        alpha = torch.linalg.solve(K_noise, train_y)

        # --- Compute RKHS norm: alpha^T K alpha ---
        rkhs_norm_squared = alpha @ K @ alpha

        return torch.sqrt(rkhs_norm_squared)

def compute_gamma(model, rkhs_norm):
    # --- Estimate of RKHS norm by multiplying safety factor 1.1 ---
    Gamma = 1.1*rkhs_norm

    # Extract kernel hyperparameters
    lengthscale = model.covar_module.base_kernel.lengthscale.detach()
    outputscale = model.covar_module.outputscale.detach().item()

    # Handle ARD vs non-ARD
    if lengthscale.numel() == 1:  
        ls_value = lengthscale.item()
        print(f"Lengthscale: {ls_value}")
        print(f"Outputscale: {outputscale}")

        # Linear taylor expansion around zero
        # 1+ Lc + Lf + LfLc + T(\eta) + LfT(\eta)
        # T(\eta(\rho)) = (Gamma * sqrt(outputscale) / lengthscale) * \rho
        Lf = 1
        Lc = 1
        Teta = (Gamma * (outputscale ** 0.5)) / ls_value
        slope = 1 + Lc + Lf + Lf*Lc + Teta + Lf*Teta
        print(f"Taylor expansion slope: {slope}")
        return slope
    else:
        # For now, use the same slope as in the noise-free case for the noisy ARD case
        return 6.6857
