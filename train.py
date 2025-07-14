from sklearn.cluster import DBSCAN
import torch
import gpytorch

def trainGPmdl(train_x,train_y):

    # Define the Exact GP Model class
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

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # Put into training mode
    model.train()
    likelihood.train()

    # Use Adam optimizer for hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Marginal log likelihood loss for exact GPs
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter =1000
    for i in range(training_iter):
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if i % 100 == 0 or i == training_iter-1:
            print(f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}')
        optimizer.step()
    return model, likelihood

def compute_rkhs_norm(model, train_x, train_y, noise_variance=None):
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get kernel matrix K (without noise)
        K = model.covar_module(train_x).evaluate()

        # Add noise variance if not provided
        if noise_variance is None:
            noise_variance = model.likelihood.noise.item()

        K_noise = K + noise_variance * torch.eye(train_x.size(0), device=train_x.device)

        # Solve for alpha = (K + σ² I)^{-1} y
        alpha = torch.linalg.solve(K_noise, train_y)

        # Compute RKHS norm: alpha^T K alpha
        rkhs_norm_squared = alpha @ K @ alpha

        return torch.sqrt(rkhs_norm_squared)
