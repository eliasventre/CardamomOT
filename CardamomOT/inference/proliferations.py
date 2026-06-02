"""
Proliferation rate inference utilities for CardamomOT.

Provides a lightweight MLP that maps protein levels to a net proliferation
rate R = b - d (birth minus death), estimated from the row marginals of the
optimal-transport coupling computed during trajectory inference.
"""
import numpy as np
import torch
import torch.nn as nn


class ProliferationMLP(nn.Module):
    """Two-hidden-layer MLP: protein levels → net proliferation rate R."""

    def __init__(self, n_proteins: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_proteins, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict(self, prot: np.ndarray) -> np.ndarray:
        """prot: (N, n_proteins) → R values (N,)."""
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(prot, dtype=torch.float32)
            return self.forward(x).numpy()


def train_proliferation_mlp(
    prot: np.ndarray,
    R_opt: np.ndarray,
    ns: int = 1,
    hidden_size: int = 64,
    n_epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 256,
    verb: bool = True,
) -> ProliferationMLP:
    """
    Train a ProliferationMLP on (prot[:, ns:], R_opt) pairs.

    Parameters
    ----------
    prot : (N_total, G_tot) protein trajectory array (including stimulus dims).
    R_opt : (N_total,) net proliferation rates from OT coupling marginals.
    ns : number of stimulus dimensions to skip in prot.
    """
    prot_genes = prot[:, ns:].astype(np.float32)
    R_targets = R_opt.astype(np.float32)
    n_proteins = prot_genes.shape[1]

    model = ProliferationMLP(n_proteins, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = torch.tensor(prot_genes)
    y = torch.tensor(R_targets)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if verb and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"[ProliferationMLP] epoch {epoch:4d}/{n_epochs}  "
                  f"MSE={epoch_loss / len(dataset):.6f}")

    return model
