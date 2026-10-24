import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

    
class SimpleMLPModel(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1, dropout: float = 0.0, **kwargs):
        super().__init__()
        
        layers = []

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim)])

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: [batch_size, num_classes]
        return self.net(x)

    

class SimpleMLPModel2(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1, dropout: float = 0.0, **kwargs):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: [batch_size, num_classes]
        return self.net(x)


class DeepMLPModel(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1, dropout: float = 0.2, **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: [batch_size, num_classes]
        return self.net(x)


class RFModel:
    """
    Random Forest wrapper for regression.
    
    """

    def __init__(self, **kwargs):

        self.model = RandomForestRegressor(random_state=42, **kwargs)


    def fit(self, X, y):
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        self.model.fit(X_np, y_np)

    def predict(self, X):
        X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        return self.model.predict(X)

        


class XGBoostModel:
    """
    XGBoost wrapper for regression.
    """

    def __init__(self, **kwargs):
        from xgboost import XGBRegressor

        self.model = XGBRegressor(random_state=42, **kwargs)


    def fit(self, X, y):
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy()if isinstance(y, torch.Tensor) else y
        self.model.fit(X_np, y_np)

    def predict(self, X):
        X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        return self.model.predict(X)




HEADS = {
    "rf": RFModel,
    "xgboost": XGBoostModel,
    "mlp": SimpleMLPModel,
    "mlp2": SimpleMLPModel2,
    "deep-mlp": DeepMLPModel,
}


def get_model(t: str, cfg):
    try:
        return HEADS[t](**cfg)
    except KeyError:
        raise ValueError(f"Unknown head type: {t}")
