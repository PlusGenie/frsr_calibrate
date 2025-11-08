from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env early if present (no-op if missing)
load_dotenv()


class AnchorsSettings(BaseModel):
    H0_km_s_Mpc: float = Field(67.4, alias="H0_km_s_Mpc")
    Omega_b0: float = 0.048
    Omega_cdm0: float = 0.262
    Omega_r0: float = 5e-5
    Omega_k0: float = 0.0


class EoSSettings(BaseModel):
    # exp_wz defaults used in typical runs
    model: str = "exp_wz"
    w0: float = -0.98
    wa: float = 0.02
    epsilon: float = 0.02
    alpha: float = 1.1
    cs2: float = 1.0


class InputsSettings(BaseModel):
    # spectral ladder high-level knobs (used in calibrate CLI)
    N: int = 6
    r: float = 2.5
    alpha: float = 0.1
    beta: float = 5.5
    E0: float = 0.1
    k_IR: float = 1.0
    q: Optional[float] = None
    Lambda_obs: float = 1.11e-52
    auto_kIR: bool = True


class ClassParamsSettings(BaseModel):
    # names matching JSON writers/consumers
    H0_km_s_Mpc: float = 67.4
    Omega_de0: Optional[float] = None  # computed later (Ω_FRSR,0)


class FRSRSettings(BaseSettings):
    """
    Top-level typed settings for FRSR.
    Loads from environment and .env automatically.
    Override with env vars like:
        FRSR__ANCHORS__OMEGA_B0=0.049
    (note the double-underscore for nesting)
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="FRSR__",
        extra="ignore",
    )

    anchors: AnchorsSettings = AnchorsSettings()
    eos: EoSSettings = EoSSettings()
    inputs: InputsSettings = InputsSettings()
    class_params: ClassParamsSettings = ClassParamsSettings()

    use_tabulated_w: bool = False

    def to_params_dict(self) -> dict:
        """Emit a dict compatible with the existing params.json shape."""
        return {
            "mode": "eos",
            "use_tabulated_w": self.use_tabulated_w,
            "anchors": self.anchors.model_dump(),
            "eos": self.eos.model_dump(),
            "inputs": self.inputs.model_dump(),
            "class_params": self.class_params.model_dump(),
        }
