import secrets
from pathlib import Path

import torch.nn
import yaml

from .config import PerthConfig


class CheckpointManager:
    def __init__(self, models_dir, run_name, dataset_hp: PerthConfig = None):
        self.save_path = Path(models_dir) / run_name
        self.save_path.mkdir(exist_ok=True, parents=True)

        self.hparams_file = self.save_path / "hparams.yaml"
        if self.hparams_file.exists():
            self.hp = self.load_hparams()
            if dataset_hp is not None:
                assert self.hp == dataset_hp
        else:
            assert dataset_hp is not None
            self.hp = dataset_hp
            self.save_hparams()

        self.id_file = self.save_path / "id.txt"
        if self.id_file.exists():
            self.id = self.id_file.read_text().strip()
        else:
            self.id = secrets.token_urlsafe(16)
            self.id_file.write_text(self.id)

    def _get_step_from_path(self, path: Path, ext: str = ".pth.tar") -> int:
        """Extract training step from checkpoint filename."""
        name = path.name.replace(ext, "")
        return int(name.split("_")[-1])

    def load_latest(self, ext: str = ".pth.tar"):
        ckpts = [
            p
            for p in self.save_path.iterdir()
            if p.name.endswith(ext) and p.name.startswith("perth_net_")
        ]
        if not ckpts:
            return None
        latest = max(ckpts, key=self._get_step_from_path)
        return torch.load(latest, map_location="cpu")

    def load_hparams(self):
        with self.hparams_file.open("r") as hp_file:
            return PerthConfig(**yaml.load(hp_file, Loader=yaml.FullLoader))

    def save_hparams(self):
        with self.hparams_file.open("w") as f:
            f.write(yaml.dump(self.hp._asdict()))

    def save_model(self, model, step: int):
        state = {
            "model": model.state_dict()
            if isinstance(model, torch.nn.Module)
            else model,
            "step": step,
        }
        checkpoint_fpath = self.save_path / f"perth_net_{step:06d}.pth.tar"
        try:
            torch.save(state, checkpoint_fpath)
        except KeyboardInterrupt:
            checkpoint_fpath.unlink(missing_ok=True)
