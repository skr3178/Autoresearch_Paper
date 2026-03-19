from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.4"


def load_repo_env(env_file: str | os.PathLike[str] | None = None) -> Path | None:
    """Load simple KEY=VALUE pairs from the repo's .env file, if present."""
    candidate = Path(env_file).expanduser() if env_file else REPO_ROOT / ".env"
    if not candidate.exists():
        return None

    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue

        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)

    return candidate


@dataclass(frozen=True)
class IntelligenceConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    env_file: Path | None

    @classmethod
    def from_env(
        cls,
        env_file: str | os.PathLike[str] | None = None,
        require_api_key: bool = True,
    ) -> "IntelligenceConfig":
        loaded_env = load_repo_env(env_file)
        provider = os.environ.get(
            "AUTORESEARCH_INTELLIGENCE_PROVIDER",
            DEFAULT_PROVIDER,
        ).strip()
        model = os.environ.get(
            "AUTORESEARCH_INTELLIGENCE_MODEL",
            DEFAULT_MODEL,
        ).strip()
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")

        if require_api_key and provider == "openai" and not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Copy .env.example to .env and set "
                "OPENAI_API_KEY before running OpenAI-backed intelligence."
            )

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            env_file=loaded_env,
        )

    def redacted_dict(self) -> dict[str, str | None]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key_present": "yes" if self.api_key else "no",
            "base_url": self.base_url,
            "env_file": str(self.env_file) if self.env_file else None,
        }


if __name__ == "__main__":
    config = IntelligenceConfig.from_env(require_api_key=False)
    for key, value in config.redacted_dict().items():
        print(f"{key}={value}")
