import os
from typing import Optional

import requests

from voicehub.providers.base import VoiceHubProvider


class DeepInfraProvider(VoiceHubProvider):

    MODEL_MAPPING = {
        "orpheustts": "canopylabs/orpheus-3b-0.1-ft",
        "kokoro": "hexgrad/Kokoro-82M",
        "sesame": "sesame/csm-1b",
        "zonos": "Zyphra/Zonos-v0.1-transformer"
    }

    BASE_URL = "https://api.deepinfra.com/v1/openai/audio/speech"

    def __init__(
        self,
        *,
        model_name: str,
        api_key: Optional[str] = None,
    ):
        if not self._check_for_model(model_name):
            raise ValueError(
                f"Model '{model_name}' is not available in DeepInfra model mapping. Available models: {list(self.MODEL_MAPPING.keys())}"
            )
        if not api_key:
            self.api_key = self._maybe_get_api_key()
            if not self.api_key:
                raise ValueError("API key is required for DeepInfra provider.")
        else:
            self.api_key = api_key
        self.model_name = model_name

    def _maybe_get_api_key(self) -> Optional[str]:
        os.environ.setdefault("DEEPINFRA_API_KEY", "")
        api_key = os.getenv("DEEPINFRA_API_KEY")
        return api_key

    def _check_for_model(self, model_name: str) -> bool:
        """
        Check if the specified model is available in the DeepInfra model mapping.

        Args:
            model_name: The name of the model to check.

        Returns:
            True if the model is available, False otherwise.
        """
        return model_name in self.MODEL_MAPPING

    def __call__(self, text: str, output_file: Optional[str] = None, **kwargs):
        """
        Call the DeepInfra model with the provided text and additional parameters.

        Args:
            text: The input text to be processed by the model.
            **kwargs: Additional parameters for the model call.

        Returns:
            The response from the DeepInfra model.
        """

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.MODEL_MAPPING[self.model_name], "input": text, **kwargs}

        response = requests.post(self.BASE_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Error calling DeepInfra API: {response.text}")
        if output_file:
            with open(output_file, "wb") as f:
                f.write(response.content)

        return response.content
