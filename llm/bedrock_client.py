import boto3
import logging
from typing import Generator
from botocore.exceptions import ClientError, BotoCoreError
from time import sleep

logger = logging.getLogger(__name__)

class BedrockClient:
    """Wrapper for AWS Bedrock models using the Converse API."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        max_retries: int = 3
    ):
        self.region = region
        self.model_id = model_id
        self.max_retries = max_retries
        try:
            self.client = boto3.client("bedrock-runtime", region_name=region)
            logger.info(f"Bedrock client initialized for region: {region}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def invoke(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Invoke model and get response using the Converse API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inference_config={"maxTokens": max_tokens, "temperature": temperature}
                )
                return response['output']['message']['content'][0]['text']
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException' and attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Throttled, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    sleep(wait_time)
                else:
                    logger.error(f"Bedrock error: {error_code} - {e}")
                    raise
            except BotoCoreError as e:
                logger.error(f"Botocore error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in invoke: {e}")
                raise
        raise RuntimeError("Max retries exceeded")

    def invoke_stream(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> Generator:
        """Stream model response token by token with error handling."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.converse_stream(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inference_config={"maxTokens": max_tokens, "temperature": temperature}
                )

                for event in response['stream']:
                    if 'contentBlockDelta' in event:
                        yield event['contentBlockDelta']['delta']['text']
                return
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException' and attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Throttled in stream, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    sleep(wait_time)
                else:
                    logger.error(f"Bedrock stream error: {error_code} - {e}")
                    raise
            except BotoCoreError as e:
                logger.error(f"Botocore stream error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in invoke_stream: {e}")
                raise
        raise RuntimeError("Max retries exceeded in streaming")