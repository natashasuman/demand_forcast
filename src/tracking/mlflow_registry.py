import os
import boto3

from src.config import settings


class SageMakerRegistrar:
    """Handles registration and lineage tracking of champion models in AWS SageMaker."""

    def __init__(self, role: str | None = None, region_name: str | None = None):
        self.role = role or settings.SAGEMAKER_EXECUTION_ROLE
        self.region_name = region_name or settings.AWS_REGION
        self._sagemaker_client = None

    @property
    def sagemaker_client(self):
        if self._sagemaker_client is None:
            self._sagemaker_client = boto3.client("sagemaker", region_name=self.region_name)
        return self._sagemaker_client

    def register_model_artifact(
        self,
        model_s3_uri: str,
        model_name: str,
        image_uri: str | None = None,
        dry_run: bool = False,
    ) -> str:
        """
        Registers a trained model artifact into the SageMaker Model Registry.
        Supports dry_run for local validation without live AWS credentials.
        """
        if dry_run or os.getenv("SAGEMAKER_DRY_RUN", "false").lower() in ("true", "1"):
            mock_arn = f"arn:aws:sagemaker:{self.region_name}:123456789012:model/{model_name}"
            print(f"[+] [DRY RUN] Registering model '{model_name}' from '{model_s3_uri}' in SageMaker.")
            print(f"[+] Mock SageMaker Model ARN: {mock_arn}")
            return mock_arn

        try:
            resolved_image = (
                image_uri
                or f"683313688378.dkr.ecr.{self.region_name}.amazonaws.com/sagemaker-xgboost:1.5-1"
            )
            container_def = {"Image": resolved_image, "ModelDataUrl": model_s3_uri}

            create_model_response = self.sagemaker_client.create_model(
                ModelName=model_name,
                ExecutionRoleArn=self.role,
                PrimaryContainer=container_def,
            )
            model_arn = create_model_response["ModelArn"]
            print(f"[+] Registered model in SageMaker Registry: {model_name} (ARN: {model_arn})")
            return model_arn
        except Exception as e:
            print(f"[-] SageMaker Registration failed: {e}")
            raise
