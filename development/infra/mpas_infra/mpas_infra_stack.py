from pathlib import Path

from aws_cdk import (
    Fn as fn,
    Stack,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
)
from constructs import Construct


def read_text(file_path: str, encoding="utf-8"):
    text = ""
    if file_path is not None:
        text = Path(file_path).read_text(encoding=encoding)
    return text


class MpasInfraStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        on_create_script_path=None,
        on_start_script_path=None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here

        onCreateScript = read_text(on_create_script_path)

        onStartScript = read_text(on_start_script_path)

        role = iam.Role(
            self,
            "MpasRole",
            **dict(
                assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "AmazonSageMakerFullAccess"
                    ),
                ],
            ),
        )

        lifecycle_config = sagemaker.CfnNotebookInstanceLifecycleConfig(
            self,
            "MpasLifecycleConfig",
            **dict(
                notebook_instance_lifecycle_config_name="MpasLifecycleConfig",
                # on_create=[
                #     dict(
                #         content_type="text/x-shellscript",
                #         content=fn.base64(onCreateScript),
                #     ),
                # ],
                # on_start=[
                #     dict(
                #         content_type="text/x-shellscript",
                #         content=fn.base64(onStartScript),
                #     ),
                # ],
            ),
        )

        model_packing_and_shipping_repository = "FastAIFastBookRepository"

        sagemaker.CfnCodeRepository(
            self,
            "MpasFastAICodeRepository",
            code_repository_name=model_packing_and_shipping_repository,
            git_config=sagemaker.CfnCodeRepository.GitConfigProperty(
                branch="main",
                repository_url="https://github.com/chiefkemist/model_packing_and_shipping.git",
            ),
        )

        sagemaker.CfnNotebookInstance(
            self,
            "MpasNotebookCPU",
            **dict(
                lifecycle_config_name=lifecycle_config.notebook_instance_lifecycle_config_name,
                role_arn=role.role_arn,
                instance_type="ml.c5.xlarge",  # "ml.p2.xlarge", # "ml.c5.2xlarge",
                additional_code_repositories=[
                    model_packing_and_shipping_repository,
                ],
            ),
        )

        # sagemaker.CfnNotebookInstance(
        #     self, "MpasNotebookGPU", **dict(
        #         lifecycle_config_name=lifecycle_config.notebook_instance_lifecycle_config_name,
        #         role_arn=role.role_arn,
        #         instance_type="ml.p2.xlarge",  # "ml.c5.xlarge", # "ml.c5.2xlarge",
        #         additional_code_repositories=[julia_academy_datascience_repository],
        #     ),
        # )
