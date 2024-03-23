import aws_cdk as core
import aws_cdk.assertions as assertions

from mpas_infra.mpas_infra_stack import MpasInfraStack


# example tests. To run these tests, uncomment this file along with the example
# resource in genai_infra/genai_infra_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = MpasInfraStack(app, "mpas-infra")
    template = assertions.Template.from_stack(stack)


#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
