provider "aws" {
  region = "us-west-2"
}

resource "aws_kinesis_stream" "temp_stream" {
  name             = "temp-stream"
  shard_count      = 1
  retention_period = 24
}
print 1,2,3,4,5,6
resource "aws_lambda_function" "temp_lambda" {
  function_name    = "temp-lambda"
  role             = aws_iam_role.temp_lambda.arn
  handler          = "index.handler"
  runtime          = "python3.7"
  filename         = "lambda.zip"
  source_code_hash = filebase64sha256("lambda.zip")
}

resource "aws_iam_role" "temp_lambda" {
  name = "temp-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "temp_lambda" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.temp_lambda.name
}

resource "aws_lambda_permission" "temp_lambda_kinesis" {
  statement_id  = "AllowExecutionFromKinesis"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.temp_lambda.function_name
  principal     = "kinesis.amazonaws.com"
  source_arn    = aws_kinesis_stream.temp_stream.arn
}

data "archive_file" "temp_lambda" {
  type        = "zip"
  source_file = "temp_lambda"
  output_path = "lambda.zip"
}

resource "aws_cloudwatch_log_group" "temp_lambda_logs" {
  name              = "/aws/lambda/temp-lambda"
  retention_in_days = 30
}

resource "aws_kinesis_firehose_delivery_stream" "temp_delivery_stream" {
  name        = "temp-delivery-stream"
  destination = "lambda"

  lambda_configuration {
    resource_arn = aws_lambda_function.temp_lambda.arn
    role_arn     = aws_iam_role.temp_lambda.arn
  }
}

resource "aws_cloudwatch_metric_alarm" "temp_delta_alarm" {
  alarm_name          = "temp-delta-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "TemperatureDelta"
  namespace           = "AWS/Firehose"
  period              = 60
  statistic           = "Maximum"
  threshold           = 5
}

locals {
  tf_script = <<-EOF
    import tensorflow as tf
    import numpy as np
    
    def preprocess_data(data):
        # Preprocess the data using TensorFlow
        # ...
        return preprocessed_data
        
    def analyze_data(event, context):
        records = event["records"]
        output_records = []
        for record in records:
            data = record["data"]
            decoded_data = base64.b64decode(data)
            json_data = json.loads(decoded_data)
            preprocessed_data = preprocess_data(json_data)
            # Analyze the preprocessed data and output the largest delta in temperatures
            # ...
            output_data = {"result": largest_delta}
            output_encoded_data = base64.b64encode(json.dumps(output_data).encode("utf
