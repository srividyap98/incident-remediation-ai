terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
  }
  backend "s3" {
    bucket = "incident-ai-tfstate"
    key    = "incident-ai/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ──────────────────────────────────────────────────────────────────
variable "aws_region"   { default = "us-east-1" }
variable "project_name" { default = "incident-ai" }
variable "environment"  { default = "prod" }

locals {
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ── IAM: Bedrock access role (used by EKS IRSA + Lambda) ─────────────────────
resource "aws_iam_role" "incident_ai_bedrock_role" {
  name = "${var.project_name}-bedrock-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "lambda.amazonaws.com" }
        Action    = "sts:AssumeRole"
      },
      {
        # IRSA for EKS pods
        Effect    = "Allow"
        Principal = { Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${var.eks_oidc_provider}" }
        Action    = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.eks_oidc_provider}:sub" = "system:serviceaccount:incident-ai:incident-ai-sa"
          }
        }
      }
    ]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "bedrock_policy" {
  name = "bedrock-invoke"
  role = aws_iam_role.incident_ai_bedrock_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.knowledge_base.arn,
          "${aws_s3_bucket.knowledge_base.arn}/*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem"]
        Resource = aws_dynamodb_table.results.arn
      }
    ]
  })
}

# ── S3: Knowledge base bucket ─────────────────────────────────────────────────
resource "aws_s3_bucket" "knowledge_base" {
  bucket = "${var.project_name}-knowledge-base-${data.aws_caller_identity.current.account_id}"
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "knowledge_base" {
  bucket = aws_s3_bucket.knowledge_base.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "knowledge_base" {
  bucket = aws_s3_bucket.knowledge_base.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

# ── DynamoDB: Results table ───────────────────────────────────────────────────
resource "aws_dynamodb_table" "results" {
  name         = "${var.project_name}-results"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "incident_id"

  attribute {
    name = "incident_id"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = local.tags
}

# ── SQS: Async incident queue ─────────────────────────────────────────────────
resource "aws_sqs_queue" "incident_queue" {
  name                       = "${var.project_name}-incidents.fifo"
  fifo_queue                 = true
  content_based_deduplication = true
  visibility_timeout_seconds = 300
  message_retention_seconds  = 86400
  tags                       = local.tags
}

# ── SNS: Alerts ───────────────────────────────────────────────────────────────
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
  tags = local.tags
}

# ── Data sources ──────────────────────────────────────────────────────────────
data "aws_caller_identity" "current" {}
variable "eks_oidc_provider" {
  description = "EKS OIDC provider URL (without https://)"
  default     = ""
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "knowledge_base_bucket" { value = aws_s3_bucket.knowledge_base.bucket }
output "results_table_name"    { value = aws_dynamodb_table.results.name }
output "incident_queue_url"    { value = aws_sqs_queue.incident_queue.url }
output "bedrock_role_arn"      { value = aws_iam_role.incident_ai_bedrock_role.arn }
