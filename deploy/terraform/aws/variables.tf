variable "region" {
  description = "AWS region for every resource."
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "Name for the EKS cluster; also prefixes the other resources."
  type        = string
  default     = "defect-triage"
}

variable "kubernetes_version" {
  description = "EKS Kubernetes version."
  type        = string
  default     = "1.31"
}

variable "node_instance_type" {
  description = "Instance type for the managed node group."
  type        = string
  default     = "t3.medium"
}

variable "node_desired_size" {
  description = "Number of worker nodes."
  type        = number
  default     = 2
}

variable "db_instance_class" {
  description = "RDS instance class for Postgres."
  type        = string
  default     = "db.t4g.micro"
}

variable "namespace" {
  description = "Kubernetes namespace the app is deployed into."
  type        = string
  default     = "defect-triage"
}

variable "ecr_repository_name" {
  description = "ECR repository for the API image."
  type        = string
  default     = "defect-triage-agent"
}

variable "github_repository" {
  description = "owner/repo allowed to assume the deploy role via GitHub OIDC."
  type        = string
  default     = "BATCHUVVDNARASIMHA/defect_triage_agent"
}

variable "create_github_oidc_provider" {
  description = "Set to false if this AWS account already has the GitHub Actions OIDC provider."
  type        = bool
  default     = true
}
