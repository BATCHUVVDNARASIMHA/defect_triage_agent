output "region" {
  value = var.region
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "configure_kubectl" {
  description = "Run this to point kubectl at the cluster."
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.region}"
}

output "ecr_repository_name" {
  value = aws_ecr_repository.app.name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "db_endpoint" {
  description = "RDS hostname (private; reachable from the EKS nodes)."
  value       = aws_db_instance.pg.address
}

output "db_master_secret_arn" {
  description = "Secrets Manager secret holding the RDS master credentials."
  value       = aws_db_instance.pg.master_user_secret[0].secret_arn
}

output "github_actions_role_arn" {
  description = "Set this as the AWS_ROLE_ARN secret in the GitHub repo."
  value       = aws_iam_role.github_actions.arn
}
