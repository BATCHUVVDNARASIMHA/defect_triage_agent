data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name = var.cluster_name
  azs  = slice(data.aws_availability_zones.available.names, 0, 2)
}

# ---------------------------------------------------------------------------
# Network: two AZs, private subnets for nodes and the database, one NAT
# gateway (cheaper; not highly available, which is fine for this project).
# ---------------------------------------------------------------------------
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.17"

  name = "${local.name}-vpc"
  cidr = "10.40.0.0/16"
  azs  = local.azs

  private_subnets  = ["10.40.1.0/24", "10.40.2.0/24"]
  public_subnets   = ["10.40.101.0/24", "10.40.102.0/24"]
  database_subnets = ["10.40.201.0/24", "10.40.202.0/24"]

  create_database_subnet_group = true
  enable_nat_gateway           = true
  single_nat_gateway           = true

  public_subnet_tags  = { "kubernetes.io/role/elb" = 1 }
  private_subnet_tags = { "kubernetes.io/role/internal-elb" = 1 }
}

# ---------------------------------------------------------------------------
# EKS cluster with one managed node group.
# ---------------------------------------------------------------------------
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.31"

  cluster_name    = local.name
  cluster_version = var.kubernetes_version

  cluster_endpoint_public_access           = true
  enable_cluster_creator_admin_permissions = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    default = {
      instance_types = [var.node_instance_type]
      min_size       = 1
      max_size       = 3
      desired_size   = var.node_desired_size
    }
  }

  # The GitHub Actions role can deploy into the app namespace only.
  access_entries = {
    github_actions = {
      principal_arn = aws_iam_role.github_actions.arn

      policy_associations = {
        app_namespace = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSEditPolicy"

          access_scope = {
            type       = "namespace"
            namespaces = [var.namespace]
          }
        }
      }
    }
  }
}

# ---------------------------------------------------------------------------
# Container registry
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "app" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true # lets `terraform destroy` remove it even with images inside

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 10 most recent images"

      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }

      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# Postgres 16 on RDS (pgvector is available as an extension; the ingest job
# runs CREATE EXTENSION). Encrypted at rest, private subnets only, and the
# master password is generated and held in Secrets Manager by RDS, so it
# never appears in Terraform state.
# ---------------------------------------------------------------------------
resource "aws_security_group" "db" {
  name_prefix = "${local.name}-db-"
  description = "Postgres access from EKS nodes only"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "Postgres from EKS worker nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_db_instance" "pg" {
  identifier     = "${local.name}-pg"
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.db_instance_class

  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true

  db_name                     = "triage"
  username                    = "triage"
  manage_master_user_password = true

  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false

  backup_retention_period = 1
  skip_final_snapshot     = true
  deletion_protection     = false
  apply_immediately       = true
}
