variable "region" {
  description = "AWS region"
  default     = "ap-south-1"
}

variable "ami_id" {
  description = "Ubuntu 22.04 LTS AMI ID for ap-south-1"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  default     = "c7i-flex.large"
}

variable "key_pair_name" {
  description = "kubernetespair"
  type        = string
}
