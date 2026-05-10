output "master_public_ip" {
  description = "Public IP of the Kubernetes master node"
  value       = aws_instance.k8s_master.public_ip
}

output "master_private_ip" {
  description = "Private IP of the master (needed for kubeadm join)"
  value       = aws_instance.k8s_master.private_ip
}

output "worker_public_ip" {
  description = "Public IP of the Kubernetes worker node"
  value       = aws_instance.k8s_worker.public_ip
}

output "worker_private_ip" {
  description = "Private IP of the worker (needed for Prometheus config)"
  value       = aws_instance.k8s_worker.private_ip
}

output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.k8s_sg.id
}
