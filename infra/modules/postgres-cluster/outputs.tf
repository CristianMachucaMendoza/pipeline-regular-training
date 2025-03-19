output "cluster_id" {
  value = yandex_mdb_postgresql_cluster.postgres_cluster.id
}

output "fqdn" {
  value = yandex_mdb_postgresql_cluster.postgres_cluster.host[0].fqdn
}

output "postgres_connection_string" {
  value     = "postgresql://${var.postgres_user}:${var.postgres_password}@${yandex_mdb_postgresql_cluster.postgres_cluster.host[0].fqdn}:6432/${var.postgres_db}"
  sensitive = true
}

output "postgres_host" {
  value = yandex_mdb_postgresql_cluster.postgres_cluster.host[0].fqdn
}

output "postgres_port" {
  value = 6432
}

output "postgres_db" {
  value = var.postgres_db
}

output "postgres_user" {
  value = var.postgres_user
}
