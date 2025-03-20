#!/bin/bash

function log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')]: $1"
}

# Обновляем пакеты
log "Updating packages"
apt-get update
apt-get upgrade -y

# Устанавливаем необходимые пакеты
log "Installing required packages"
apt-get install -y python3-pip python3-venv nginx

# Создаем пользователя mlflow
log "Creating mlflow user"
useradd -m -s /bin/bash mlflow

# Создаем виртуальное окружение для MLflow
log "Setting up Python virtual environment"
sudo -u mlflow bash -c "mkdir -p /home/mlflow/venv"
sudo -u mlflow bash -c "python3 -m venv /home/mlflow/venv"
sudo -u mlflow bash -c "/home/mlflow/venv/bin/pip install --upgrade pip"
sudo -u mlflow bash -c "/home/mlflow/venv/bin/pip install mlflow==2.21.0 psycopg-binary==3.2.6 boto3==1.37.16"

# Копируем конфигурационный файл
log "Copying configuration file"
cp /home/ubuntu/mlflow.conf /home/mlflow/mlflow.conf
chown mlflow:mlflow /home/mlflow/mlflow.conf

# Копируем systemd сервис
log "Setting up systemd service"
cp /home/ubuntu/mlflow.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable mlflow.service
systemctl start mlflow.service

# Настраиваем Nginx как прокси
log "Setting up Nginx"
cat > /etc/nginx/sites-available/mlflow << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

ln -sf /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
systemctl restart nginx

log "MLflow setup completed successfully" 