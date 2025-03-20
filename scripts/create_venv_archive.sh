#!/bin/bash
# Скрипт для упаковки существующего виртуального окружения в архив для PySpark

set -e

# Определяем директории
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"
VENVS_DIR="${PROJECT_ROOT}/venvs"
ARCHIVE_NAME="fraud_detection_venv.tar.gz"
ARCHIVE_PATH="${VENVS_DIR}/${ARCHIVE_NAME}"

# Проверяем существование .venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Ошибка: Виртуальное окружение .venv не найдено в ${PROJECT_ROOT}"
    echo "Убедитесь, что у вас активировано виртуальное окружение и установлены все зависимости"
    exit 1
fi

# Создаем директорию для архива, если ее нет
mkdir -p "$VENVS_DIR"

# Создаем архив, исключая ненужные файлы для уменьшения размера
echo "Создаем архив из существующего виртуального окружения ${VENV_DIR}..."

cd "${PROJECT_ROOT}"
tar --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='*.dist-info' \
    --exclude='*.egg-info' \
    --exclude='pip-selfcheck.json' \
    --exclude='.venv/bin/activate*' \
    --exclude='.venv/bin/pip*' \
    --exclude='.venv/bin/wheel*' \
    --exclude='.venv/bin/easy_install*' \
    --exclude='.venv/bin/python*-config' \
    -czf "${ARCHIVE_PATH}" .venv

echo "Архив создан: ${ARCHIVE_PATH}"
echo "Размер архива: $(du -h "${ARCHIVE_PATH}" | cut -f1)"
echo "Готово!" 