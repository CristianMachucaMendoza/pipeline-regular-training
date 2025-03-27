#!/bin/bash
# Скрипт для создания самодостаточного виртуального окружения и упаковки его в архив для PySpark

set -e

# Определяем директории
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"
TEMP_VENV_DIR="${PROJECT_ROOT}/.temp_venv"
VENVS_DIR="${PROJECT_ROOT}/venvs"
ARCHIVE_NAME="venv.tar.gz"
ARCHIVE_PATH="${VENVS_DIR}/${ARCHIVE_NAME}"

# Функция очистки при выходе
cleanup() {
    echo "Очистка временных файлов..."
    rm -rf "$TEMP_VENV_DIR"
    echo "Очистка завершена."
}

# Устанавливаем обработчик выхода
trap cleanup EXIT

# Создаем директорию для архива, если ее нет
mkdir -p "$VENVS_DIR"

# Проверяем наличие virtualenv
if ! command -v virtualenv &> /dev/null; then
    echo "Установка virtualenv..."
    pip install virtualenv
fi

# Проверяем требования к файлу requirements.txt
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Предупреждение: requirements.txt не найден в ${PROJECT_ROOT}"
    echo "Попытка создать requirements.txt из текущего окружения..."
    
    if [ -d "$VENV_DIR" ]; then
        "${VENV_DIR}/bin/pip" freeze > "$REQUIREMENTS_FILE"
        echo "Создан requirements.txt из текущего виртуального окружения"
    else
        echo "Ошибка: Виртуальное окружение .venv не найдено и requirements.txt отсутствует"
        echo "Создайте requirements.txt или активируйте виртуальное окружение"
        exit 1
    fi
fi

# Создаем новое виртуальное окружение с копированием файлов (не символическими ссылками)
echo "Создаем новое самодостаточное виртуальное окружение в ${TEMP_VENV_DIR}..."
virtualenv --always-copy "$TEMP_VENV_DIR"

# Устанавливаем зависимости в новое окружение
echo "Устанавливаем зависимости из requirements.txt..."
"${TEMP_VENV_DIR}/bin/pip" install -r "$REQUIREMENTS_FILE"

# Проверяем, что интерпретатор Python действительно является файлом, а не символической ссылкой
PYTHON_BIN="${TEMP_VENV_DIR}/bin/python"
if [ -L "$PYTHON_BIN" ]; then
    echo "Предупреждение: Python в виртуальном окружении все еще является символической ссылкой"
    ls -la "$PYTHON_BIN"
    echo "Убедитесь, что команда virtualenv --always-copy работает правильно в вашей системе"
fi

# Создаем архив, исключая ненужные файлы для уменьшения размера
echo "Создаем архив из подготовленного виртуального окружения ${TEMP_VENV_DIR}..."

cd "${TEMP_VENV_DIR}"
tar --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='*.dist-info' \
    --exclude='*.egg-info' \
    --exclude='pip-selfcheck.json' \
    -czf "${ARCHIVE_PATH}" .

echo "Проверяем содержимое архива..."
tar -tvf "${ARCHIVE_PATH}" | grep "bin/python"

echo "Архив создан: ${ARCHIVE_PATH}"
echo "Размер архива: $(du -h "${ARCHIVE_PATH}" | cut -f1)"
echo "Готово!"