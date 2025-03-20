#!/usr/bin/env python3
"""
Скрипт для проверки подключения к PostgreSQL с SSL.
"""

import os
import sys
import argparse
import psycopg2

def test_connection(host, port, dbname, user, password, sslmode="verify-full"):
    """
    Проверяет соединение с PostgreSQL базой данных.

    Args:
        host (str): Хост PostgreSQL сервера
        port (int): Порт PostgreSQL сервера
        dbname (str): Имя базы данных
        user (str): Имя пользователя
        password (str): Пароль пользователя
        sslmode (str): Режим SSL (по умолчанию verify-full)

    Returns:
        bool: True если соединение успешно, иначе False
    """
    try:
        # Создаем строку соединения
        conn_string = f"host={host} port={port} dbname={dbname} user={user} password={password} sslmode={sslmode}"

        # Проверяем существование сертификата
        root_cert = os.path.expanduser("~/.postgresql/root.crt")
        if os.path.exists(root_cert):
            conn_string += f" sslrootcert={root_cert}"
            print(f"Сертификат найден: {root_cert}")
        else:
            print("Предупреждение: Сертификат не найден!")
            print(f"Ожидаемый путь: {root_cert}")
            print("Попытка соединения без указания пути к сертификату.")

        print(f"Соединение с: {host}:{port}, БД: {dbname}, Пользователь: {user}, SSL mode: {sslmode}")
        conn = psycopg2.connect(conn_string)

        # Получаем версию PostgreSQL
        cursor = conn.cursor()
        cursor.execute('SELECT version();')
        version = cursor.fetchone()[0]
        print(f"Соединение установлено успешно!")
        print(f"Версия PostgreSQL: {version}")

        # Получаем информацию о SSL
        cursor.execute("SELECT ssl, version FROM pg_stat_ssl WHERE pid = pg_backend_pid();")
        ssl_info = cursor.fetchone()
        if ssl_info:
            ssl_enabled, ssl_version = ssl_info
            print(f"SSL соединение: {'Включено' if ssl_enabled else 'Выключено'}")
            print(f"SSL версия: {ssl_version}")
        else:
            print("Не удалось получить информацию о SSL соединении.")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Ошибка соединения: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Проверка соединения с PostgreSQL с использованием SSL")
    parser.add_argument("--host", required=True, help="Хост PostgreSQL")
    parser.add_argument("--port", type=int, default=6432, help="Порт PostgreSQL (по умолчанию 6432)")
    parser.add_argument("--dbname", required=True, help="Имя базы данных")
    parser.add_argument("--user", required=True, help="Имя пользователя")
    parser.add_argument("--password", required=True, help="Пароль")
    parser.add_argument("--sslmode", default="verify-full", choices=["disable", "allow", "prefer", "require", "verify-ca", "verify-full"],
                        help="Режим SSL (по умолчанию verify-full)")

    args = parser.parse_args()

    # Проверка соединения
    success = test_connection(
        args.host,
        args.port,
        args.dbname,
        args.user,
        args.password,
        args.sslmode
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
