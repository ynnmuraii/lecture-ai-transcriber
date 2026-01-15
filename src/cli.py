#!/usr/bin/env python3

import sys
from pathlib import Path

def main():
    print("Lecture Transcriber")
    
    # Проверяем аргументы
    if len(sys.argv) < 2:
        print("\nИспользование: python -m src <video_file>")
        print("Пример: python -m src lecture.mp4")
        return
    
    input_file = sys.argv[1]
    
    # Проверяем файл
    if not Path(input_file).exists():
        print(f"\n❌ Файл не найден: {input_file}")
        return
    
    print(f"\n📁 Входной файл: {input_file}")
    print(f"📊 Размер файла: {Path(input_file).stat().st_size / (1024*1024):.1f} MB")
    
    # Показываем что будем делать
    print("\n🔄 Этапы обработки:")
    print("1. 🎵 Извлечение аудио из видео")
    print("2. 🎤 Транскрипция с помощью Whisper")
    print("3. 🧹 Очистка текста")
    print("4. 🔗 Склеивание сегментов")
    print("5. 📐 Форматирование формул")
    print("6. 📝 Генерация результата")
 
    print("✅ CLI готов для интеграции с компонентами")


if __name__ == "__main__":
    main()