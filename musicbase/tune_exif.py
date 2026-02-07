import pandas as pd
import sys
from io import StringIO

def is_effectively_empty(val) -> bool:
    """
    Определяет, является ли значение «пустым» в расширенном смысле:
    - NaN / None
    - пустая строка или только пробелы
    - строка, состоящая ТОЛЬКО из знаков '?' (возможно с пробелами)
    """
    if pd.isna(val):
        return True
    s = str(val).strip()
    if not s:
        return True
    # Проверяем: состоит ли строка исключительно из символов '?'
    return all(c == '?' for c in s)

def read_csv_with_fallback_encoding(
    filepath: str,
    encodings: list = None
) -> pd.DataFrame:
    """
    Безопасное чтение CSV с обработкой ошибок кодировки.
    При неудаче — построчное чтение с пропуском битых строк.
    """
    if encodings is None:
        #encodings = ['utf-8-sig', 'utf-8', 'cp1251', 'latin1']
        encodings = ['utf-8-sig', 'utf-8']
    
    # Этап 1: пробуем стандартные кодировки
    for enc in encodings:
        try:
            df = pd.read_csv(
                filepath,
                dtype=str,
                na_values=['', ' ', 'None'],
                keep_default_na=False,
                encoding=enc,
                encoding_errors='strict'
            )
            print(f"✓ Файл прочитан в кодировке: {enc}", file=sys.stderr)
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    
    # Этап 2: построчное чтение с пропуском ошибок
    print("⚠ Стандартные кодировки не сработали. Читаю построчно...", file=sys.stderr)
    valid_lines = []
    invalid_count = 0
    total_lines = 0
    
    with open(filepath, 'rb') as f:
        for line_bytes in f:
            total_lines += 1
            try:
                line = line_bytes.decode('utf-8')
                valid_lines.append(line)
            except UnicodeDecodeError:
                invalid_count += 1
                continue
    
    if not valid_lines:
        raise RuntimeError(f"Не удалось прочитать ни одной валидной строки из {filepath}")
    
    print(f"✓ Пропущено строк с ошибками кодировки: {invalid_count} из {total_lines}", file=sys.stderr)
    csv_content = ''.join(valid_lines)
    return pd.read_csv(
        StringIO(csv_content),
        dtype=str,
        na_values=['', ' ', 'None'],
        keep_default_na=False,
        encoding='utf-8'
    )

def process_exiftool_csv(input_csv: str, output_csv: str):
    """
    Обрабатывает CSV от exiftool:
    - пропускает строки с ошибками кодировки
    - удаляет изображения (jpg, png, gif, webp и др.)
    - удаляет записи, где Artist/Album/Title пустые ИЛИ состоят только из '?'
    - сохраняет только нужные поля
    """
    # 1. Безопасное чтение
    df = read_csv_with_fallback_encoding(input_csv)
    df.columns = df.columns.str.strip()
    
    # 2. Проверка обязательных колонок
    required_cols = {'SourceFile', 'FileName', 'Artist', 'Album', 'Title', 'Genre'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}. "
                         f"Запустите exiftool с: -csv -SourceFile -FileName -Artist -Album -Title -Genre")
    
    # 3. Фильтрация изображений (регистронезависимо)
    image_exts = r'\.(jpg|jpeg|png|gif|webp|bmp|tiff?|svg|ico|html|htm|xml|swf)$'
    mask_images = df['SourceFile'].str.lower().str.contains(image_exts, regex=True, na=False)
    removed_images = mask_images.sum()
    df = df[~mask_images].copy()
    
    # 4. Фильтрация записей без валидных метаданных
    mask_bad_metadata = (
        df['Artist'].apply(is_effectively_empty) &
        df['Album'].apply(is_effectively_empty) &
        df['Title'].apply(is_effectively_empty)
    )
    removed_bad_metadata = mask_bad_metadata.sum()
    df = df[~mask_bad_metadata].copy()
    
    # 5. Оставить только нужные поля
    keep_cols = ['SourceFile', 'FileName', 'Artist', 'Album', 'Title', 'Genre']
    df = df[keep_cols].copy()
    
    # 6. Очистка значений
    for col in ['Artist', 'Album', 'Title', 'Genre']:
        df[col] = df[col].fillna('').str.strip()
    
    # 7. Сохранение результата
    df.to_csv(
        output_csv,
        index=False,
        encoding='utf-8-sig',
        quoting=1  # csv.QUOTE_ALL
    )
    
    # Статистика
    print(f"\n✅ Обработка завершена", file=sys.stderr)
    print(f"   Всего после чтения:      {len(df) + removed_images + removed_bad_metadata}", file=sys.stderr)
    print(f"   Удалено изображений:     {removed_images}", file=sys.stderr)
    print(f"   Удалено записей с ???/пусто: {removed_bad_metadata}", file=sys.stderr)
    print(f"   Сохранено валидных:      {len(df)}", file=sys.stderr)
    print(f"📁 Результат: {output_csv}", file=sys.stderr)
    return df

# Пример использования:
if __name__ == "__main__":
    # Сначала получите CSV от exiftool:
    # exiftool -csv -r -SourceFile -FileName -Artist -Album -Title -Genre \
    #          -Track -Year -Duration -Bitrate /путь/к/медиа > media_raw.csv
    
    if len(sys.argv) == 3:
        #print(f"Аргумент: {sys.argv[1]}")
        input_csv_filename = sys.argv[1]
        output_csv_filename = sys.argv[2]
    else:
        print("Аргументы не переданы")
        sys.exit()
        
    
    df_clean = process_exiftool_csv(input_csv_filename, output_csv_filename)
