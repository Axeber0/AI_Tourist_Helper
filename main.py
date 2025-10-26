import asyncio
import logging
import urllib.parse
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import re

# === НАСТРОЙКИ ===
BOT_TOKEN = '...'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

geolocator = Nominatim(user_agent="nnwalk_bot_v1")
_last_nominatim_call = 0.0
_NOMINATIM_DELAY = 1.1

async def safe_geocode(query: str):
    global _last_nominatim_call
    now = time.time()
    wait_time = _NOMINATIM_DELAY - (now - _last_nominatim_call)
    if wait_time > 0:
        await asyncio.sleep(wait_time)
        _last_nominatim_call = time.time()
    else:
        _last_nominatim_call = now

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: geolocator.geocode(query, timeout=10))
    except Exception as e:
        logger.error(f"Nominatim error: {e}")
        return None

# === ЗАГРУЗКА ОБЪЕКТОВ ИЗ EXCEL ===
def load_cultural_objects(file_path: str):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        logger.info(f"Загружены столбцы: {list(df.columns)}")
        logger.info(f"Всего строк: {len(df)}")

        required_cols = ['coordinate', 'title', 'address']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют столбцы: {missing}. Есть: {list(df.columns)}")

        df = df.dropna(subset=required_cols)
        objects = []

        for _, row in df.iterrows():
            try:
                coord_str = str(row['coordinate']).strip()
                if not coord_str.startswith('POINT'):
                    continue
                parts = coord_str.split()
                if len(parts) < 3:
                    continue
                lon = float(parts[1].replace('(', ''))
                lat = float(parts[2].replace(')', ''))
                category = int(row.get('category_id', 0)) if pd.notna(row.get('category_id')) else 0
                description = str(row.get('description', '')).strip()
                objects.append({
                    'id': row.get('id'),
                    'title': str(row['title']).strip(),
                    'address': str(row['address']).strip(),
                    'description': description,
                    'lat': lat,
                    'lon': lon,
                    'category_id': category
                })
            except Exception as e:
                logger.warning(f"Ошибка парсинга объекта {row.get('id', 'unknown')}: {e}")
        logger.info(f"Загружено объектов: {len(objects)}")
        return objects
    except Exception as e:
        logger.error(f"Ошибка загрузки Excel: {e}")
        return []

CULTURAL_OBJECTS = load_cultural_objects('cultural_objects_mnn.xlsx')

# === СООТВЕТСТВИЕ ИНТЕРЕСОВ → КАТЕГОРИЙ ===
INTEREST_TO_CATEGORY = {
    'парки': [2],
    'памятники': [1],
    'история': [1, 5, 7],
    'музеи': [7],
    'театры': [8],
    'культура': [6, 7, 8],
    'архитектура': [5],
    'развлечение': [3, 6, 8],
}

# === FSM ===
class RouteForm(StatesGroup):
    interests = State()
    time = State()
    location = State()

bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def clean_description_brief(desc: str) -> str:
    """Оставляет только полные предложения (до последней точки/восклицания/вопроса)."""
    if not desc or desc.strip().lower() in {'-', 'нет', 'none', '', 'памятник', 'здание', 'сооружение'}:
        return ""

    # Убираем HTML
    desc = re.sub(r'<[^>]+>', ' ', desc)
    desc = re.sub(r'\s+', ' ', desc).strip()


    # Ищем последнее завершённое предложение
    last_end = -1
    for i in range(len(desc)):
        if desc[i] in '.!?':
            last_end = i + 1  # включаем знак препинания

    if last_end <= 0:
        return ""  # нет полного предложения

    result = desc[:last_end].strip()

    # Ограничиваем длину (например, до 400 символов), но только по границе предложения
    if len(result) > 400:
        # Ищем последнюю точку до 400-го символа
        cutoff = result[:400].rfind('.')
        if cutoff == -1:
            cutoff = result[:400].rfind('!')
        if cutoff == -1:
            cutoff = result[:400].rfind('?')
        if cutoff != -1:
            result = result[:cutoff + 1].strip()
        else:
            result = ""  # если нет завершённого предложения — не показываем

    return result

def build_yandex_route_url(points):
    valid = [f"{lat:.5f},{lon:.5f}" for lat, lon in points if lat and lon]
    if len(valid) < 2:
        return "https://yandex.ru/maps/"
    rtext = "~".join(valid)
    return f"https://yandex.ru/maps/?rtext={urllib.parse.quote(rtext)}&rtt=mt"

def generate_route_from_data(interests: str, time_hours: float, start_lat: float, start_lon: float) -> str:
    requested_categories = set()
    interests_lower = interests.lower()
    for keyword, cats in INTEREST_TO_CATEGORY.items():
        if keyword in interests_lower:
            requested_categories.update(cats)
    if not requested_categories:
        requested_categories = set(range(1, 11))

    candidates = [obj for obj in CULTURAL_OBJECTS if obj['category_id'] in requested_categories]
    if not candidates:
        candidates = CULTURAL_OBJECTS
    if not candidates:
        return "Не удалось найти объекты. Попробуйте другие интересы."

    total_available_min = int(time_hours * 60)
    current_lat, current_lon = start_lat, start_lon
    visited = []
    total_time = 0

    CATEGORY_TIME = {
        1: 12,  # памятники
        2: 25,  # парки
        4: 15,  # набережные
        5: 25,  # архитектура
        7: 45,  # музеи
        8: 60,  # театры
        10: 10, # мозаики
    }

    while candidates and total_time < total_available_min:
        nearest = None
        min_dist = float('inf')
        for obj in candidates:
            d = haversine(current_lat, current_lon, obj['lat'], obj['lon'])
            if d < min_dist:
                min_dist = d
                nearest = obj

        if nearest is None:
            break

        walk_time = max(1, int(min_dist / 80))
        visit_time = CATEGORY_TIME.get(nearest['category_id'], 20)
        new_total = total_time + walk_time + visit_time

        if new_total > total_available_min:
            break

        visited.append(nearest)
        total_time = new_total
        current_lat, current_lon = nearest['lat'], nearest['lon']
        candidates.remove(nearest)

    if not visited:
        return "Не удалось подобрать маршрут. Попробуйте увеличить время."

    # Заголовок с "водой", но без перегруза
    route_text = "Вот твой маршрут по Нижнему с учётом осмотра достопримечательностей!\n\n"
    points = [(start_lat, start_lon)]

    for idx, obj in enumerate(visited, 1):
        route_text += f"{idx}. {obj['title']}\n"
        desc = clean_description_brief(obj['description'])
        if desc:
            route_text += f"\n   {desc}\n"
        route_text += "\n"
        points.append((obj['lat'], obj['lon']))

    # Если вы хотите УБРАТЬ время полностью — закомментируйте следующие 2 строки:
    route_text += f"Итого: ~{total_time} мин.\n\n"
    route_text += f"Проложить маршрут: {build_yandex_route_url(points)}"
    return route_text

# === ХЭНДЛЕРЫ ===
@dp.message(Command('start'))
async def start_handler(message: types.Message, state: FSMContext):
    await message.reply("Привет! Я твой гид по Нижнему. Чем интересуешься? (парки, памятники, история, мозаики и т.д.)")
    await state.set_state(RouteForm.interests)


@dp.message(RouteForm.interests)
async def process_interests(message: types.Message, state: FSMContext):
    await state.update_data(interests=message.text)
    await message.reply("Сколько часов гуляем? (например: 1.5 или 2)")
    await state.set_state(RouteForm.time)

@dp.message(RouteForm.time)
async def process_time(message: types.Message, state: FSMContext):
    try:
        time_hours = float(message.text.replace(',', '.'))
        if not (0.5 <= time_hours <= 6):
            await message.reply("Укажи от 0.5 до 6 часов.")
            return
        await state.update_data(time_hours=time_hours)
        kb = ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="Отправить геопозицию", request_location=True)]],
            resize_keyboard=True, one_time_keyboard=True
        )
        await message.reply("Где ты? Отправь геопозицию или напиши адрес (например: пл Минина и Пожарского)", reply_markup=kb)
        await state.set_state(RouteForm.location)
    except ValueError:
        await message.reply("Напиши число, например: 2")

@dp.message(RouteForm.location)
async def process_location(message: types.Message, state: FSMContext):
    data = await state.get_data()
    interests = data['interests']
    time_hours = data['time_hours']

    lat, lon = None, None

    if message.location:
        lat, lon = message.location.latitude, message.location.longitude
    elif message.text:
        query = message.text.strip()
        query_lower = query.lower()

        for obj in CULTURAL_OBJECTS:
            if query_lower in obj['address'].lower() or query_lower in obj['title'].lower():
                lat, lon = obj['lat'], obj['lon']
                break

        if lat is None:
            full_query = f"{query}, Нижний Новгород"
            location = await safe_geocode(full_query)
            if location:
                lat, lon = location.latitude, location.longitude
            else:
                await message.reply("Не удалось найти этот адрес. Попробуй: пл Минина и Пожаского, Кремль или отправь геопозицию.")
                return
    else:
        await message.reply("Нужен адрес или геопозиция!")
        return

    await message.reply("Строю маршрут...")
    route = generate_route_from_data(interests, time_hours, lat, lon)
    await message.reply(route, disable_web_page_preview=True)
    await state.clear()

# === ЗАПУСК ===
async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())