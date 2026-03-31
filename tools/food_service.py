"""Food nutrition service - PostgreSQL primary, LLM fallback.

Handles:
- Parsing food name, quantity, unit from natural language
- PostgreSQL lookup via food_repository
- LLM fallback when DB has no match
- Macros estimation via LLM when DB only has calories
"""

import re
from typing import Optional

from tools.db import ensure_tables
from tools.food_repository import find_food_by_alias, find_foods_by_name


# Unit → grams conversion (common Chinese dietary units)
UNIT_TO_GRAMS = {
    "克": 1,
    "g": 1,
    "G": 1,
    "千克": 1000,
    "公斤": 1000,
    "kg": 1000,
    "KG": 1000,
    "斤": 500,
    "两": 50,
    "毫升": 1,  # approximated as grams for liquid
    "ml": 1,
    "ML": 1,
    "升": 1000,
    "l": 1000,
    "L": 1000,
}

# Chinese numeral to Arabic
CHINESE_NUMERALS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _parse_chinese_number(text: str) -> int:
    """Convert Chinese numerals to integer."""
    total = 0
    for char in text:
        if char in CHINESE_NUMERALS:
            total = total * 10 + CHINESE_NUMERALS[char] if total < 10 else total + CHINESE_NUMERALS[char]
        elif char.isdigit():
            total = total * 10 + int(char)
    return total if total > 0 else 1


def _parse_quantity(text: str) -> tuple[Optional[float], Optional[str], str]:
    """Parse quantity, unit, and food name from input.

    Examples:
        "两个鸡蛋"        -> (2.0, "个", "鸡蛋")
        "100g鸡胸肉"     -> (100.0, "g", "鸡胸肉")
        "一碗米饭"       -> (1.0, "碗", "米饭")
        "半斤牛肉"       -> (250.0, "g", "牛肉")
        "鸡胸肉"         -> (100.0, "g", "鸡胸肉")  # default 100g

    Returns:
        (quantity, unit, food_name_cleaned)
        quantity may be None (will default to 100g)
    """
    text = text.strip()

    # Ordered unit alternation to prevent partial matching issues:
    # e.g. "千克" must be matched before individual "千", "g", "k"
    _UNITS = r'(?:千克|公斤|斤|两|毫升|升|克|ml|g|L|l)'
    # Number/chinese-numeral pattern
    _NUMS = r'[0-9零一二三四五六七八九十两半]+'

    # Pattern: number + unit + food name  (e.g. "100g鸡胸肉", "半斤牛肉")
    m = re.match(rf'^({_NUMS})\s*({_UNITS})\s*(.+)$', text)
    if m:
        qty_str = m.group(1)
        unit = m.group(2).lower()
        food = m.group(3).strip()
        qty = float(_parse_chinese_number(qty_str)) if qty_str not in ("半", "半份") else 0.5
        grams = qty * UNIT_TO_GRAMS.get(unit, 100)
        return grams, "g", food

    # Pattern: food name + number + unit  (e.g. "鸡胸肉 200g")
    m2 = re.match(rf'^(.+?)\s+([0-9]+)\s*({_UNITS})\s*$', text)
    if m2:
        food = m2.group(1).strip()
        unit = m2.group(3).lower()
        qty = float(m2.group(2))
        grams = qty * UNIT_TO_GRAMS.get(unit, 100)
        return grams, "g", food

    # Pattern: "X个Y" or "X碗Y" (quantity + informal unit + food)
    pattern3 = r'^([零一二三四五六七八九十两半]+)\s*([个碗杯盘份只根块只条把]+)\s*(.+)$'
    m3 = re.match(pattern3, text)
    if m3:
        qty_str = m3.group(1)
        food = m3.group(3).strip()
        qty = float(_parse_chinese_number(qty_str)) if qty_str not in ("半",) else 0.5
        # Informal units default to ~100g
        return qty * 100.0, "g", food

    # No quantity/unit found, treat entire input as food name with default 100g
    return None, None, text


def _scale_nutrition(food: dict, grams: float) -> dict:
    """Scale nutrition values by actual grams consumed.

    Args:
        food: Dict with *_per_100g fields.
        grams: Grams actually consumed.

    Returns:
        Nutrition dict with scaled calories, protein, fat, carbs.
    """
    factor = grams / 100.0
    return {
        "name": food.get("name", "未知食物"),
        "calories": round((food.get("calories_per_100g") or 0) * factor, 1),
        "protein": round((food.get("protein_per_100g") or 0) * factor, 1),
        "fat": round((food.get("fat_per_100g") or 0) * factor, 1),
        "carbs": round((food.get("carbs_per_100g") or 0) * factor, 1),
    }


class FoodNutritionService:
    """Food nutrition lookup with DB primary + LLM fallback."""

    def __init__(self):
        # Lazy import to avoid circular dependency
        from agent.llm import get_llm
        self.llm = get_llm()
        self._tables_ensured = False

    def _ensure_tables_once(self):
        """Ensure tables exist, but only once and non-fatally."""
        if self._tables_ensured:
            return
        try:
            ensure_tables()
            self._tables_ensured = True
        except Exception as e:
            print(f"[FoodService] ensure_tables skipped: {e}")
            self._tables_ensured = True  # don't retry every call

    def lookup(self, user_input: str) -> tuple[dict, bool]:
        """Look up food nutrition from user input.

        Steps:
        1. Parse quantity, unit, food name
        2. Query PostgreSQL by food name
        3. If found: scale nutrition and return (has_macros from DB)
        4. If not found: fall back to LLM

        Args:
            user_input: Natural language food description,
                        e.g., "两个鸡蛋", "100g鸡胸肉", "一碗米饭"

        Returns:
            (nutrition_dict, had_macros_in_db)
            nutrition_dict keys: name, calories, protein, fat, carbs
            had_macros_in_db: True if protein/fat/carbs came from DB (not LLM fill)
        """
        grams, unit, food_name = _parse_quantity(user_input)
        if grams is None:
            grams = 100.0  # default to 100g

        food_name_clean = re.sub(r'[的与和配一起等]+$', '', food_name).strip()
        if not food_name_clean:
            food_name_clean = food_name

        print(f"[FoodService] lookup: input={user_input!r}, parsed_grams={grams}, food={food_name_clean!r}")

        # Try DB lookup (non-fatal: any exception means skip DB and go to LLM)
        try:
            self._ensure_tables_once()
            food = find_food_by_alias(food_name_clean)
            if food:
                print(f"[FoodService] DB hit: {food['name']}")
                nutrition = _scale_nutrition(food, grams)
                had_macros = food.get("protein_per_100g") is not None
                return nutrition, had_macros

            # Try broader search
            matches = find_foods_by_name(food_name_clean, limit=3)
            if matches:
                food = matches[0]
                print(f"[FoodService] DB partial hit: {food['name']}")
                nutrition = _scale_nutrition(food, grams)
                had_macros = food.get("protein_per_100g") is not None
                return nutrition, had_macros
        except Exception as e:
            print(f"[FoodService] DB lookup skipped: {e}")

        # DB miss → LLM fallback
        print(f"[FoodService] DB miss, falling back to LLM for: {food_name_clean!r}")
        return self._llm_fallback(food_name_clean, grams)

    def _llm_fallback(self, food_name: str, grams: float) -> tuple[dict, bool]:
        """Use LLM to estimate nutrition when DB has no match.

        Args:
            food_name: Parsed food name.
            grams: Grams consumed.

        Returns:
            (nutrition_dict, had_macros_in_db=False)
        """
        factor = grams / 100.0
        prompt = f"""你是一个食物营养分析专家。请估算以下食物每100克的营养成分。

食物名称：{food_name}

请以JSON格式返回（仅返回JSON，不要其他文字）：
{{"name": "食物名称", "calories": 热量数值(kcal), "protein": 蛋白质克数, "fat": 脂肪克数, "carbs": 碳水化合物克数}}

注意：
1. 只估算真实存在的食物，不要编造罕见食物
2. 如果不确定某项营养素，填写null而不是猜测
3. 返回的食物名称要准确"""

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            import json as _json, re as _re
            match = _re.search(r'\{[^}]+\}', response.content)
            if match:
                data = _json.loads(match.group())
                name = data.get("name", food_name)
                return {
                    "name": name,
                    "calories": round((float(data.get("calories") or 0)) * factor, 1),
                    "protein": round((float(data.get("protein") or 0)) * factor, 1),
                    "fat": round((float(data.get("fat") or 0)) * factor, 1),
                    "carbs": round((float(data.get("carbs") or 0)) * factor, 1),
                }, False
        except Exception as e:
            print(f"[FoodService] LLM fallback error: {e}")

        # Last resort: return unknown
        return {
            "name": food_name,
            "calories": 0,
            "protein": 0,
            "fat": 0,
            "carbs": 0,
        }, False


# Singleton
_service = None


def get_food_service() -> FoodNutritionService:
    global _service
    if _service is None:
        _service = FoodNutritionService()
    return _service