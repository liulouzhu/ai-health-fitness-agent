"""Food nutrition repository - PostgreSQL backed.

Provides simple fuzzy matching via alias lookup.
"""

import re
from typing import Optional
from tools.db import get_cursor


def _normalize(text: str) -> str:
    """Lightweight normalization for matching: lowercase, strip spaces."""
    return text.lower().strip()


def find_food_by_alias(alias: str) -> Optional[dict]:
    """Find a food by alias name using simple exact + prefix matching.

    Uses PostgreSQL's simple text search for prefix matching (alias || ':*').

    Args:
        alias: The food name or alias to search for.

    Returns:
        Food dict with id, name, category, calories_per_100g, protein_per_100g,
        fat_per_100g, carbs_per_100g, or None if not found.
    """
    norm = _normalize(alias)

    with get_cursor() as cur:
        # Try exact match on alias first
        cur.execute(
            """
            SELECT f.id, f.name, f.category, f.description, f.source,
                   f.calories_per_100g, f.protein_per_100g, f.fat_per_100g,
                   f.carbs_per_100g
            FROM foods f
            JOIN food_aliases fa ON fa.food_id = f.id
            WHERE fa.alias = %s
            LIMIT 1
            """,
            (norm,)
        )
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "category": row[2],
                "description": row[3],
                "source": row[4],
                "calories_per_100g": row[5],
                "protein_per_100g": row[6],
                "fat_per_100g": row[7],
                "carbs_per_100g": row[8],
            }

    return None


def find_foods_by_name(name: str, limit: int = 5) -> list[dict]:
    """Search foods by name with simple LIKE matching.

    Args:
        name: Food name to search for.
        limit: Maximum results to return.

    Returns:
        List of matching food dicts.
    """
    norm = _normalize(name)
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT f.id, f.name, f.category, f.description, f.source,
                   f.calories_per_100g, f.protein_per_100g, f.fat_per_100g,
                   f.carbs_per_100g
            FROM foods f
            LEFT JOIN food_aliases fa ON fa.food_id = f.id
            WHERE LOWER(f.name) LIKE %s
               OR LOWER(fa.alias) LIKE %s
            ORDER BY length(f.name) ASC
            LIMIT %s
            """,
            ("%" + norm + "%", "%" + norm + "%", limit)
        )
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "category": row[2],
                "description": row[3],
                "source": row[4],
                "calories_per_100g": row[5],
                "protein_per_100g": row[6],
                "fat_per_100g": row[7],
                "carbs_per_100g": row[8],
            }
            for row in rows
        ]


def get_portions(food_id: int) -> list[dict]:
    """Get all portions for a food.

    Args:
        food_id: Food ID.

    Returns:
        List of portion dicts with portion_name, portion_grams, calories_per_portion.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, portion_name, portion_grams, calories_per_portion
            FROM food_portions
            WHERE food_id = %s
            ORDER BY portion_grams ASC
            """,
            (food_id,)
        )
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "portion_name": row[1],
                "portion_grams": row[2],
                "calories_per_portion": row[3],
            }
            for row in rows
        ]


def food_exists(name: str) -> bool:
    """Check if a food already exists in the database."""
    with get_cursor() as cur:
        cur.execute("SELECT 1 FROM foods WHERE name = %s LIMIT 1", (_normalize(name),))
        return cur.fetchone() is not None