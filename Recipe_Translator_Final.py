import os
import sqlite3
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Check readme file - HF-Token needs to be added
load_dotenv()
client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

# 1. Mathematical Confidence Score (No Penalties)
def calculate_confidence_score(original_functions, candidate_functions):
    if not original_functions or not candidate_functions:
        return 0.0

    intersection = original_functions.intersection(candidate_functions)
    if not intersection:
        return 0.0

    # Overlap calculation
    overlap = len(intersection) / min(len(original_functions), len(candidate_functions))
    return round(overlap, 2)


# 2. Gather database intel
def translate_ingredient_pipeline(original_name, target_cuisine):
    with sqlite3.connect('DataModelling_Recipe_DB.db') as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT i.ingredient_id, i.humanities_notes, group_concat(f.name)
            FROM Ingredients i
            LEFT JOIN IngredientFunctions inf ON i.ingredient_id = inf.ingredient_id
            LEFT JOIN Functions f ON inf.function_id = f.function_id
            WHERE LOWER(i.name) = LOWER(?) GROUP BY i.ingredient_id
        """, (original_name,))

        orig_row = cursor.fetchone()
        if not orig_row:
            return f"Error: '{original_name}' not found."

        original_id, original_notes, original_functions_str = orig_row
        original_functions = set(original_functions_str.split(',')) if original_functions_str else set()

        cursor.execute("""
            SELECT i.ingredient_id, i.name, i.humanities_notes, group_concat(f.name)
            FROM Ingredients i
            JOIN IngredientCuisines ic ON i.ingredient_id = ic.ingredient_id
            JOIN Cuisines c ON ic.cuisine_id = c.cuisine_id
            LEFT JOIN IngredientFunctions inf ON i.ingredient_id = inf.ingredient_id
            LEFT JOIN Functions f ON inf.function_id = f.function_id
            WHERE LOWER(c.name) = LOWER(?)
            GROUP BY i.ingredient_id
        """, (target_cuisine.lower(),))
        candidates = cursor.fetchall()

        # 3. Score candidates
        scored = []
        for c_id, c_name, c_notes, c_fns_str in candidates:
            c_fns = set(c_fns_str.split(',')) if c_fns_str else set()
            score = calculate_confidence_score(original_functions, c_fns)
            if score > 0.1:
                scored.append({"id": c_id, "name": c_name, "score": score, "notes": c_notes})

        scored = sorted(scored, key=lambda x: x['score'], reverse=True)
        if not scored:
            return f"No appropriate substitutes found in {target_cuisine} pantry."
        best_match = scored[0]

        # 4. RAG Prompt -> handing data to mighty AI brain
        prompt = f"""
        You are a Culinary Historian and Computational Gastronomist.
        Analyze this cross-cultural ingredient substitution:

        Source: {original_name} ({original_notes})
        Target Substitution: {best_match['name']} ({best_match['notes']})
        Confidence Score: {best_match['score'] * 100}%

        Tasks:
        1. Output a markdown statement showing the substitution and confidence metrics.
        2. Write a 2-3 sentence 'Cultural Logic' narrative explaining how this preserves structural parameters (texture, moisture, chemical function) while adapting to the historical availability of {target_cuisine} cuisine.
        """

        completion = client.chat_completion(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return completion.choices[0].message.content

# Print ingredient swap
if __name__ == "__main__":
    print(translate_ingredient_pipeline("Egg", "German"))

#5. Recipe Translation Implementation
def translate_recipe_pipeline(recipe_title, target_cuisine):
    with sqlite3.connect('DataModelling_Recipe_DB.db') as conn:
        cursor = conn.cursor()

        # 5.1. Fetch recipe instructions
        cursor.execute("SELECT recipe_id, instructions FROM Recipes WHERE LOWER(title) = LOWER(?)", (recipe_title,))
        recipe_row = cursor.fetchone()
        if not recipe_row: return f"Error: Recipe '{recipe_title}' not found."
        recipe_id, original_instructions = recipe_row

        # 5.2. Fetch recipe ingredients
        cursor.execute("""
            SELECT i.name, ri.quantity, ri.unit, i.ingredient_id, i.humanities_notes
            FROM RecipeIngredients ri
            JOIN Ingredients i ON ri.ingredient_id = i.ingredient_id
            WHERE ri.recipe_id = ?
        """, (recipe_id,))
        recipe_ingredients = cursor.fetchall()

    translated_ingredients_list = []
    substitution_log = []
    total_score = 0
    valid_sub_count = 0

    # 5.3. Dynamic Loop: Calculate real matches for each ingredient
    for orig_name, qty, unit, orig_id, orig_notes in recipe_ingredients:
        with sqlite3.connect('DataModelling_Recipe_DB.db') as conn:
            inner_cursor = conn.cursor()

            # Fetch functions of the original recipe ingredient
            inner_cursor.execute("""
                SELECT group_concat(f.name) FROM IngredientFunctions inf
                JOIN Functions f ON inf.function_id = f.function_id
                WHERE inf.ingredient_id = ?
            """, (orig_id,))
            orig_fns_row = inner_cursor.fetchone()
            orig_fns = set(orig_fns_row[0].split(',')) if orig_fns_row[0] else set()

            # Fetch all candidate target ingredients for the target cuisine
            inner_cursor.execute("""
                SELECT i.ingredient_id, i.name, i.humanities_notes, group_concat(f.name)
                FROM Ingredients i
                JOIN IngredientCuisines ic ON i.ingredient_id = ic.ingredient_id
                JOIN Cuisines c ON ic.cuisine_id = c.cuisine_id
                LEFT JOIN IngredientFunctions inf ON i.ingredient_id = inf.ingredient_id
                LEFT JOIN Functions f ON inf.function_id = f.function_id
                WHERE LOWER(c.name) = LOWER(?)
                GROUP BY i.ingredient_id
            """, (target_cuisine.lower(),))
            candidates = inner_cursor.fetchall()

        # Score all target candidates against the original recipe ingredient
        scored_candidates = []
        for c_id, c_name, c_notes, c_fns_str in candidates:
            c_fns = set(c_fns_str.split(',')) if (c_fns_str and isinstance(c_fns_str, str)) else set()
            score = calculate_confidence_score(original_functions=orig_fns, candidate_functions=c_fns)
            if score > 0.0:
                scored_candidates.append({"name": c_name, "score": score, "notes": c_notes})

        scored_candidates = sorted(scored_candidates, key=lambda x: x['score'], reverse=True)

        # Assign the mathematically optimal winner
        if scored_candidates:
            best_match = scored_candidates[0]
            translated_ingredients_list.append(f"* {qty} {unit} {best_match['name']} (substituted from {orig_name})")
            substitution_log.append(
                f"- Swapped {orig_name} -> {best_match['name']} (Confidence: {best_match['score'] * 100}%)")
            total_score += best_match['score']
            valid_sub_count += 1
        else:
            translated_ingredients_list.append(f"* {qty} {unit} {orig_name}")

    avg_confidence = (total_score / valid_sub_count) * 100 if valid_sub_count > 0 else 100

    # 6. RAG Prompt to rewrite the recipe realistically
    recipe_prompt = f"""
    You are an expert Culinary Historian and Computational Gastronomist.
    Translate this full recipe text into the culinary framework of {target_cuisine} cuisine.

    Original Recipe Title: {recipe_title}
    Original Instructions: {original_instructions}

    Target Ingredient Adaptations:
    {"\n".join(translated_ingredients_list)}

    Substitution Mapping:
    {"\n".join(substitution_log)}

    Overall Recipe Confidence Score: {round(avg_confidence, 1)}%

    Tasks:
    1. Output a Markdown "Dual-Recipe Interface" showing the final adapted ingredients compared to the original.
    2. Rewrite the cooking instructions step-by-step so they incorporate the new ingredients naturally (e.g., if a cured meat changes to a German equivalent, adapt the rendering/cooking instructions).
    3. Conclude with a 3-sentence unified 'Cultural Logic' reflection on this transformation.
    """

    completion = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": recipe_prompt}],
        max_tokens=800,
        temperature=0.1,
    )
    return completion.choices[0].message.content

# Execute Recipe Translation
if __name__ == "__main__":
    print("\n=== FINAL RECIPE ENGINE OUTPUT ===")
    print(translate_recipe_pipeline("Pasta alla Carbonara", "Japanese"))