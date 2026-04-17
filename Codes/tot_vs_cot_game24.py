import itertools
import logging
import os
import time
import re
from typing import List, Tuple, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno buscando el archivo .env
load_dotenv(find_dotenv())
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Cliente de OpenAI configurado para OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL_NAME = "openai/gpt-5.4-mini"

class APITracker:
    """Clase para contar las llamadas a la API."""
    def __init__(self):
        self.count = 0

    def call(self, *args, **kwargs):
        self.count += 1
        return client.chat.completions.create(*args, **kwargs)

api_tracker = APITracker()

class GameState:
    """Representa un estado en el Juego del 24."""
    def __init__(self, numbers: List[float], history: List[str] = None):
        self.numbers = sorted(numbers)
        self.history = history or []

    def is_terminal(self) -> bool:
        return len(self.numbers) == 1

    def get_result(self) -> float:
        return self.numbers[0] if self.is_terminal() else None

    def __repr__(self):
        return f"State({self.numbers})"

    def __eq__(self, other):
        return isinstance(other, GameState) and self.numbers == other.numbers

    def __hash__(self):
        return hash(tuple(self.numbers))

class ThoughtGenerator:
    """Generador de pensamientos usando combinatoria pura en Python."""
    @staticmethod
    def next_steps(state: GameState) -> List[GameState]:
        next_states = []
        nums = state.numbers
        n = len(nums)
        for i, j in itertools.combinations(range(n), 2):
            a, b = nums[i], nums[j]
            remaining = [nums[k] for k in range(n) if k != i and k != j]
            
            ops = [
                (a + b, f"{a} + {b} = {a+b}"),
                (a * b, f"{a} * {b} = {a*b}"),
                (a - b, f"{a} - {b} = {a-b}"),
                (b - a, f"{b} - {a} = {b-a}")
            ]
            if abs(b) > 1e-9: ops.append((a / b, f"{a} / {b} = {a/b:.2f}"))
            if abs(a) > 1e-9: ops.append((b / a, f"{b} / {a} = {b/a:.2f}"))

            for res, op_str in ops:
                next_states.append(GameState(remaining + [res], state.history + [op_str]))
        return next_states

class Evaluator:
    """Evaluador de estados usando LLM con razonamiento activado."""
    @staticmethod
    def evaluate(state: GameState) -> float:
        if state.is_terminal():
            return 1.0 if abs(state.get_result() - 24) < 1e-6 else 0.0
        
        prompt = (
            f"Given the numbers {state.numbers}, can you reach 24 using basic operations (+, -, *, /)? "
            "Evaluate as 'sure' (if highly likely or immediate), 'maybe' (if possible), or 'impossible' (if definitely not). "
            "Reply ONLY with one of these three words."
        )
        
        try:
            response = api_tracker.call(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                extra_body={"reasoning": {"enabled": True}}
            )
            msg = response.choices[0].message
            ans = (msg.content or "").strip().lower()
            
            # Si el contenido está vacío, buscamos en detalles de razonamiento si están disponibles
            if not ans and hasattr(msg, 'reasoning'):
                ans = (msg.reasoning or "").strip().lower()

            if 'sure' in ans: return 1.0
            if 'maybe' in ans: return 0.5
            return 0.0
        except Exception as e:
            logger.error(f"Error parseando respuesta de {MODEL_NAME}: {e}")
            return 0.5

def tot_search(initial_numbers: List[float], beam_size: int = 3):
    """Implementación ToT mediante BFS."""
    start_time = time.time()
    initial_api_count = api_tracker.count
    current_states = [GameState(initial_numbers)]
    
    for step in range(1, 4):
        logger.info(f"ToT Step {step}: Expanding {len(current_states)} nodes...")
        all_next = []
        for s in current_states:
            all_next.extend(ThoughtGenerator.next_steps(s))
        
        unique_next = list(set(all_next))
        scored = []
        for s in unique_next:
            score = Evaluator.evaluate(s)
            scored.append((score, s))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        current_states = [s for score, s in scored[:beam_size] if score > 0]
        if not current_states: break

    success = False
    best_state = None
    for s in current_states:
        if s.is_terminal() and abs(s.get_result() - 24) < 1e-6:
            success = True
            best_state = s
            break
            
    return {
        "success": success,
        "result": best_state,
        "time": time.time() - start_time,
        "api_calls": api_tracker.count - initial_api_count
    }

def solve_cot(numbers: List[float]):
    """Implementación Chain of Thought (CoT)."""
    start_time = time.time()
    initial_api_count = api_tracker.count
    
    prompt = (
        f"Input numbers: {numbers}. Goal: Use each number exactly once with basic arithmetic (+, -, *, /) to get exactly 24. "
        "Show your step-by-step reasoning and provide the final equation."
    )
    
    try:
        response = api_tracker.call(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            extra_body={"reasoning": {"enabled": True}}
        )
        content = response.choices[0].message.content
        # Intentamos extraer un resultado exitoso si menciona que llegó a 24
        success = "24" in content and ("=" in content or "result" in content.lower())
        
        return {
            "success": success,
            "raw_response": content,
            "time": time.time() - start_time,
            "api_calls": api_tracker.count - initial_api_count
        }
    except Exception as e:
        logger.error(f"Error en CoT: {e}")
        return {"success": False, "time": time.time() - start_time, "api_calls": 0}

def verify_math(numbers: List[float], expression: str) -> bool:
    """
    Verificación simple (opcional para CoT avanzado). 
    En este script se basa primordialmente en la detección en el texto.
    """
    return "24" in expression

def print_comparison_table(results_tot, results_cot):
    """Imprime una tabla comparativa profesional."""
    print("\n" + "="*80)
    print(f"{'MÉTODO':<20} | {'ÉXITO':<10} | {'TIEMPO (s)':<15} | {'API CALLS':<15}")
    print("-" * 80)
    
    tot_success = "SÍ" if results_tot['success'] else "NO"
    cot_success = "SÍ" if results_cot['success'] else "NO"
    
    print(f"{'Tree of Thoughts':<20} | {tot_success:<10} | {results_tot['time']:<15.2f} | {results_tot['api_calls']:<15}")
    print(f"{'Chain of Thought':<20} | {cot_success:<10} | {results_cot['time']:<15.2f} | {results_cot['api_calls']:<15}")
    print("="*80 + "\n")

if __name__ == "__main__":
    nums = [4, 9, 10, 13]
    print(f"Iniciando Benchmarking para el Juego del 24 con {nums}...\n")
    
    # Ejecutar CoT
    logger.info("Ejecutando Chain of Thought (CoT)...")
    cot_res = solve_cot(nums)
    
    # Ejecutar ToT
    logger.info("Ejecutando Tree of Thoughts (ToT)...")
    tot_res = tot_search(nums, beam_size=3)
    
    print_comparison_table(tot_res, cot_res)
    
    if tot_res['success']:
        print(f"Solución ToT encontrada: {' -> '.join(tot_res['result'].history)}")
    
    if cot_res['success']:
        # Mostramos un fragmento de la respuesta CoT
        print(f"Respuesta CoT (fragmento): {cot_res['raw_response'][:200]}...")
