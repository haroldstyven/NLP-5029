import itertools
import logging
import os
import time
import re
from typing import List, Tuple, Optional, Any
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv(find_dotenv())
API_KEY = os.getenv("API_KEY")

# Configurar Gemini
if not API_KEY:
    logger.error("No se encontró la API_KEY en el archivo .env")
else:
    genai.configure(api_key=API_KEY)

MODEL_NAME = 'gemini-2.5-flash'
model = genai.GenerativeModel(MODEL_NAME)

class APITracker:
    """Clase para contar las llamadas a la API y manejar Rate Limits básicos."""
    def __init__(self):
        self.count = 0
        self.last_call_time = 0

    def call_gemini(self, prompt: str) -> str:
        # Rate limit simple (ej. 1 segundo entre llamadas) para evitar 429 en cuentas free
        elapsed = time.time() - self.last_call_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        
        self.count += 1
        try:
            response = model.generate_content(prompt)
            self.last_call_time = time.time()
            return response.text
        except Exception as e:
            logger.error(f"Error en llamada a Gemini: {e}")
            if "429" in str(e):
                logger.info("Rate limit alcanzado. Esperando 10 segundos...")
                time.sleep(10)
                return self.call_gemini(prompt)
            return ""

api_tracker = APITracker()

class GameState:
    """Estado del Juego del 24."""
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
    """Generador de combinaciones matemáticas."""
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

class GeminiEvaluator:
    """Evaluador de estados usando Gemini."""
    @staticmethod
    def evaluate(state: GameState) -> float:
        if state.is_terminal():
            return 1.0 if abs(state.get_result() - 24) < 1e-6 else 0.0
        
        prompt = (
            f"En el Juego del 24, dados estos números: {state.numbers}, ¿es posible llegar a 24?\n"
            "Responde únicamente con una palabra: 'sure' (si estás seguro), 'maybe' (si crees que es posible) o 'impossible' (si es imposible)."
        )
        
        response_text = api_tracker.call_gemini(prompt).lower()
        if 'sure' in response_text: return 1.0
        if 'maybe' in response_text: return 0.5
        return 0.0

def tot_bfs_search(initial_numbers: List[float], beam_size: int = 3):
    """Tree of Thoughts search with Beam Pruning."""
    start_time = time.time()
    initial_api_count = api_tracker.count
    current_states = [GameState(initial_numbers)]
    
    for step in range(1, 4):
        logger.info(f"ToT Paso {step}: Expandiendo {len(current_states)} estados...")
        all_next = []
        for s in current_states:
            all_next.extend(ThoughtGenerator.next_steps(s))
        
        unique_next = list(set(all_next))
        scored = []
        for s in unique_next:
            score = GeminiEvaluator.evaluate(s)
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
    """Chain of Thought resolution."""
    start_time = time.time()
    initial_api_count = api_tracker.count
    
    prompt = (
        f"Input: {numbers}. Tarea: Usa cada número exactamente una vez con (+, -, *, /) para llegar a 24. "
        "Muestra tu razonamiento paso a paso y termina con la ecuación final exacta."
    )
    
    response_text = api_tracker.call_gemini(prompt)
    success = "24" in response_text and ("=" in response_text or "resultado" in response_text.lower())
    
    return {
        "success": success,
        "response": response_text,
        "time": time.time() - start_time,
        "api_calls": api_tracker.count - initial_api_count
    }

def print_result_table(res_tot, res_cot):
    print("\n" + "="*85)
    print(f"{'MÉTODO':<20} | {'ÉXITO':<10} | {'TIEMPO (s)':<15} | {'API CALLS':<15} | {'ECUACIÓN'}")
    print("-" * 85)
    
    tot_success = "SÍ" if res_tot['success'] else "NO"
    tot_eq = " -> ".join(res_tot['result'].history) if res_tot['success'] else "N/A"
    
    cot_success = "SÍ" if res_cot['success'] else "NO"
    # Extraer algo parecido a una ecuación del CoT (heurística simple)
    cot_eq = "Refer to text" if res_cot['success'] else "N/A"

    print(f"{'Tree of Thoughts':<20} | {tot_success:<10} | {res_tot['time']:<15.2f} | {res_tot['api_calls']:<15} | {tot_eq}")
    print(f"{'Chain of Thought':<20} | {cot_success:<10} | {res_cot['time']:<15.2f} | {res_cot['api_calls']:<15} | {cot_eq}")
    print("="*85 + "\n")

if __name__ == "__main__":
    nums = [4, 9, 10, 13]
    print(f"Iniciando Comparativa Gemini: ToT vs CoT para {nums}\n")
    
    # Ejecutar CoT
    logger.info("Iniciando Chain of Thought (CoT)...")
    cot_metrics = solve_cot(nums)
    
    # Ejecutar ToT
    logger.info("Iniciando Tree of Thoughts (ToT)...")
    tot_metrics = tot_bfs_search(nums, beam_size=3)
    
    print_result_table(tot_metrics, cot_metrics)
    
    if cot_metrics['success']:
        print("Razonamiento CoT detectado:")
        lines = cot_metrics['response'].split('\n')
        for line in lines[:8]: # Primeras 8 líneas
            print(f"  {line}")
        print("  ...")
