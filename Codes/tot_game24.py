import itertools
import logging
import os
from typing import List, Tuple, Optional, Dict
import math

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class GameState:
    """
    Representa un estado en el Juego del 24.
    Contiene los números actuales y el historial de operaciones realizadas.
    """
    def __init__(self, numbers: List[float], history: List[str] = None):
        self.numbers = sorted(numbers)
        self.history = history or []

    def is_terminal(self) -> bool:
        """Un estado es terminal si solo queda un número."""
        return len(self.numbers) == 1

    def get_result(self) -> float:
        """Retorna el único número restante en un estado terminal."""
        return self.numbers[0] if self.is_terminal() else None

    def __repr__(self):
        return f"State(nums={self.numbers}, path='{' -> '.join(self.history)}')"

    def __eq__(self, other):
        return isinstance(other, GameState) and self.numbers == other.numbers

    def __hash__(self):
        return hash(tuple(self.numbers))

class ThoughtGenerator:
    """
    Generador de pensamientos: Produce todos los estados posibles a partir del actual
    combinando dos números mediante operaciones aritméticas.
    """
    @staticmethod
    def next_steps(state: GameState) -> List[GameState]:
        next_states = []
        nums = state.numbers
        n = len(nums)
        
        # Seleccionamos todas las combinaciones posibles de 2 índices
        for i, j in itertools.combinations(range(n), 2):
            a, b = nums[i], nums[j]
            remaining_nums = [nums[k] for k in range(n) if k != i and k != j]
            
            # Operaciones posibles
            # Usamos un conjunto para evitar resultados duplicados en el mismo paso
            potential_results = []
            
            # a + b
            potential_results.append((a + b, f"{a} + {b} = {a+b}"))
            # a * b
            potential_results.append((a * b, f"{a} * {b} = {a*b}"))
            # a - b y b - a
            potential_results.append((a - b, f"{a} - {b} = {a-b}"))
            potential_results.append((b - a, f"{b} - {a} = {b-a}"))
            # a / b (si b != 0)
            if abs(b) > 1e-9:
                potential_results.append((a / b, f"{a} / {b} = {a/b:.2f}"))
            # b / a (si a != 0)
            if abs(a) > 1e-9:
                potential_results.append((b / a, f"{b} / {a} = {b/a:.2f}"))

            for res, op_str in potential_results:
                new_nums = remaining_nums + [res]
                next_states.append(GameState(new_nums, state.history + [op_str]))
        
        return next_states

class StateEvaluator:
    """
    Evaluador de estados: Clasifica la probabilidad de llegar a 24 desde un estado dado.
    """
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        if use_llm:
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except (ImportError, Exception) as e:
                logger.warning(f"No se pudo inicializar OpenAI: {e}. Usando Mock Evaluator.")
                self.use_llm = False

    def evaluate(self, state: GameState) -> float:
        """
        Evalúa el estado y devuelve una puntuación heurística.
        sure = 1.0, maybe = 0.5, impossible = 0.0
        """
        if state.is_terminal():
            return 1.0 if abs(state.get_result() - 24) < 1e-6 else 0.0
        
        if self.use_llm:
            return self._llm_evaluate(state)
        else:
            return self._mock_evaluate(state)

    def _llm_evaluate(self, state: GameState) -> float:
        """
        Simulación de llamada a LLM siguiendo el paper ToT.
        """
        prompt = f"Dado los números {state.numbers}, ¿es posible llegar a 24 usando +, -, *, /? Responde solo con 'sure', 'maybe' o 'impossible'."
        try:
            # Aquí iría la llamada real
            # response = self.client.chat.completions.create(...)
            # Por ahora, delegamos al mock para este ejercicio, pero dejamos la estructura
            return self._mock_evaluate(state)
        except Exception:
            return 0.5 

    def _mock_evaluate(self, state: GameState) -> float:
        """
        Heurística básica para simular la evaluación del LLM:
        - 'sure' si 24 es alcanzable directamente en el siguiente paso.
        - 'maybe' si hay números que podrían combinar hacia 24.
        - 'impossible' si los números son muy grandes (>1000) o irrelevantes.
        """
        nums = state.numbers
        
        # Caso 'sure' simplificado: ¿pueden 2 números dar 24?
        if len(nums) == 2:
            a, b = nums[0], nums[1]
            if any(abs(res - 24) < 1e-6 for res in [a+b, a*b, a-b, b-a, a/b if b!=0 else 0, b/a if a!=0 else 0]):
                return 1.0
        
        # Caso 'impossible': Evidentemente fuera de rango (heurística simple)
        if all(n > 24 for n in nums) and min(nums) > 24 and sum(nums) > 100:
             # Nota: Esto es arriesgado en el 24 (ej. 25-1), pero ilustra la poda
             pass 

        return 0.5 # 'maybe' por defecto para permitir exploración

def tot_bfs_search(initial_nums: List[float], beam_size: int = 3):
    """
    Algoritmo Tree of Thoughts con Búsqueda en Anchura (BFS) y Beam Search.
    """
    logger.info(f"Iniciando ToT para {initial_nums} con beam_size={beam_size}")
    
    current_states = [GameState(initial_nums)]
    
    # El juego del 24 se resuelve en 3 pasos (4->3, 3->2, 2->1)
    for step in range(1, 4):
        logger.info(f"--- PASO {step} ---")
        all_next_states = []
        
        for state in current_states:
            generated = ThoughtGenerator.next_steps(state)
            all_next_states.extend(generated)
        
        # Eliminar duplicados para optimizar
        unique_next_states = list(set(all_next_states))
        logger.info(f"Generados {len(all_next_states)} pensamientos ({len(unique_next_states)} únicos)")
        
        # Evaluar cada estado
        evaluator = StateEvaluator(use_llm=False)
        scored_states = []
        for s in unique_next_states:
            score = evaluator.evaluate(s)
            scored_states.append((score, s))
        
        # Ordenar por score y seleccionar los mejores (Beam Search)
        # En ToT real, se suelen filtrar los 'impossible' primero
        scored_states.sort(key=lambda x: x[0], reverse=True)
        
        # Pruning: Solo nos quedamos con los b mejores
        pruned_states = scored_states[:beam_size]
        current_states = [s for score, s in pruned_states if score > 0]
        
        logger.info(f"Podando: de {len(scored_states)} a {len(current_states)} estados seleccionados.")
        for score, s in pruned_states:
            if score > 0:
                logger.info(f"  [Seleccionado] Score: {score} | Nums: {s.numbers}")

        if not current_states:
            logger.error("No se encontraron más ramas viables. Fallo en la búsqueda.")
            return None

    # Verificar si alguno de los estados finales es exitoso
    for state in current_states:
        if state.is_terminal() and abs(state.get_result() - 24) < 1e-6:
            logger.info("¡SOLUCIÓN ENCONTRADA!")
            return state
            
    logger.warning("No se encontró una solución exacta a 24 en los mejores pensamientos.")
    return None

if __name__ == "__main__":
    # Prueba del script
    input_numbers = [4, 9, 10, 13]
    b_limit = 3
    
    solution = tot_bfs_search(input_numbers, beam_size=b_limit)
    
    print("\n" + "="*50)
    print("RESULTADO DEL TREE OF THOUGHTS")
    print("="*50)
    if solution:
        print(f"Números Iniciales: {input_numbers}")
        print(f"Ecuación Final Ganadora:")
        for i, step in enumerate(solution.history, 1):
            print(f"  Paso {i}: {step}")
        print(f"Resultado: {solution.get_result()}")
    else:
        print("No se encontró una solución con el beam_size actual.")
    print("="*50)
