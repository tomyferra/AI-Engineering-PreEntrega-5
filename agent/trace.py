"""Registro de la traza ReAct: convierte los updates del grafo en eventos legibles (log) y estructurados (JSON)."""
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

MAX_CHARS_LOG = 400


def _resumir(texto: str, limite: int = MAX_CHARS_LOG) -> str:
    texto = " ".join(texto.split())
    return texto if len(texto) <= limite else texto[:limite] + "…"


def _formatear_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


class TraceRecorder:
    """Acumula los turnos de una sesión. Cada turno: pregunta del usuario, pasos del ciclo ReAct y respuesta.

    Con json_path/log_path la traza se persiste: al crearse carga los turnos previos del archivo y al
    terminar cada turno lo reescribe, así la traza de un thread_id sigue creciendo entre ejecuciones
    (igual que su memoria en el checkpointer).
    """

    def __init__(self, echo: Optional[Callable[[str], None]] = print, json_path: Optional[Path] = None,
                 log_path: Optional[Path] = None, llm: Optional[str] = None):
        self._echo = echo
        self._json_path = json_path
        self._log_path = log_path
        self._llm = llm
        self.turnos: List[dict] = []
        self.lineas: List[str] = []
        if json_path and json_path.exists():
            self.turnos = json.loads(json_path.read_text(encoding="utf-8"))["turnos"]
        if log_path and log_path.exists():
            self.lineas = log_path.read_text(encoding="utf-8").rstrip("\n").split("\n")

    def _log(self, linea: str) -> None:
        self.lineas.append(linea)
        if self._echo:
            self._echo(linea)

    @property
    def _turno(self) -> dict:
        return self.turnos[-1]

    def start_turn(self, thread_id: str, mensaje: str) -> None:
        self.turnos.append({"turno": len(self.turnos) + 1, "thread_id": thread_id,
                            "inicio": datetime.now().isoformat(timespec="seconds"), "llm": self._llm,
                            "usuario": mensaje, "n_pasos": 0, "pasos": [], "respuesta": None,
                            "llamadas_a_tools": 0})
        self._log("")
        self._log(f"[turno {self._turno['turno']} | thread_id={thread_id} | {self._turno['inicio']}]")
        self._log(f'Usuario: "{mensaje}"')

    def on_update(self, nodo: str, mensajes: List[BaseMessage]) -> None:
        """Un update = un paso del grafo. Varias tool calls en paralelo comparten el mismo número de paso."""
        if not mensajes:
            return
        self._turno["n_pasos"] += 1
        paso = self._turno["n_pasos"]
        for mensaje in mensajes:
            if isinstance(mensaje, AIMessage) and mensaje.tool_calls:
                llamadas = [{"tool": tc["name"], "args": tc["args"], "id": tc["id"]} for tc in mensaje.tool_calls]
                self._turno["pasos"].append({"paso": paso, "nodo": nodo, "tipo": "accion",
                                             "razonamiento": mensaje.text or None, "tool_calls": llamadas})
                self._turno["llamadas_a_tools"] += len(llamadas)
                if mensaje.text:
                    self._log(f"  · [paso {paso}] El agente razona: {_resumir(mensaje.text)}")
                for llamada in llamadas:
                    self._log(f"  → [paso {paso}] El agente decide usar la herramienta: "
                              f"{llamada['tool']}({_formatear_args(llamada['args'])})")
            elif isinstance(mensaje, ToolMessage):
                contenido = mensaje.text
                self._turno["pasos"].append({"paso": paso, "nodo": nodo, "tipo": "observacion", "tool": mensaje.name,
                                             "status": mensaje.status, "contenido": _parse_json(contenido)})
                etiqueta = "ERROR" if mensaje.status == "error" or '"error"' in contenido else "devuelve"
                self._log(f"  ← [paso {paso}] La herramienta {mensaje.name} {etiqueta}: {_resumir(contenido)}")
            elif isinstance(mensaje, AIMessage):
                self._turno["pasos"].append({"paso": paso, "nodo": nodo, "tipo": "respuesta_final",
                                             "contenido": mensaje.text})
                self._log(f"  ✓ [paso {paso}] El agente razona: ya tiene los datos → responde.")

    def end_turn(self, respuesta: Optional[str], error: Optional[str] = None) -> None:
        self._turno["respuesta"] = respuesta
        self._turno["error"] = error
        if error:
            self._log(f"  ✗ Error: {error}")
        if respuesta:
            self._log(f'Respuesta: "{respuesta}"')
        self._save()

    def _save(self) -> None:
        if self._json_path:
            self._json_path.parent.mkdir(parents=True, exist_ok=True)
            documento = {"total_llamadas_a_tools": sum(t["llamadas_a_tools"] for t in self.turnos),
                         "turnos": self.turnos}
            self._json_path.write_text(json.dumps(documento, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text("\n".join(self.lineas).lstrip() + "\n", encoding="utf-8")


def _parse_json(contenido: str):
    try:
        return json.loads(contenido)
    except (json.JSONDecodeError, TypeError):
        return contenido
