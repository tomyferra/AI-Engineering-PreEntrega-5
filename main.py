"""Chat con el agente de GenPet.

Uso:
    python main.py                                   # chat interactivo en el thread "default"
    python main.py --thread ventas-octubre           # retoma (o crea) una sesión persistente
    python main.py --thread t1 "¿Cuántos pedidos tuvo Patitas Felices?"   # una sola pregunta
    python main.py --sin-rag ...                     # sin Pinecone: solo tools de pedidos

La traza ReAct de cada thread se guarda en traces/<thread_id>.json y traces/<thread_id>.log.
"""
import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from agent.session import create_agent, run_turn
from agent.trace import TraceRecorder
from config import TRACES_DIR, get_llm_settings

SALIR = {"salir", "exit", "quit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente ReAct de GenPet con memoria persistente")
    parser.add_argument("pregunta", nargs="?", help="Pregunta única; si se omite, abre un chat interactivo")
    parser.add_argument("--thread", default="default", help="thread_id de la sesión (default: 'default')")
    parser.add_argument("--sin-rag", action="store_true", help="No cargar la tool de conocimiento (Pinecone)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    llm = get_llm_settings()
    # La traza se imprime en vivo (cada decisión del agente y cada observación de las tools) y se guarda en
    # traces/<thread_id>.json y .log, acumulando los turnos del thread entre ejecuciones.
    recorder = TraceRecorder(echo=print, json_path=TRACES_DIR / f"{args.thread}.json",
                             log_path=TRACES_DIR / f"{args.thread}.log",
                             llm=f"{llm.provider.value}:{llm.model}")

    async with create_agent(use_rag=not args.sin_rag) as graph:
        if args.pregunta:
            await run_turn(graph, args.thread, args.pregunta, recorder)
            return

        print(f"Agente GenPet · thread_id={args.thread} · escribí 'salir' para terminar")
        while True:
            try:
                mensaje = (await asyncio.to_thread(input, "\nVos: ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if mensaje.lower() in SALIR:
                break
            if mensaje:
                await run_turn(graph, args.thread, mensaje, recorder)


if __name__ == "__main__":
    asyncio.run(main())
