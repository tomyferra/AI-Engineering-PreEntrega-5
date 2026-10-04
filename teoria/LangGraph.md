# Unidad 1 · Arquitectura de grafos: nodos, aristas y estado en LangGraph

Bienvenido al Módulo 5. Hasta ahora, has construido cadenas lineales con LCEL y has implementado arquitecturas RAG para dotar a tus modelos de una base de conocimientos. Sin embargo, en el mundo real, los problemas rara vez son lineales. Los sistemas de IA más avanzados no solo "responden", sino que razonan, iteran y corrigen sus propios errores.

Imagina un sistema de soporte técnico. No es una simple secuencia de pasos; es un flujo donde, dependiendo de la complejidad, el caso puede resolverse, pedir más información al usuario o escalarse a un humano. Intentar programar esto con cadenas `if/else` interminables o loops de chat básicos resulta en un código frágil, difícil de depurar y casi imposible de escalar.

Aquí es donde entra **LangGraph**. En esta unidad, aprenderás a pensar en términos de **Grafos de Estado**, una evolución necesaria para construir agentes autónomos con principios de ingeniería de software.

> 🎯 **A dónde vas con este módulo** (leé esto primero)

---

## 1. ¿Por qué LangGraph? El paso de Cadenas a Grafos

En los módulos anteriores, utilizamos LangChain para crear secuencias de pasos (cadenas). Estas cadenas son excelentes para flujos de trabajo predecibles:

```text
Input -> Prompt -> LLM -> Output
```

Sin embargo, los agentes autónomos requieren **ciclos**. Un agente necesita observar el resultado de una herramienta y decidir: *"¿He terminado?"* o *"¿Debo intentar otra cosa?"*. Las cadenas de LangChain (DAGs por naturaleza en su versión básica) dificultan la implementación de estos ciclos de forma limpia.

LangGraph extiende LangChain permitiendo la creación de grafos que pueden contener ciclos. Pero lo más importante no es el ciclo en sí, sino la **gestión del estado**. En un sistema complejo, necesitamos una "memoria de trabajo" que persista mientras el agente navega por diferentes procesos.

## 2. El Estado: La Memoria Compartida del Grafo

El concepto más fundamental en LangGraph es el **Estado (State)**. Si visualizamos el grafo como una línea de ensamblaje en una fábrica, el Estado es el producto que se mueve por la cinta transportadora. Cada estación (nodo) puede leer el producto, modificarlo o añadirle piezas antes de pasarlo a la siguiente.

### Definición mediante `TypedDict`

En Python 3.12+, definimos el estado utilizando `TypedDict`. Esto nos permite tener un esquema claro y tipado de qué datos están "vivos" en nuestra aplicación.

```python
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Definimos el esquema de nuestra "memoria de trabajo"
class AgentState(TypedDict):
    # El historial de mensajes es el estado más común
    # Usamos Annotated y un "reducer" para decidir cómo se actualiza este campo
    messages: Annotated[List[BaseMessage], add_messages]
    # Podemos añadir campos personalizados para lógica de negocio
    is_escalated: bool
    current_step: str
```

### El concepto de Reducers

Un error común es pensar que un nodo sobrescribe todo el estado. En realidad, los nodos devuelven **actualizaciones**. Si el estado tiene 10 campos y un nodo devuelve un diccionario con 1 campo, solo ese campo se actualiza.

¿Qué pasa si queremos que una lista de mensajes crezca en lugar de sobrescribirse? Usamos un **Reducer**. En el ejemplo de arriba, `add_messages` es una función que le dice a LangGraph: *"No borres los mensajes anteriores, anexa los nuevos a la lista"*.

## 3. Nodos: Las Unidades de Trabajo

Los **Nodos** son simplemente funciones de Python (síncronas o `async`) que toman el estado actual como entrada y devuelven una actualización del estado.

Un nodo no tiene por qué ser un LLM. Puede ser:

- **Un llamado a un LLM:** procesa el historial de mensajes y decide qué hacer.
- **Una herramienta (Tool):** realiza una consulta a una base de datos o API.
- **Lógica de negocio:** valida un JSON, formatea un texto o cambia una bandera en el estado.

### Anatomía de un Nodo

```python
async def call_model(state: AgentState):
    # 1. Recuperamos información del estado
    messages = state['messages']

    # 2. Realizamos la operación (ej: llamado al LLM)
    response = await model.ainvoke(messages)

    # 3. Devolvemos la actualización
    # LangGraph se encarga de fusionar esto con el estado global
    return {"messages": [response]}
```

> **Principio de diseño:** Mantén los nodos pequeños y con una única responsabilidad. Esto facilita las pruebas unitarias y la observabilidad. En LangGraph, un nodo es una unidad de software testeable, no una "caja negra mágica".

## 4. Aristas: El Control de Flujo

Si los nodos son las estaciones de trabajo, las **Aristas (Edges)** son las reglas que definen hacia dónde se mueve el producto después de salir de una estación.

Existen dos tipos principales de aristas:

### A. Aristas Directas (Normal Edges)

Simplemente conectan el Nodo A con el Nodo B: *"Siempre que termines de procesar en A, ve a B"*.

```python
workflow.add_edge("limpiar_texto", "analizar_sentimiento")
```

### B. Aristas Condicionales (Conditional Edges)

Permiten que el flujo se bifurque basándose en la lógica. Se apoyan en una **función de enrutamiento**.

```python
def route_after_analysis(state: AgentState):
    # Si el sentimiento es negativo, escalamos a un humano
    if state["sentiment"] == "negative":
        return "human_support"
    # Si no, terminamos
    return "end"

# En la definición del grafo:
workflow.add_conditional_edges(
    "analizar_sentimiento",
    route_after_analysis,
    {
        "human_support": "nodo_humano",
        "end": END
    }
)
```

Aquí es donde ocurre la "magia" de los agentes autónomos. Una arista condicional puede devolver el flujo a un nodo anterior, creando un **ciclo de reflexión**. Por ejemplo, un nodo "Validador" podría decidir que la respuesta del LLM no es lo suficientemente buena y enviarla de vuelta al nodo "Escritor" para una corrección.

## 5. El Flujo de Trabajo (Compiled Graph)

Una vez definidos los nodos y las aristas, el grafo debe ser **compilado**. Al compilarlo, LangGraph valida que:

- Haya un punto de entrada (`START`).
- Todos los caminos tengan una salida posible o lleguen al nodo especial `END`.
- No existan nodos aislados.

### Metáfora del "Libro de Elige tu propia aventura"

Imagina que cada nodo es una página del libro.

- **El Estado (State)** es lo que el lector recuerda (su inventario, su salud).
- **Los Nodos** son el contenido de la página que el lector lee.
- **Las Aristas** son las instrucciones al final de la página: *"Si decides abrir la puerta, ve a la página 45; si decides correr, ve a la página 12"*.
- **Las Aristas Condicionales** son instrucciones basadas en el inventario: *"Si tienes la llave dorada (estado), ve a la página 100; si no, ve a la página 20"*.

## 6. Errores Comunes y Mejores Prácticas

### Error 1: Cargar el Estado con datos innecesarios

Un error frecuente es intentar meter objetos pesados (como clientes de bases de datos o conexiones de red) dentro del objeto `State`.

> **Mejor práctica:** El estado debe contener solo datos que necesiten persistir o ser compartidos entre nodos. Los objetos de infraestructura deben pasarse a través de la configuración o ser parte de la clase que envuelve al grafo.

### Error 2: Nodos demasiado grandes

Si un nodo hace un llamado al LLM, procesa el resultado, limpia los datos y guarda en la base de datos, estás perdiendo el beneficio de la granularidad de los grafos.

> **Mejor práctica:** Divide las operaciones. Un nodo para el LLM y un nodo para el post-procesamiento. Esto permite insertar aristas condicionales en medio (ej. *"¿Logró el LLM extraer los datos correctamente o necesitamos reintentar?"*).

### Error 3: Olvidar el Reducer en listas de mensajes

Si defines `messages: List[BaseMessage]` en tu `TypedDict` sin un reducer, cada vez que un nodo devuelva `{"messages": [new_msg]}`, la lista anterior desaparecerá.

> **Mejor práctica:** Usa siempre `Annotated[List[BaseMessage], add_messages]` para flujos de chat convencionales.

## 7. Ejemplo resuelto paso a paso: un grafo de soporte con ciclo y escalado

El entregable de esta unidad te pide diseñar el diagrama. Acá ves su contraparte en código, completa y ejecutable: un grafo mínimo que busca una solución, reintenta si no la encuentra (ciclo) y escala a un humano tras 3 intentos. Corre sin API keys, porque simulamos el nodo de búsqueda con lógica simple para que puedas ver la mecánica del grafo.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. El Estado: la "memoria de trabajo" que fluye por el grafo
class SupportState(TypedDict):
    query: str
    attempts: int
    solved: bool
    escalated: bool

# 2. Nodo: busca en la base de conocimiento (simulado)
def search_knowledge(state: SupportState) -> dict:
    intento = state["attempts"] + 1
    encontrado = intento >= 2          # simulamos que recién resuelve en el 2º intento
    print(f"[search] intento {intento} -> {'resuelto' if encontrado else 'sin resultado'}")
    return {"attempts": intento, "solved": encontrado}

# 3. Nodo: escala a un humano
def human_support(state: SupportState) -> dict:
    print("[human] caso escalado a soporte humano")
    return {"escalated": True}

# 4. Arista condicional: decide el camino según el estado
def route(state: SupportState) -> str:
    if state["solved"]:
        return "end"
    if state["attempts"] >= 3:         # corta el ciclo para no quedar infinito
        return "escalate"
    return "retry"                     # vuelve a 'search' -> esto crea el CICLO

# 5. Construcción del grafo
workflow = StateGraph(SupportState)
workflow.add_node("search", search_knowledge)
workflow.add_node("human", human_support)
workflow.add_edge(START, "search")
workflow.add_conditional_edges(
    "search",
    route,
    {"end": END, "escalate": "human", "retry": "search"},
)
workflow.add_edge("human", END)
app = workflow.compile()

# 6. Ejecución
final = app.invoke(
    {"query": "No puedo iniciar sesión", "attempts": 0, "solved": False, "escalated": False}
)
print(final)
```

**Salida esperada:**

```text
[search] intento 1 -> sin resultado
[search] intento 2 -> resuelto
{'query': 'No puedo iniciar sesión', 'attempts': 2, 'solved': True, 'escalated': False}
```

**Qué pasó, paso a paso:**

1. El Estado (`SupportState`) arranca con `attempts=0`. Cada nodo devuelve solo los campos que cambia (una actualización), no el estado completo; LangGraph se encarga de fusionarlo.
2. `START → search`: el primer intento no encuentra solución (`solved=False`).
3. La arista condicional `route` lee el estado: como no está resuelto y aún no llegó a 3 intentos, devuelve `"retry"`, que vuelve al nodo `search`. Ese retorno a un nodo anterior es el **ciclo** que diferencia a LangGraph de una cadena lineal.
4. En el segundo intento `solved=True`, así que `route` devuelve `"end"` y el grafo termina sin escalar.
5. Si la búsqueda nunca resolviera, al llegar a `attempts >= 3` la ruta sería `"escalate"` → nodo `human` → `escalated=True` → `END`. Probalo cambiando la condición `encontrado` a `False` para ver ese camino.

Cuando armes el diagrama del entregable, este código es el "esqueleto" que estás representando: **nodos** = cajas, `add_conditional_edges` = la bifurcación, y el `"retry"` = la flecha que vuelve hacia atrás.

### Segundo ejemplo resuelto: bucle de reflexión (escritor ↔ crítico)

El primer ejemplo usó un ciclo de reintento. Este muestra el otro patrón cíclico clásico de los agentes: el **bucle de reflexión**, donde un nodo "escritor" genera y un nodo "crítico" decide si está lo bastante bueno o si hay que revisar otra vez. Corre sin API keys (simulamos que la calidad mejora en cada vuelta).

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Estado: el borrador, su puntaje de calidad y cuántas veces lo revisamos
class WriteState(TypedDict):
    tema: str
    borrador: str
    score: int
    revisiones: int

# 2. Nodo escritor: genera/mejora el borrador (simulado: suma 3 puntos por vuelta)
def escribir(state: WriteState) -> dict:
    nuevo_score = state["score"] + 3
    print(f"[escritor] revisión {state['revisiones'] + 1} -> score {nuevo_score}")
    return {
        "borrador": f"Borrador sobre '{state['tema']}' (calidad {nuevo_score}/10)",
        "score": nuevo_score,
        "revisiones": state["revisiones"] + 1,
    }

# 3. Nodo crítico: no modifica el estado; la decisión la toma la arista condicional
def criticar(state: WriteState) -> dict:
    return {}

# 4. Arista condicional: ¿aprobamos, nos rendimos o seguimos revisando?
def route(state: WriteState) -> str:
    if state["score"] >= 8:
        return "aprobado"
    if state["revisiones"] >= 4:        # tope para no iterar infinito
        return "rendirse"
    return "revisar"                    # vuelve al escritor -> CICLO de reflexión

# 5. Construcción del grafo
g = StateGraph(WriteState)
g.add_node("escritor", escribir)
g.add_node("critico", criticar)
g.add_edge(START, "escritor")
g.add_edge("escritor", "critico")
g.add_conditional_edges(
    "critico",
    route,
    {"aprobado": END, "rendirse": END, "revisar": "escritor"},
)
app = g.compile()

# 6. Ejecución
final = app.invoke({"tema": "onboarding de usuarios", "borrador": "", "score": 0, "revisiones": 0})
print(f"Final tras {final['revisiones']} revisiones | score {final['score']}")
print(final["borrador"])
```

**Salida esperada:**

```text
[escritor] revisión 1 -> score 3
[escritor] revisión 2 -> score 6
[escritor] revisión 3 -> score 9
Final tras 3 revisiones | score 9
Borrador sobre 'onboarding de usuarios' (calidad 9/10)
```

**Qué pasó, paso a paso:**

1. El flujo es `escritor → crítico → (decisión)`. El crítico no cambia el estado; solo es el punto donde la arista condicional evalúa.
2. Tras cada vuelta, `route` mira el `score`: si llega a 8 termina (`aprobado`); si no, vuelve al escritor (`revisar`). Ese retorno es el **ciclo de reflexión**.
3. La condición `revisiones >= 4` es un **freno de seguridad**: sin él, un borrador que nunca llega a 8 generaría un loop infinito.
4. En 3 vueltas el score pasa 3 → 6 → 9 y se aprueba.

A diferencia del primer ejemplo (ciclo de reintento ante un fallo), este es un **ciclo de mejora iterativa**: el sistema se autocritica hasta alcanzar un umbral de calidad. Es la base de los agentes "reflexivos".

## 8. Síntesis y Conexiones

En esta unidad hemos sentado las bases arquitectónicas de los sistemas de IA del futuro. En lugar de scripts lineales, ahora diseñamos sistemas dinámicos.

**Puntos clave:**

- **Estado:** la única fuente de verdad tipada que fluye por el sistema.
- **Nodos:** funciones aisladas que transforman el estado.
- **Aristas:** reglas de navegación que permiten ciclos y bifurcaciones.

En la siguiente unidad, daremos vida a estos nodos integrando **Tool Calling**, permitiendo que el estado no solo guarde texto, sino también los resultados de acciones ejecutadas en el mundo real. ¡Prepárate para conectar tu grafo con herramientas externas!

---

# Unidad 2 · Tool Calling: definición y ejecución segura

En el desarrollo de sistemas de IA, un LLM "desnudo" es como un cerebro brillante en una habitación sin ventanas: puede razonar, pero no puede interactuar con el mundo exterior. El **Tool Calling** (invocación de herramientas) es el mecanismo que le da "manos" a ese cerebro.

## 1. ¿Qué es Tool Calling?

El modelo **no ejecuta código Python directamente**. En su lugar, genera una solicitud estructurada (usualmente JSON) indicando qué función quiere ejecutar y con qué parámetros. Tu aplicación intercepta esa solicitud, ejecuta el código real y le devuelve el resultado al modelo.

## 2. El "contrato" de la herramienta

Para que un modelo como GPT-4 o Claude 3.5 use una herramienta con éxito, necesita un contrato claro. Este contrato tiene cuatro pilares:

- **Nombre:** identificador único (ej. `get_stock_price`).
- **Descripción:** explicación semántica. El LLM decide si usar la herramienta basándose en esto.
- **Esquema de entrada (Input Schema):** definición estricta de variables (tipos, obligatoriedad).
- **Esquema de salida:** qué esperar de vuelta.

## 3. Definición segura con Pydantic y LangChain

En entornos escalables no podemos confiar en que el LLM envíe siempre los datos correctos. Usamos **Pydantic** para validar los tipos antes de que el código toque nuestra infraestructura.

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class CalculatorInput(BaseModel):
    a: int = Field(description="Primer número")
    b: int = Field(description="Segundo número")

@tool(args_schema=CalculatorInput)
def multiply(a: int, b: int) -> int:
    """Multiplica dos números enteros."""
    return a * b
```

## 4. El ciclo de ejecución (Reason-Act Loop)

1. **Reasoning:** el LLM recibe el prompt y las definiciones de herramientas. Decide que necesita `multiply`.
2. **Tool Call:** el LLM responde con un objeto especial: `tool_calls: [{name: 'multiply', args: {a: 5, b: 2}}]`.
3. **Execution (local):** tu servidor detecta esta intención, valida los argumentos con Pydantic y ejecuta la función `multiply(5, 2)`.
4. **Observation:** el resultado (`10`) se envía de vuelta al LLM en un nuevo mensaje con el rol `tool`.
5. **Final Response:** el LLM procesa el `10` y responde al usuario: *"El resultado es 10"*.

## 5. Errores comunes y seguridad

### Error 1: Confundir invocación con ejecución

El LLM no ejecuta el código. Tu arquitectura debe garantizar que las herramientas tengan permisos limitados (**Least Privilege**).

> **Mejor práctica:** Nunca permitas que una herramienta ejecute SQL raw generado por el LLM sin validación.

### Error 2: Descripciones ambiguas

Si tu descripción es *"Busca datos"*, el modelo fallará.

> **Mejor práctica:** Sé específico: *"Busca precios de acciones en tiempo real para empresas del S&P 500"*.

### Error 3: Ignorar errores

Si la herramienta falla (ej. timeout de API) y no se maneja, el agente se rompe o queda sin contexto.

> **Mejor práctica:** Captura el error y envíalo como un string al LLM, para que este pueda explicarle al usuario qué salió mal o intentar otra estrategia.

## 6. Síntesis y conexiones

En la siguiente práctica conectarás este concepto con el sistema RAG que construiste anteriormente, permitiendo que tu agente decida cuándo consultar la base de conocimientos.

---

# Unidad 3 · Ciclos ReAct y persistencia de checkpoints en agentes autónomos

Como arquitecto de IA, tu objetivo no es solo que el sistema responda, sino que sea **resiliente**. En sistemas complejos, un agente puede fallar a mitad de una ejecución de 10 pasos debido a un error de red o un timeout. Sin persistencia, perderías todo el razonamiento y los resultados parciales (y el dinero gastado en tokens).

## 1. El ciclo ReAct (Reasoning + Acting)

El patrón **ReAct** es el estándar de oro para agentes autónomos. A diferencia de una cadena lineal, ReAct permite al LLM:

- **Pensar (Reason):** analizar la solicitud del usuario y decidir qué necesita.
- **Actuar (Act):** seleccionar una herramienta y generar los parámetros de llamada.
- **Observar (Observe):** recibir la salida de la herramienta y re-evaluar el estado.

Este proceso es **cíclico**. El LLM puede decidir que necesita usar tres herramientas diferentes antes de dar una respuesta final. LangGraph facilita esto permitiendo que una arista (edge) vuelva al mismo nodo o a un nodo anterior, creando un historial de mensajes que crece con cada "observación".

## 2. Checkpoints: la caja negra de tu agente

En LangGraph, un **Checkpointer** es un mecanismo de almacenamiento que guarda una "fotografía" del estado del grafo después de cada nodo ejecutado.

- **Para qué sirve:** si el proceso se detiene, puedes reanudarlo simplemente pasando el mismo `thread_id`. El sistema buscará en la base de datos (SQLite, Redis o memoria) y cargará el último estado conocido.
- **Time Travel:** los checkpoints permiten no solo reanudar, sino "volver atrás" en el tiempo (debugging) o requerir aprobación humana antes de continuar con un nodo crítico (**Human-in-the-loop**).

## 3. Enunciado del problema

Vas a construir un **agente climatológico** capaz de razonar si necesita herramientas y que no pierda la memoria incluso si interrumpimos la ejecución del script.

### Objetivo

- Implementar un ciclo ReAct completo.
- Configurar persistencia en memoria.
- Demostrar que el agente recuerda la ubicación mencionada en el primer mensaje al recibir un segundo mensaje, gracias al `thread_id`.

### Restricciones

- Usa `Annotated[list[BaseMessage], add_messages]` en tu estado para asegurar que los mensajes no se sobrescriban, sino que se concatenen.
- El grafo debe terminar solo cuando el LLM decida que no hay más herramientas que llamar.
- No olvides que el nodo de herramientas (`ToolNode`) debe mapear sus resultados de vuelta al agente.
