# LangGraph: Mejores prácticas y errores comunes

Guía de referencia para trabajar con LangGraph en esta entrega. Fuente: `teoria/LangGraph.md`.

## Conceptos base

- **Estado (State):** la memoria de trabajo compartida que fluye por el grafo. Es la única fuente de verdad tipada del sistema.
- **Nodos:** funciones de Python (síncronas o `async`) que reciben el estado y devuelven una **actualización** del estado.
- **Aristas:** reglas de navegación entre nodos. Pueden ser directas (`add_edge`) o condicionales (`add_conditional_edges`), y permiten ciclos y bifurcaciones.

## Mejores prácticas

### Estado

- Definir el estado con `TypedDict`, para tener un esquema claro y tipado de los datos "vivos" del grafo.
- Guardar en el estado solo datos que necesiten persistir o compartirse entre nodos.
- Para listas que deben crecer (ej. historial de mensajes), usar un reducer: `Annotated[List[BaseMessage], add_messages]`.
- Importar el reducer explícitamente: `from langgraph.graph.message import add_messages`.
- Al hacer `invoke`, pasar el estado inicial con **todos** los campos que los nodos van a leer (ej. `attempts: 0`, `solved: False`).

### Nodos

- Mantener los nodos pequeños y con **una única responsabilidad**. Un nodo es una unidad de software testeable, no una "caja negra mágica".
- Devolver solo los campos que el nodo cambia (`return {"attempts": intento}`), no el estado completo; LangGraph se encarga de fusionarlo.
- Un nodo no tiene por qué ser un LLM: puede ser una herramienta (DB/API) o lógica de negocio (validar JSON, formatear texto, cambiar una bandera).
- Separar el llamado al LLM del post-procesamiento en nodos distintos, para poder insertar aristas condicionales entre ellos (ej. reintentar si el LLM no extrajo bien los datos).
- Un nodo puede devolver `{}` si solo funciona como punto de decisión (ej. el nodo "crítico"); la decisión la toma la arista condicional.

### Aristas y ciclos

- Usar aristas condicionales con una **función de enrutamiento** que lee el estado y devuelve una clave; mapear **todas** las claves posibles a un nodo o a `END`.
- Todo ciclo (reintento o reflexión) debe tener un **freno de seguridad** en el estado: un contador (`attempts`, `revisiones`) con un tope que corta el loop.
- Definir qué pasa cuando se alcanza el tope: escalar (ej. a soporte humano) o terminar (`"rendirse": END`).
- Patrones cíclicos de referencia:
  - **Ciclo de reintento:** ante un fallo, volver al mismo nodo (`"retry": "search"`).
  - **Ciclo de reflexión:** escritor → crítico → (aprobar | revisar), hasta alcanzar un umbral de calidad.

### Compilación

Antes de `compile()`, verificar que:

- Haya un punto de entrada (`START`).
- Todos los caminos tengan una salida posible o lleguen al nodo especial `END`.
- No existan nodos aislados.

### Desarrollo y diseño

- Probar la mecánica del grafo simulando los nodos con lógica simple (sin API keys) antes de conectar LLMs reales.
- Al diagramar: nodos = cajas, `add_conditional_edges` = bifurcación, ciclo (`retry`/`revisar`) = flecha que vuelve hacia atrás.

## Errores comunes

### Error 1: Cargar el Estado con datos innecesarios

Meter objetos pesados (clientes de bases de datos, conexiones de red) dentro del `State`.

> **Mejor práctica:** El estado debe contener solo datos que necesiten persistir o ser compartidos entre nodos. Los objetos de infraestructura deben pasarse a través de la configuración o ser parte de la clase que envuelve al grafo.

### Error 2: Nodos demasiado grandes

Un nodo que llama al LLM, procesa el resultado, limpia los datos y guarda en la base de datos pierde el beneficio de la granularidad de los grafos.

> **Mejor práctica:** Dividir las operaciones. Un nodo para el LLM y un nodo para el post-procesamiento. Esto permite insertar aristas condicionales en medio (ej. "¿Logró el LLM extraer los datos correctamente o necesitamos reintentar?").

### Error 3: Olvidar el Reducer en listas de mensajes

Si se define `messages: List[BaseMessage]` en el `TypedDict` sin un reducer, cada vez que un nodo devuelva `{"messages": [new_msg]}`, la lista anterior desaparecerá.

> **Mejor práctica:** Usar siempre `Annotated[List[BaseMessage], add_messages]` para flujos de chat convencionales.

### Error 4: Pensar que un nodo sobrescribe todo el estado

Los nodos devuelven actualizaciones: si el estado tiene 10 campos y el nodo devuelve 1, solo ese campo cambia. Devolver el estado completo "por las dudas" es innecesario y oculta qué modifica cada nodo.

> **Mejor práctica:** Devolver únicamente los campos que el nodo modifica.

### Error 5: Ciclos sin condición de corte

Una arista condicional que vuelve a un nodo anterior sin un tope genera un loop infinito si la condición de éxito nunca se cumple (ej. un borrador que nunca llega al score mínimo).

> **Mejor práctica:** Llevar un contador en el estado y cortar el ciclo al alcanzar un máximo (`attempts >= 3` → escalar; `revisiones >= 4` → rendirse).

### Error 6: Resolver la lógica con `if/else` o loops de chat en lugar de un grafo

Programar flujos con bifurcaciones (resolver, pedir más info, escalar) con cadenas `if/else` interminables genera código frágil, difícil de depurar y casi imposible de escalar.

> **Mejor práctica:** Modelar cada paso como un nodo y cada decisión como una arista condicional.
