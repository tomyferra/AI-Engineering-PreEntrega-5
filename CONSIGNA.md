# Pre-entrega 5: Agente de razonamiento cíclico con memoria persistente

## Criterios de aceptación

Para que este checkpoint se considere aprobado, el entregable debe cumplir con lo siguiente:

- **Autonomía:** el agente debe ser capaz de determinar por sí mismo cuándo llamar a una herramienta basándose en el prompt del usuario (sin rutas manuales `if/else`).
- **Ciclo de retorno:** si una herramienta devuelve un error o información incompleta, el agente debe ser capaz de realizar un segundo intento o pedir aclaraciones.
- **Resiliencia de estado:** al proporcionar un `thread_id`, el agente debe ser capaz de recordar interacciones previas dentro de una misma sesión de razonamiento.
- **Código limpio:** uso de Python 3.12, tipado estático (Type Hints) y gestión asíncrona (`asyncio`).

## Guía de implementación sugerida

### Fase 1: El contrato de herramientas

Define tus funciones utilizando el decorador `@tool` de LangChain. Asegúrate de incluir docstrings extremadamente descriptivos; recuerda que el LLM decide qué herramienta usar basándose únicamente en esa descripción.

### Fase 2: Definición del estado y el grafo

Crea el esquema de tu estado. En LangGraph, el estado es inmutable y se actualiza mediante reducers (usualmente `operator.add` para la lista de mensajes). Configura el `StateGraph` conectando el nodo del modelo con el nodo de ejecución de herramientas mediante una arista condicional (`tools_condition`).

### Fase 3: Persistencia

Configura un Checkpointer. Esto es lo que permite que el agente sea "arquitectónicamente escalable". Sin persistencia, tu agente es efímero; con ella, es capaz de manejar flujos de trabajo largos que requieren intervención humana o esperar procesos externos.

## Errores comunes a evitar

- **Descripciones vagas:** si el agente no usa la herramienta que esperas, el error suele estar en el docstring de la función, no en la lógica del grafo.
- **Bucles infinitos:** no establecer un límite de recursión (`recursion_limit`) al invocar el grafo. Define siempre un techo (ej. 10 pasos) para evitar costos inesperados en la API.
- **Estado sucio:** olvidar que el estado se acumula. Asegúrate de limpiar o resumir mensajes si el contexto se vuelve demasiado grande.

## 📦 Qué entregás y en qué formato

- **Tipo:** 💻 Código — un repositorio público de GitHub.
- **Artefacto concreto:** repo con el `StateGraph`, al menos una herramienta propia, persistencia (`SqliteSaver`), la traza ReAct como log o `.json`, y un `README.md`.
- **Qué NO hace falta:** no entregás documento aparte; la traza del razonamiento se entrega como log/JSON dentro del repo.

Repositorio de GitHub que contenga el código del agente en un entorno asíncrono, incluyendo la definición del grafo, las herramientas y la configuración del checkpointer. Debe incluir un archivo README explicando cómo levantar el entorno y un ejemplo de traza de ejecución (`.json` o log).

## Entregable

1. Inicializa un proyecto Python 3.12+ con Poetry o venv.
2. Define un `StateGraph` que herede de `MessagesState`.
3. Crea al menos una herramienta personalizada que simule una operación de base de datos o búsqueda técnica.
4. Configura un LLM (OpenAI o Anthropic) vinculado a las herramientas (`llm.bind_tools()`).
5. Implementa la lógica de persistencia usando `SqliteSaver` (para desarrollo local).
6. Realiza una prueba de ejecución donde el agente deba llamar a la herramienta al menos dos veces para llegar a la conclusión (razonamiento multi-paso).
7. Sube el código a un repositorio público y asegúrate de no incluir las API Keys (usa variables de entorno).

## Ejemplo de la traza esperada (razonamiento cíclico)

Tu agente debe mostrar un ciclo ReAct como este (incluilo como log o `.json` en el repo):

```text
Usuario: "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"
→ El agente decide usar la herramienta: buscar_pedidos(cliente_id=102)
→ La herramienta devuelve: { "pedidos": 3, "total": 14500 }
→ El agente razona: ya tiene los datos → responde.
Respuesta: "El cliente 102 tuvo 3 pedidos por un total de $14.500."
```

(Con el mismo `thread_id`, si después preguntás "¿y el último?", el agente recuerda el contexto.)

## Checklist de entrega

- [ ] Repo público (sin API keys; variables de entorno) con README de cómo levantar el entorno.
- [ ] `StateGraph` (hereda de `MessagesState`) con nodo de modelo + nodo de herramientas y arista condicional (`tools_condition`).
- [ ] Al menos 1 herramienta con `@tool` y docstring descriptivo.
- [ ] Persistencia con `SqliteSaver` + `thread_id` (recuerda la sesión).
- [ ] Prueba con razonamiento multi-paso (la herramienta se invoca ≥2 veces) y `recursion_limit` definido.
- [ ] Ejemplo de traza de ejecución (`.json` o log) incluido.
