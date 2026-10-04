"""Helpers de formato compartidos por las tools y el generador de datos."""


def ars(valor: float) -> str:
    """Formato de moneda argentino: $1.234.567 (negativos como -$1.234)."""
    signo = "-" if valor < 0 else ""
    return f"{signo}$" + f"{abs(round(valor)):,}".replace(",", ".")
