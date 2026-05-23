def canonical_name(name: str) -> str:
    """
    Convert 'Carlos Alcaraz' → 'Alcaraz C.'
    Leave already-canonical names untouched
    """
    if "," in name or "." in name:
        return name.strip()

    parts = name.strip().split()
    if len(parts) < 2:
        return name.strip()

    first = parts[0]
    last = parts[-1]
    return f"{last} {first[0]}."
