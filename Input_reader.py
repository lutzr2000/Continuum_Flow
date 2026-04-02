

def read_value_from_input(filename, keyword):
    for line in open(filename):
        if line.strip().startswith("#") or "=" not in line:
            continue
        k, v = map(str.strip, line.split("=", 1))
        if k == keyword:
            if v.lower() in ["true", "false"]:
                return v.lower() == "true"
            try:
                return int(v)
            except ValueError:
                try:
                    return float(v)
                except ValueError:
                    return v
    raise KeyError(f"{keyword} nicht gefunden")