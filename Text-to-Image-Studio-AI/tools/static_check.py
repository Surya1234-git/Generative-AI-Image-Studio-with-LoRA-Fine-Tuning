import ast, builtins, sys, symtable
from pathlib import Path
ROOT = Path(".")
files = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts]
def modname(p): return ".".join(p.with_suffix("").parts).replace(".__init__", "")
tops = {}
for p in files:
    tree = ast.parse(p.read_text())
    names = set(dir(builtins)) | {"__file__", "__name__"}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)): names.add(n.name)
        elif isinstance(n, ast.Import): names |= {(a.asname or a.name.split(".")[0]) for a in n.names}
        elif isinstance(n, ast.ImportFrom): names |= {(a.asname or a.name) for a in n.names}
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in ast.walk(n):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store): names.add(t.id)
        elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For)):
            for t in ast.walk(n):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store): names.add(t.id)
                if isinstance(t, (ast.FunctionDef, ast.ClassDef)): names.add(t.name)
                if isinstance(t, ast.Import): names |= {(a.asname or a.name.split(".")[0]) for a in t.names}
                if isinstance(t, ast.ImportFrom): names |= {(a.asname or a.name) for a in t.names}
    tops[modname(p)] = names
problems = []
def walk_tables(tab, modnames, path):
    for sym in tab.get_symbols():
        if sym.is_global() and sym.is_referenced() and not sym.is_assigned() and tab.get_type() != "module":
            if sym.get_name() not in modnames:
                problems.append(f"{path}: undefined name '{sym.get_name()}' in {tab.get_name()}()")
    for ch in tab.get_children(): walk_tables(ch, modnames, path)
for p in files:
    src = p.read_text(); m = modname(p)
    walk_tables(symtable.symtable(src, str(p), "exec"), tops[m], p)
    tree = ast.parse(src); alias = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and (n.module.split(".")[0] in {"modules","services","components","config","views"}):
            for a in n.names:
                full = f"{n.module}.{a.name}"
                if full in tops: alias[a.asname or a.name] = full          # imported a submodule
                elif n.module in tops and a.name not in tops[n.module]:
                    problems.append(f"{p}: cannot import '{a.name}' from {n.module}")
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias:
            if n.attr not in tops[alias[n.value.id]]:
                problems.append(f"{p}:{n.lineno}: module {alias[n.value.id]} has no attribute '{n.attr}'")
print("files checked:", len(files)); print("\n".join(problems) or "NO PROBLEMS FOUND")
