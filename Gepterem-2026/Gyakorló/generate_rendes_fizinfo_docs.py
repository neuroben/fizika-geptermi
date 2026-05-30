from __future__ import annotations

import ast
import html
import importlib.util
import inspect
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "RENDES_fizinfo.py"
OUTPUT_PATH = ROOT / "RENDES_fizinfo_dokumentacio.html"
MODULE_ALIAS = "rendes_fizinfo_docs_source"


TOP_LEVEL_SECTIONS = OrderedDict(
    [
        (
            "altalanos",
            {
                "title": "Általános számítási függvények",
                "members": ["deriv", "deriv_nd", "integ", "integ_nd", "vect_abs", "arg_eq"],
                "description": "Alap numerikus rutinok deriváláshoz, integráláshoz és egyszerű vektorműveletekhez.",
            },
        ),
        (
            "gps",
            {
                "title": "GPS és smoothing",
                "members": ["GPS_Logger_to_xyt", "GPS_to_num_kinem", "num_kinem_smooth_r"],
                "description": "GPS-adatok beolvasása, num_kinem objektummá alakítása és simítása.",
            },
        ),
        (
            "modellek",
            {
                "title": "Kész fizikai modellek",
                "members": ["create_grav_közeg_talaj_F_m", "create_rocket_F_m"],
                "description": "Előre elkészített erő- és tömegfüggvény-generátorok tipikus dinamikai modellekhez.",
            },
        ),
    ]
)


TOP_LEVEL_SUMMARIES = {
    "deriv": "Táblázatosan megadott függvény numerikus deriváltját számolja; tipikusan helyből sebességet vagy sebességből gyorsulást ad.",
    "deriv_nd": "Többdimenziós adatsor numerikus deriváltja, például vektorhelyzetből sebességvektor.",
    "integ": "Egyváltozós adatsor numerikus integrálja; például gyorsulásból sebesség vagy sebességből út.",
    "integ_nd": "Többdimenziós mennyiség numerikus integrálja, vektoros pályaszámításhoz.",
    "vect_abs": "Vektorok hosszát számolja ki, vagyis a nagyságot irány nélkül.",
    "arg_eq": "Megközelítő egyezést keres tömbben; tipikusan események vagy küszöbátlépések megtalálására hasznos.",
    "GPS_Logger_to_xyt": "GPS logger fájlból idő- és helyadatokat olvas be továbbfeldolgozáshoz.",
    "GPS_to_num_kinem": "GPS-fájlból közvetlenül num_kinem objektumot épít fel.",
    "num_kinem_smooth_r": "Zajos kinematikai pályaadatokat simít és új num_kinem objektumot készít.",
    "create_grav_közeg_talaj_F_m": "Kész erő- és tömegfüggvényt ad labda típusú mozgásokhoz gravitációval, közegellenállással és talajmodellel.",
    "create_rocket_F_m": "Kész erő- és tömegfüggvényt ad rakétamodellhez gravitációval, légellenállással és tolóerővel.",
}


CLASS_SUMMARIES = {
    "num_kinem": {
        "summary": "Kinematikai adatokból dolgozó alap class: időrácsot, helyet, sebességet, gyorsulást és sok származtatott mennyiséget kezel.",
        "meaning": "Akkor hasznos, ha a pályát vagy annak valamelyik komponensét analitikusan vagy mérésből ismerjük, és ebből akarunk sebességet, gyorsulást, pályahosszt vagy speciális lekérdezéseket számolni.",
        "workflow": "Jellemző használat: időrács beállítása → r/v/a függvény megadása vagy adatok betöltése → full_kinem_calc() → lekérdezések és ábrák.",
    },
    "num_dinam": {
        "summary": "Dinamikai szimulációs class, amely erő- és tömegfüggvényből numerikusan integrálja a mozgást.",
        "meaning": "Akkor kell, ha a mozgás nem közvetlenül adott, hanem az erőhatásokból és a kezdőfeltételekből kell kiszámolni a teljes pályát.",
        "workflow": "Jellemző használat: időparaméterek → tömegfüggvény → erőfüggvény → kezdőhely és kezdősebesség → opcionális leállási feltétel → full_dinam_calc().",
    },
}


PROPERTY_SUMMARIES = {
    "r_x": "Az x koordináták gyors elérése a teljes helyvektor-sorozatból.",
    "r_y": "Az y koordináták gyors elérése a teljes helyvektor-sorozatból.",
    "v_x": "A sebesség x komponense.",
    "v_y": "A sebesség y komponense.",
    "v_abs": "Sebességnagyság, vagyis a haladás pillanatnyi tempója.",
    "a_abs": "Gyorsulásnagyság, iránytól függetlenül.",
    "dist": "Az origótól mért pillanatnyi távolság.",
    "max_height": "A pálya legnagyobb y koordinátája.",
    "final_dist": "A végpont origótól mért távolsága.",
    "pathlength": "Az indulástól összegzett pályahossz.",
    "a_t_abs": "Előjeles tangenciális gyorsulás; negatív érték lassulást jelent.",
    "a_t_magnitude": "A tangenciális gyorsulás nagysága előjel nélkül.",
    "a_cp_abs": "A centripetális gyorsulás nagysága.",
    "Rinv": "A görbületi sugár reciproka; nagyobb érték szorosabb kanyart jelez.",
    "m": "Dinamikai szimulációnál a pillanatnyi tömeg idősora.",
}


METHOD_SUMMARIES = {
    "set_time_range": "Beállítja a kinematikai számítás időrácsát.",
    "set_time_param": "Beállítja a dinamikai szimuláció időparamétereit.",
    "set_r_fun": "Analitikus helyfüggvényt ad meg a kinematikai objektumnak.",
    "set_v_fun": "Analitikus sebességfüggvényt ad meg.",
    "set_a_fun": "Analitikus gyorsulásfüggvényt ad meg.",
    "set_mass_fun": "Beállítja a dinamikai modell tömegfüggvényét.",
    "set_F_fun": "Beállítja a dinamikai modell eredő erőfüggvényét.",
    "set_stop_cond": "Leállási feltételt ad a dinamikai integráláshoz.",
    "set_r0_v0": "Beállítja a kezdőhelyet és a kezdősebességet.",
    "calc_r_to_v": "Hely-idő adatokból sebességet számít numerikusan.",
    "calc_v_to_a": "Sebesség-idő adatokból gyorsulást számít numerikusan.",
    "calc_a_to_v": "Gyorsulásból sebességet integrál.",
    "calc_v_to_r": "Sebességből helyet integrál.",
    "calc_delta_r": "Szomszédos pontok közötti elmozdulásvektorokat számít.",
    "calc_delta_r_abs": "Az elmozdulásvektorok hosszát számítja ki.",
    "calc_pathlength": "Az indulástól mért összegzett pályahosszt építi fel.",
    "calc_at_acp_Rinv": "Tangenciális és centripetális gyorsulást, valamint görbületi adatokat számol.",
    "full_kinem_calc": "A kinematikai származtatott mennyiségek teljes láncát kiszámolja.",
    "full_dinam_calc": "A teljes dinamikai szimulációt lefuttatja az erőmodell alapján.",
    "plot_simple": "Egyszerű időfüggő ábrákat rajzol egy vagy több mennyiségről.",
    "plot_rva_coord": "A hely-, sebesség- és gyorsuláskomponenseket rajzolja ki.",
    "plot_rcomp": "Síkbeli pályagörbét rajzol koordinátakomponensekből.",
    "clip_below_y": "Levágja a pályát egy választott y szint alatt.",
    "y_values_at_x": "Megadja, hogy adott x helyen milyen y értékeken megy át a pálya.",
    "x_values_at_y": "Megadja, hogy adott y szinten milyen x értékeken megy át a pálya.",
    "passes_near_y_at_x": "Azt ellenőrzi, hogy adott x helyen a pálya közel megy-e egy célmagassághoz.",
    "passes_near_x_at_y": "Azt ellenőrzi, hogy adott y szinten a pálya közel megy-e egy cél-x értékhez.",
    "flies_over_obstacle": "Megmondja, hogy a pálya átrepül-e egy megadott akadály fölött.",
    "x_ranges_above_height": "Azokat az x-intervallumokat adja meg, ahol a pálya egy magasság fölött halad.",
    "kinetic_energy": "Mozgási energiát számít állandó vagy időfüggő tömegre.",
    "kinetic_energy_constant_m": "Mozgási energiát számít állandó tömegre egyszerűsített formában.",
    "Newton_step": "A dinamikai integrálás egyetlen lépését hajtja végre.",
    "__init__": "Létrehozza az objektumot a kívánt dimenziószámmal és alapállapottal.",
}


GROUP_TITLES = OrderedDict(
    [
        ("constructor", "Létrehozás"),
        ("setup", "Beállítás"),
        ("calculation", "Számítás"),
        ("query", "Lekérdezés és helper"),
        ("plot", "Ábrázolás"),
        ("properties", "Property-k"),
        ("other", "Egyéb publikus elemek"),
    ]
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(MODULE_ALIAS, SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_source() -> tuple[ast.Module, str]:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(SOURCE_PATH)), source


def get_all_names(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
    return []


def first_sentence(text: str) -> str:
    cleaned = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    if not cleaned:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", cleaned)
    return match.group(1).strip() if match else cleaned


def short_summary(name: str, doc: str, fallback_map: dict[str, str]) -> str:
    if name in fallback_map:
        return fallback_map[name]
    sentence = first_sentence(doc)
    if sentence:
        return sentence
    return f"A(z) `{name}` publikus API-elem rövid leírása még nincs kézzel finomhangolva."


def extract_sections(doc: str) -> dict[str, str]:
    if not doc:
        return {}
    result: dict[str, str] = {}
    headings = ["Fizikai jelentés", "Paraméterek", "Visszatérési érték", "Hatások", "Példák"]
    pattern = re.compile(rf"^({'|'.join(re.escape(h) for h in headings)}):\s*$", re.MULTILINE)
    matches = list(pattern.finditer(doc))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        result[match.group(1)] = doc[start:end].strip()
    return result


def format_signature(callable_obj: Any, name: str, *, class_name: str | None = None) -> str:
    try:
        if isinstance(callable_obj, property):
            signature = inspect.signature(callable_obj.fget)
        else:
            signature = inspect.signature(callable_obj)
        prefix = f"{class_name}." if class_name else ""
        return f"{prefix}{name}{signature}"
    except (TypeError, ValueError):
        prefix = f"{class_name}." if class_name else ""
        return f"{prefix}{name}(...)"


def method_group(name: str, item_type: str) -> str:
    if item_type == "property":
        return "properties"
    if name == "__init__":
        return "constructor"
    if name.startswith("set_"):
        return "setup"
    if name.startswith("calc_") or name.startswith("full_") or name == "Newton_step":
        return "calculation"
    if name.startswith("plot_"):
        return "plot"
    if (
        "_at_" in name
        or name.startswith("flies_")
        or name.startswith("x_ranges_")
        or name.startswith("passes_")
        or name.startswith("clip_")
        or name.startswith("kinetic_energy")
    ):
        return "query"
    return "other"


def render_text_block(text: str) -> str:
    if not text:
        return ""
    paragraphs = [f"<p>{html.escape(' '.join(line.split()))}</p>" for line in text.split("\n\n") if line.strip()]
    return "".join(paragraphs)


def render_section_list(section_text: str) -> str:
    if not section_text:
        return ""
    lines = [line.rstrip() for line in section_text.splitlines() if line.strip()]
    if not lines:
        return ""
    parts = ["<ul>"]
    for line in lines:
        parts.append(f"<li>{html.escape(line.strip())}</li>")
    parts.append("</ul>")
    return "".join(parts)


def search_blob(*parts: str) -> str:
    return " ".join(part for part in parts if part).lower()


def badge(label: str, kind: str = "default") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(label)}</span>'


def render_api_item(item: dict[str, Any]) -> str:
    sections = item["sections"]
    summary = html.escape(item["summary"])
    description = render_text_block(item["doc"]) if item["doc"] else "<p class='muted'>Nincs külön docstring-szöveg.</p>"
    params_html = render_section_list(sections.get("Paraméterek", ""))
    returns_html = render_text_block(sections.get("Visszatérési érték", ""))
    physical_html = render_text_block(sections.get("Fizikai jelentés", ""))
    effects_html = render_text_block(sections.get("Hatások", ""))

    extra_blocks = []
    if params_html:
        extra_blocks.append(f"<section><h5>Paraméterek</h5>{params_html}</section>")
    if returns_html:
        extra_blocks.append(f"<section><h5>Visszatérési érték</h5>{returns_html}</section>")
    if physical_html:
        extra_blocks.append(f"<section><h5>Fizikai jelentés</h5>{physical_html}</section>")
    if effects_html:
        extra_blocks.append(f"<section><h5>Hatások</h5>{effects_html}</section>")

    kind_badge = badge(item["type_label"], item["type"])
    group_badge = badge(item["group_title"], "group") if item.get("group_title") else ""

    return f"""
    <details class="api-item" id="{html.escape(item['anchor'])}" data-search="{html.escape(item['search'])}">
      <summary>
        <div class="summary-head">
          <div class="summary-title">
            {kind_badge}
            {group_badge}
            <span class="item-name">{html.escape(item['display_name'])}</span>
          </div>
          <code class="signature">{html.escape(item['signature'])}</code>
        </div>
        <div class="summary-text">{summary}</div>
      </summary>
      <div class="item-body">
        <section>
          <h5>Rövid leírás</h5>
          {description}
        </section>
        {''.join(extra_blocks)}
      </div>
    </details>
    """


def collect_top_level_data(module: Any, tree: ast.Module, public_names: list[str]) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in public_names:
            obj = getattr(module, node.name)
            doc = inspect.getdoc(obj) or ""
            data[node.name] = {
                "name": node.name,
                "display_name": node.name,
                "type": "function",
                "type_label": "függvény",
                "signature": format_signature(obj, node.name),
                "summary": short_summary(node.name, doc, TOP_LEVEL_SUMMARIES),
                "doc": doc,
                "sections": extract_sections(doc),
                "anchor": f"fn-{node.name}",
                "search": search_blob(node.name, doc, TOP_LEVEL_SUMMARIES.get(node.name, "")),
            }
    return data


def is_property_getter(node: ast.FunctionDef) -> bool:
    return any(isinstance(dec, ast.Name) and dec.id == "property" for dec in node.decorator_list)


def is_property_setter(node: ast.FunctionDef) -> bool:
    return any(isinstance(dec, ast.Attribute) and dec.attr == "setter" for dec in node.decorator_list)


def collect_class_data(module: Any, tree: ast.Module) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name not in {"num_kinem", "num_dinam"}:
            continue

        cls_obj = getattr(module, node.name)
        class_doc = inspect.getdoc(cls_obj) or ""
        class_info = {
            "name": node.name,
            "summary": CLASS_SUMMARIES[node.name]["summary"],
            "meaning": CLASS_SUMMARIES[node.name]["meaning"],
            "workflow": CLASS_SUMMARIES[node.name]["workflow"],
            "doc": class_doc,
            "groups": {group: [] for group in GROUP_TITLES},
        }

        for child in node.body:
            if not isinstance(child, ast.FunctionDef):
                continue
            if child.name.startswith("_") and child.name != "__init__":
                continue
            if is_property_setter(child):
                continue

            raw_attr = getattr(cls_obj, child.name)
            item_type = "property" if isinstance(raw_attr, property) or is_property_getter(child) else "method"
            doc = inspect.getdoc(raw_attr.fget if isinstance(raw_attr, property) else raw_attr) or ""
            fallback = PROPERTY_SUMMARIES if item_type == "property" else METHOD_SUMMARIES

            group = method_group(child.name, item_type)
            class_info["groups"][group].append(
                {
                    "name": child.name,
                    "display_name": child.name,
                    "type": item_type,
                    "type_label": "property" if item_type == "property" else "metódus",
                    "signature": format_signature(raw_attr, child.name, class_name=node.name),
                    "summary": short_summary(child.name, doc, fallback),
                    "doc": doc,
                    "sections": extract_sections(doc),
                    "anchor": f"{node.name}-{child.name}",
                    "search": search_blob(node.name, child.name, doc, fallback.get(child.name, "")),
                    "group_title": GROUP_TITLES[group],
                }
            )
        result[node.name] = class_info
    return result


def render_class_block(class_name: str, class_info: dict[str, Any]) -> str:
    group_html = []
    total_items = 0
    for group_key, title in GROUP_TITLES.items():
        items = class_info["groups"][group_key]
        if not items:
            continue
        total_items += len(items)
        rendered = "".join(render_api_item(item) for item in items)
        group_html.append(f"<section class='method-group'><h4>{html.escape(title)}</h4>{rendered}</section>")

    return f"""
    <section class="class-panel" id="class-{html.escape(class_name)}">
      <header class="class-header">
        <div>
          <div class="eyebrow">{badge('class', 'class')} {badge(str(total_items) + ' elem', 'group')}</div>
          <h3>{html.escape(class_name)}</h3>
        </div>
        <p class="class-summary">{html.escape(class_info['summary'])}</p>
      </header>
      <div class="class-meta">
        <section>
          <h4>Mit reprezentál?</h4>
          <p>{html.escape(class_info['meaning'])}</p>
        </section>
        <section>
          <h4>Tipikus workflow</h4>
          <p>{html.escape(class_info['workflow'])}</p>
        </section>
      </div>
      {''.join(group_html)}
    </section>
    """


def page_html(top_level: dict[str, dict[str, Any]], classes: dict[str, dict[str, Any]]) -> str:
    top_sections_html = {}
    for section_id, section in TOP_LEVEL_SECTIONS.items():
        items = [top_level[name] for name in section["members"] if name in top_level]
        top_sections_html[section_id] = f"""
            <section class="section-panel" id="section-{section_id}">
              <header class="section-header">
                <div class="eyebrow">{badge('szekció', 'group')} {badge(str(len(items)) + ' elem', 'group')}</div>
                <h2>{html.escape(section['title'])}</h2>
                <p>{html.escape(section['description'])}</p>
              </header>
              {''.join(render_api_item(item) for item in items)}
            </section>
            """

    total_top = len(top_level)
    total_class_items = sum(
        len(items)
        for class_info in classes.values()
        for items in class_info["groups"].values()
    )

    return f"""<!doctype html>
<html lang="hu">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RENDES_fizinfo.py dokumentáció</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --panel: rgba(255, 252, 248, 0.92);
      --panel-strong: #fffdfa;
      --line: #dbcdb9;
      --text: #1e1f22;
      --muted: #5f645f;
      --accent: #0d6b78;
      --accent-soft: #d9eef1;
      --accent-2: #a54b2a;
      --shadow: 0 12px 32px rgba(53, 41, 23, 0.08);
      --radius: 18px;
      --radius-sm: 12px;
      --sans: "Aptos", "Segoe UI", "Trebuchet MS", sans-serif;
      --mono: "Cascadia Code", "Consolas", monospace;
    }}

    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(13, 107, 120, 0.12), transparent 32%),
        radial-gradient(circle at top right, rgba(165, 75, 42, 0.10), transparent 28%),
        linear-gradient(180deg, #fbf8f3 0%, var(--bg) 58%, #f2ede5 100%);
    }}

    .page {{
      display: grid;
      grid-template-columns: 290px minmax(0, 1fr);
      gap: 24px;
      max-width: 1520px;
      margin: 0 auto;
      padding: 24px;
    }}

    .sidebar {{
      position: sticky;
      top: 16px;
      align-self: start;
      padding: 20px;
      border: 1px solid rgba(219, 205, 185, 0.8);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}

    .brand {{
      margin-bottom: 18px;
    }}

    .brand h1 {{
      margin: 0 0 8px;
      font-size: 1.65rem;
      line-height: 1.1;
    }}

    .brand p {{
      margin: 0;
      color: var(--muted);
      font-size: 0.96rem;
      line-height: 1.45;
    }}

    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 16px 0 22px;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.01em;
      border: 1px solid transparent;
    }}

    .badge-default, .badge-method {{
      background: #f0f4f5;
      color: #134a53;
      border-color: #d4e2e5;
    }}

    .badge-function {{
      background: #e7f3ee;
      color: #1f5f44;
      border-color: #cbe2d6;
    }}

    .badge-property {{
      background: #f7f0da;
      color: #7a5b00;
      border-color: #ebdfb0;
    }}

    .badge-class {{
      background: #fbe7df;
      color: #8a3c1d;
      border-color: #f2cfbf;
    }}

    .badge-group {{
      background: transparent;
      color: var(--muted);
      border-color: var(--line);
    }}

    .search-box {{
      margin-bottom: 18px;
    }}

    .search-box label {{
      display: block;
      margin-bottom: 8px;
      font-weight: 700;
      font-size: 0.9rem;
    }}

    .search-box input {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      font: inherit;
      color: var(--text);
    }}

    .search-help {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.85rem;
      line-height: 1.4;
    }}

    .nav-block h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}

    .nav-block a {{
      display: block;
      padding: 8px 10px;
      margin: 4px 0;
      color: var(--text);
      text-decoration: none;
      border-radius: 10px;
    }}

    .nav-block a:hover {{
      background: rgba(13, 107, 120, 0.08);
    }}

    .content {{
      min-width: 0;
    }}

    .hero {{
      padding: 28px;
      border: 1px solid rgba(219, 205, 185, 0.85);
      border-radius: 28px;
      background:
        linear-gradient(140deg, rgba(255,255,255,0.96), rgba(248,245,239,0.96)),
        linear-gradient(120deg, rgba(13,107,120,0.10), rgba(165,75,42,0.08));
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}

    .hero .eyebrow {{
      color: var(--accent);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 0.78rem;
      margin-bottom: 10px;
    }}

    .hero h2 {{
      margin: 0 0 12px;
      font-size: clamp(1.8rem, 3vw, 2.8rem);
      line-height: 1.05;
    }}

    .hero p {{
      max-width: 78ch;
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.55;
    }}

    .section-panel, .class-panel {{
      margin-bottom: 22px;
      padding: 22px;
      border: 1px solid rgba(219, 205, 185, 0.88);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}

    .section-header h2, .class-header h3 {{
      margin: 6px 0 10px;
      font-size: 1.5rem;
    }}

    .section-header p, .class-summary, .class-meta p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}

    .class-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin: 18px 0 8px;
    }}

    .class-meta section {{
      padding: 16px;
      border-radius: var(--radius-sm);
      background: rgba(255, 255, 255, 0.55);
      border: 1px solid rgba(219, 205, 185, 0.7);
    }}

    .class-meta h4, .method-group h4 {{
      margin: 0 0 8px;
      font-size: 1rem;
    }}

    .method-group {{
      margin-top: 20px;
    }}

    .api-item {{
      margin-top: 14px;
      border: 1px solid rgba(219, 205, 185, 0.9);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.76);
      overflow: hidden;
    }}

    .api-item summary {{
      list-style: none;
      cursor: pointer;
      padding: 16px 18px;
    }}

    .api-item summary::-webkit-details-marker {{
      display: none;
    }}

    .api-item[open] summary {{
      border-bottom: 1px solid rgba(219, 205, 185, 0.8);
      background: rgba(13, 107, 120, 0.04);
    }}

    .summary-head {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 8px;
    }}

    .summary-title {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      min-width: 0;
    }}

    .item-name {{
      font-size: 1.03rem;
      font-weight: 800;
    }}

    .signature {{
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      color: #234956;
      background: rgba(215, 236, 240, 0.72);
      border: 1px solid rgba(190, 219, 223, 0.9);
      border-radius: 10px;
      padding: 6px 10px;
      font-size: 0.85rem;
      max-width: 48%;
    }}

    .summary-text {{
      color: var(--muted);
      line-height: 1.5;
    }}

    .item-body {{
      padding: 18px;
      display: grid;
      gap: 16px;
    }}

    .item-body section {{
      padding: 14px 16px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(228, 220, 208, 0.9);
    }}

    .item-body h5 {{
      margin: 0 0 10px;
      font-size: 0.95rem;
      color: var(--accent-2);
    }}

    .item-body p, .item-body li {{
      color: var(--text);
      line-height: 1.55;
    }}

    .item-body ul {{
      margin: 0;
      padding-left: 18px;
    }}

    .muted {{
      color: var(--muted);
    }}

    .hidden-by-search {{
      display: none !important;
    }}

    @media (max-width: 1080px) {{
      .page {{
        grid-template-columns: 1fr;
      }}

      .sidebar {{
        position: static;
      }}

      .class-meta {{
        grid-template-columns: 1fr;
      }}

      .summary-head {{
        flex-direction: column;
      }}

      .signature {{
        max-width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <div class="brand">
        <h1>RENDES_fizinfo.py</h1>
        <p>Gyorsan kereshető, oktatási fókuszú HTML-dokumentáció a teljes publikus API-ról.</p>
      </div>

      <div class="stats">
        {badge(str(total_top) + " top-level elem", "function")}
        {badge("2 class", "class")}
        {badge(str(total_class_items) + " class-elem", "method")}
      </div>

      <div class="search-box">
        <label for="doc-search">Gyors kereső</label>
        <input id="doc-search" type="search" placeholder="Keresés névre, docstringre, témára...">
        <div class="search-help">Szűr névre, signature-re és a rövid magyar leírásra is.</div>
      </div>

      <nav class="nav-block">
        <h2>Ugrás a szekciókhoz</h2>
        <a href="#section-altalanos">Általános számítási függvények</a>
        <a href="#class-num_kinem">num_kinem</a>
        <a href="#class-num_dinam">num_dinam</a>
        <a href="#section-gps">GPS és smoothing</a>
        <a href="#section-modellek">Kész fizikai modellek</a>
      </nav>
    </aside>

    <main class="content">
      <section class="hero">
        <div class="eyebrow">Cheat sheet + referencia</div>
        <h2>Fizikai programozáshoz készült, böngészhető API-doksi</h2>
        <p>
          A cél nem csak az, hogy lásd, milyen függvények vannak a modulban, hanem az is,
          hogy gyorsan megtaláld, melyik elem mire való. A top-level függvények, a két fő class,
          a publikus metódusok és a property-k külön, kereshető és összecsukható blokkokban
          jelennek meg.
        </p>
      </section>

      {top_sections_html['altalanos']}
      {render_class_block('num_kinem', classes['num_kinem'])}
      {render_class_block('num_dinam', classes['num_dinam'])}
      {top_sections_html['gps']}
      {top_sections_html['modellek']}
    </main>
  </div>

  <script>
    const search = document.getElementById('doc-search');
    const items = Array.from(document.querySelectorAll('.api-item'));
    const sections = Array.from(document.querySelectorAll('.section-panel, .class-panel, .method-group'));

    function updateSearch() {{
      const query = search.value.trim().toLowerCase();

      items.forEach((item) => {{
        const haystack = item.dataset.search || '';
        const visible = !query || haystack.includes(query);
        item.classList.toggle('hidden-by-search', !visible);
      }});

      sections.forEach((section) => {{
        const visibleItems = section.querySelectorAll('.api-item:not(.hidden-by-search)');
        section.classList.toggle('hidden-by-search', visibleItems.length === 0);
      }});
    }}

    search.addEventListener('input', updateSearch);
  </script>
</body>
</html>
"""


def main() -> None:
    module = load_module()
    tree, _source = parse_source()
    public_names = get_all_names(tree)
    top_level = collect_top_level_data(module, tree, public_names)
    classes = collect_class_data(module, tree)
    OUTPUT_PATH.write_text(page_html(top_level, classes), encoding="utf-8")
    print(f"Kész: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
