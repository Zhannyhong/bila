"""
Convert the PBBM book library's Collada meshes to OBJ, for the MuJoCo scene.

Why this exists
  `TAMS-Group/PBBM-robot-librarian` models its books as Collada (.dae), which
  MuJoCo cannot load. Every book mesh is a unit cube (eight vertices at +-1)
  whose six faces carry the six textures of a book: front cover, back cover,
  top and bottom edges, spine and fore edge. The real size comes from the URDF,
  which scales that cube by the book's half extents (urdf/book_cube.xacro), so
  one conversion per book is enough - only the textures differ.

  Two MuJoCo limits shape the output:
    * one material per mesh, so the cube is written as six one-face OBJ files
      (`<stem>/material0.obj` ... `material5.obj`) next to the whole mesh
      (`<stem>/model.obj`). `shelf_scene.py` attaches the six as non-colliding
      visual geoms on the book's box geom.
    * textures must be PNG (a .jpg fails with "unsupported content type"), and
      the six JPGs of the 27 books are 608 MB of pixels decoded, so they are
      re-encoded as downscaled PNGs. Books that share an image share the PNG.

  One book was never finished in the repo: book-ai86.dae still names the
  template cube's numbered placeholders (blender/cube/1.png ... 6.png), though
  the book's six photos sit beside it. PHOTO_FIXES points it at them, turned as
  the finished books' copies are (see there).

  The repo's articulated books are also made of unit cubes, one per page
  section, in a subdirectory of `meshes/` (`voynich/book-voynich-part1.dae`).
  They convert the same way, to `<book>-part<N>`, and `shelf_scene.py` draws one
  per hinged section (ShelfConfig(book_pages=N)) instead of one per book.

Usage
  git clone --depth 1 --filter=blob:none --sparse \\
      https://github.com/TAMS-Group/PBBM-robot-librarian /tmp/pbbm
  git -C /tmp/pbbm sparse-checkout set librarian_common/tams_books_world

  python -m bila.sim.dae_to_obj --src /tmp/pbbm/librarian_common/tams_books_world/meshes

Writes `generated_assets/books/` (gitignored) with index.json, and then:

  from bila.sim.shelf_scene import ShelfConfig
  ShelfConfig(book_meshes=True)          # or: python -m bila.sim.shelf_scene --book-meshes
  ShelfConfig(book_meshes=True, book_pages=4)   # articulated, as in the repo
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import xml.etree.ElementTree as ET

from PIL import Image

from bila import ROOT as HERE
OUT_DEFAULT = os.path.join(HERE, "generated_assets", "books")

VECTOR = {"VERTEX": 3, "NORMAL": 3, "TEXCOORD": 2}

# Books whose .dae names the template cube's placeholder images instead of the
# book's photos: {book: {placeholder: (photo beside the .dae, turn in degrees)}}.
# The template's face slots are in the finished books' order (back, cover, top,
# bottom, spine, fore edge), and a finished book wears its back, bottom, spine
# and fore-edge photos turned 180, 180, 270 and 90 degrees counter-clockwise
# (bits-back-180.jpg ... bits-front-edge-90.jpg beside the untouched originals),
# so these are turned the same way: book-ai86's photos are taken as the
# originals are (portrait, the spine's title running top to bottom).
PHOTO_FIXES = {
    "book-ai86": {"1.png": ("ai86-back.jpg", 180), "2.png": ("ai86-cover.jpg", 0), "3.png": ("ai86-top.jpg", 0),
                  "4.png": ("ai86-bottom.jpg", 180), "5.png": ("ai86-spine.jpg", 270),
                  "6.png": ("ai86-front-edge.jpg", 90)},
}


# ---------------------------------------------------------------------------
# Collada reading. The namespaces vary between exporters, so tags are compared
# without them.
# ---------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _kids(el, name: str):
    return [c for c in el if _local(c.tag) == name]


def _first(el, name: str):
    for c in el.iter():
        if _local(c.tag) == name:
            return c
    return None


def _text(el, default=None):
    return el.text.strip() if el is not None and el.text else default


def _numbers(el):
    return [float(v) for v in (_text(el, "") or "").split()]


def _libs(root) -> dict:
    """{library name: [entries]} for the libraries this converter needs."""
    names = ("library_images", "library_materials", "library_effects", "library_geometries")
    out = {}
    for lib in _kids(root, "COLLADA") or [root]:
        for name in names:
            for el in _kids(lib, name):
                out.setdefault(name, []).extend(list(el))
    return out


def read_dae(path: str):
    """(material id -> image file name, [(material id, [face, ...])]).

    A face is a list of (position, normal, texcoord) tuples, copied verbatim
    from the file: the meshes are already unit cubes in MuJoCo-loadable ranges,
    so there is nothing to transform or rescale here."""
    root = ET.parse(path).getroot()
    libs = _libs(root)

    images = {im.get("id"): _text(_first(im, "init_from")) for im in libs.get("library_images", [])}
    effects = {}
    for eff in libs.get("library_effects", []):
        for surface in eff.iter():
            if _local(surface.tag) == "surface":
                effects[eff.get("id")] = images.get(_text(_first(surface, "init_from")))
    material_image = {}
    for mat in libs.get("library_materials", []):
        url = _first(mat, "instance_effect").get("url").lstrip("#")
        material_image[mat.get("id")] = effects.get(url)

    groups = []
    for geo in libs.get("library_geometries", []):
        mesh = _first(geo, "mesh")
        if mesh is None:
            continue
        arrays = {s.get("id"): _numbers(_first(s, "float_array")) for s in _kids(mesh, "source")}
        positions = _first(mesh, "vertices")
        pos_id = _kids(positions, "input")[0].get("source").lstrip("#") if positions is not None else None
        for tri in _kids(mesh, "triangles"):
            groups.append((tri.get("material") or "default", _read_faces(tri, arrays, pos_id)))
    return material_image, groups


def _read_faces(tri, arrays: dict, pos_id: str):
    """The `<triangles>` block as quads where possible: the exporter writes each
    rectangle as two triangles sharing a diagonal, which merges back exactly."""
    slots = []
    for inp in _kids(tri, "input"):
        semantic = inp.get("semantic")
        if semantic not in VECTOR:
            continue
        source = pos_id if semantic == "VERTEX" else inp.get("source").lstrip("#")
        slots.append((int(inp.get("offset")), semantic, arrays.get(source, [])))
    if not slots or pos_id is None:
        return []
    stride = max(offset for offset, _, _ in slots) + 1
    flat = [int(v) for v in (_text(_first(tri, "p"), "") or "").split()]

    rows = []
    for k in range(0, len(flat) - stride + 1, stride):
        corner = {}
        for offset, semantic, array in slots:
            i = flat[k + offset]
            n = VECTOR[semantic]
            corner[semantic] = tuple(array[n * i:n * i + n]) or (0.0,) * n
        if "VERTEX" in corner:
            rows.append((corner["VERTEX"], corner.get("NORMAL", (0.0, 0.0, 1.0)),
                         corner.get("TEXCOORD", (0.0, 0.0))))

    if (len(rows) == 6 and rows[0][0] == rows[3][0] and rows[2][0] == rows[4][0]):
        return [[rows[0], rows[1], rows[2], rows[5]]]          # two triangles -> one quad
    return [rows[k:k + 3] for k in range(0, len(rows) - 2, 3)]


def _face_axis(face):
    """('x'|'y'|'z'|None, +-1) of an axis-aligned face: the axis all of its
    vertices share. The scenes use this to check which way a mesh faces."""
    for ax in range(3):
        values = [p[ax] for p, _, _ in face]
        if max(values) - min(values) < 1e-4:
            return "xyz"[ax], (1 if values[0] > 0 else -1)
    return None, 0


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _f(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def obj_text(groups, header) -> str:
    """`groups` is [(material id, [face, ...])]. Positions, normals and texture
    coordinates are indexed 1:1, so every face line is `f i/i/i ...`."""
    order, index = [], {}
    body = []
    for material, faces in groups:
        body.append(f"usemtl {material}")
        for face in faces:
            for corner in face:
                if corner not in index:
                    index[corner] = len(order) + 1
                    order.append(corner)
            body.append("f " + " ".join(f"{index[c]}/{index[c]}/{index[c]}" for c in face))
    lines = list(header)
    lines += [f"v {_f(c[0])}" for c in order]
    lines += [f"vt {_f(c[2])}" for c in order]
    lines += [f"vn {_f(c[1])}" for c in order]
    return "\n".join(lines + body) + "\n"


def mtl_text(entries) -> str:
    """`entries` is [(material id, texture path relative to the .mtl, or None)]."""
    lines = []
    for material, texture in entries:
        lines += [f"newmtl {material}", "Kd 1.000000 1.000000 1.000000",
                  "Ka 0.000000 0.000000 0.000000", "Ks 0.100000 0.100000 0.100000", "illum 2"]
        if texture:
            lines.append(f"map_Kd {texture}")
        lines.append("")
    return "\n".join(lines)


class SharedFiles:
    """Textures, written once per unique image whatever asset asks for them."""

    def __init__(self, root: str):
        self.dir = os.path.join(root, "tex")
        self.written = {}

    def texture(self, root: str, path: str, max_size: int, turn: int = 0) -> str:
        """The image at `path` (turned `turn` degrees counter-clockwise) as a PNG
        under `tex/`, or None if there is none."""
        if not path or not os.path.exists(path):
            return None
        with Image.open(path) as im:
            im = im.convert("RGB")
            if turn:
                im = im.rotate(turn, expand=True)
            if max_size and max(im.size) > max_size:
                scale = max_size / max(im.size)
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                               Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        data = buf.getvalue()
        stem = _safe(os.path.splitext(os.path.basename(path))[0]) + (f"-{turn}" if turn else "")
        name = f"{hashlib.md5(data).hexdigest()[:8]}-{stem}.png"
        if name not in self.written:
            os.makedirs(self.dir, exist_ok=True)
            with open(os.path.join(self.dir, name), "wb") as fh:
                fh.write(data)
            self.written[name] = len(data)
        return os.path.join("tex", name)


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def convert(dae: str, out: str, shared: SharedFiles, max_size: int) -> dict:
    """One .dae -> a directory of OBJs, a .mtl and textures. Returns its index entry."""
    stem = _safe(os.path.splitext(os.path.basename(dae))[0])
    material_image, groups = read_dae(dae)
    if not groups:
        raise ValueError(f"no triangles in {dae}")
    fixes = PHOTO_FIXES.get(stem, {})
    turns = {}
    for material, image in material_image.items():
        if image in fixes:
            material_image[material], turns[material] = fixes[image]

    textures = {}
    for material in dict.fromkeys(m for m, _ in groups):
        image = material_image.get(material)
        source = image if image and os.path.isabs(image) else os.path.join(os.path.dirname(dae), image or "")
        textures[material] = shared.texture(out, source, max_size, turns.get(material, 0))

    faces = []
    for material, group in groups:
        directory = os.path.join(out, stem)
        os.makedirs(directory, exist_ok=True)
        name = f"{material}.obj"
        with open(os.path.join(directory, name), "w") as fh:
            fh.write(obj_text([(material, group)], [f"# {os.path.basename(dae)}: {material}"]))
        axis, sign = _face_axis(group[0])
        faces.append({"material": material, "obj": f"{stem}/{name}",
                      "texture": textures[material], "image": material_image.get(material),
                      "axis": axis, "sign": sign})

    header = [f"# {os.path.basename(dae)}, converted from Collada by dae_to_obj.py",
              f"# unit cube in the source frame: x = depth (spine at -x), y = thickness, z = height",
              f"mtllib model.mtl"]
    full = obj_text([(m, g) for m, g in groups], header)
    with open(os.path.join(out, stem, "model.obj"), "w") as fh:
        fh.write(full)
    entries = [(m, os.path.relpath(os.path.join(out, t), os.path.join(out, stem)) if t else None)
               for m, t in textures.items()]
    with open(os.path.join(out, stem, "model.mtl"), "w") as fh:
        fh.write(mtl_text(entries))
    return {"name": stem, "dae": os.path.basename(dae), "obj": f"{stem}/model.obj",
            "mtl": f"{stem}/model.mtl", "faces": faces}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory holding the repo's .dae files (meshes/)")
    ap.add_argument("--out", default=OUT_DEFAULT, help="output directory (default: %(default)s)")
    ap.add_argument("--pattern", default="**/book-*.dae",
                    help="which files to convert, relative to --src; the default "
                         "reaches into the subdirectories too, where the page "
                         "sections of the articulated books live (default: %(default)s)")
    ap.add_argument("--max-tex", type=int, default=512, metavar="PX",
                    help="longest side of a texture in pixels, 0 to keep the original (default: %(default)s)")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.src, a.pattern), recursive=True))
    if not paths:
        raise SystemExit(f"no files match {a.pattern!r} in {a.src}")
    shared = SharedFiles(a.out)
    assets = []
    for path in paths:
        entry = convert(path, a.out, shared, a.max_tex)
        assets.append(entry)
        print(f"{entry['name']:22s} {len(entry['faces'])} faces  "
              f"{sum(1 for f in entry['faces'] if f['texture'])} textures")
    index = {"source": os.path.abspath(a.src), "assets": assets}
    with open(os.path.join(a.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    print(f"{len(assets)} books -> {os.path.join(a.out, 'index.json')} "
          f"({sum(shared.written.values()) / 1e6:.1f} MB of textures)")


if __name__ == "__main__":
    main()
