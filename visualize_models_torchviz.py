"""Generate model graphs (torchviz) for SimpleNN, EEG_CNN and EEG_CNN2.

Creates dot and png files in the `Images/` folder. If Graphviz is not
installed the script will save the dot source and print installation tips.

Usage:
    python visualize_models_torchviz.py
"""

from pathlib import Path
import torch
import shutil
import subprocess

# sys not needed

from support_func import NN_classes


OUTDIR = Path("Images")
OUTDIR.mkdir(exist_ok=True)


def make_dummy_input(model_name, num_channels=32, num_timepoints=256, batch=1):
    # Models in this repo expect shape (batch, channels, time)
    if model_name == "SimpleNN":
        # SimpleNN flattens, but we still provide (batch, channels, time)
        return torch.randn(batch, num_channels, num_timepoints)
    elif model_name in ("EEG_CNN", "EEG_CNN2"):
        return torch.randn(batch, num_channels, num_timepoints)
    else:
        return torch.randn(batch, num_channels, num_timepoints)


def visualize_model(model_cls, model_name, device=torch.device("cpu")):
    model = model_cls(32, 256, 2).to(device)
    model.eval()

    x = make_dummy_input(model_name).to(device)

    try:
        from torchviz import make_dot

        out = model(x)
        dot = make_dot(out, params=dict(model.named_parameters()))

        dot_path = OUTDIR / f"{model_name}.dot"
        png_path = OUTDIR / f"{model_name}.png"

        dot.save(str(dot_path))

        # Try to render PNG and SVG using system 'dot' if available
        dot_bin = shutil.which("dot")
        if dot_bin is not None:
            try:
                # PNG
                subprocess.run(
                    [
                        dot_bin,
                        "-Tpng",
                        str(dot_path),
                        "-o",
                        str(png_path),
                    ],
                    check=True,
                )

                # SVG
                svg_path = OUTDIR / f"{model_name}.svg"
                subprocess.run(
                    [
                        dot_bin,
                        "-Tsvg",
                        str(dot_path),
                        "-o",
                        str(svg_path),
                    ],
                    check=True,
                )

                print(f"Rendered: {png_path} and {svg_path}")
            except subprocess.CalledProcessError:
                print("dot failed to render. Dot saved to:", dot_path)
        else:
            print("Graphviz 'dot' not found in PATH. Dot saved to:", dot_path)

    except ImportError:
        # Fallback: save a tiny representation
        dot_path = OUTDIR / f"{model_name}.dot"
        with open(dot_path, "w") as fh:
            fh.write(f"// torchviz not installed. Model: {model_name}\n")
        print(f"torchviz not installed. Saved placeholder dot to {dot_path}")


def visualize_simple(model_cls, model_name, device=torch.device("cpu")):
    """Create a compact graph showing module type and output shapes.

    This uses forward hooks to record the output tensor shapes of modules
    during a single forward pass on a dummy input. Then builds a small DOT
    where each node is "module_name: ModuleType [shape]".
    """
    model = model_cls(32, 256, 2).to(device)
    model.eval()

    hooks = []
    records = []

    def make_hook(name):
        def hook(module, inp, out):
            try:
                shape = tuple(out.shape)
            except Exception:
                shape = str(type(out))
            records.append((name, module.__class__.__name__, shape))

        return hook

    # register hooks for leaf modules
    for name, module in model.named_modules():
        # skip the top-level module (empty name)
        if name == "":
            continue
        # only attach to common layer types to avoid clutter
        if isinstance(
            module,
            (
                torch.nn.Conv1d,
                torch.nn.Linear,
                torch.nn.BatchNorm1d,
                torch.nn.MaxPool1d,
                torch.nn.Dropout,
                torch.nn.GRU,
                torch.nn.LSTM,
            ),
        ):
            hooks.append(module.register_forward_hook(make_hook(name)))

    x = make_dummy_input(model_name).to(device)
    with torch.no_grad():
        try:
            _ = model(x)
        except Exception:
            # Some models may expect different shapes; ignore failures
            pass

    for h in hooks:
        h.remove()

    # Build a simple dot
    simple_path = OUTDIR / f"{model_name}_simple.dot"
    with open(simple_path, "w", encoding="utf-8") as fh:
        fh.write("digraph G {\n")
        fh.write("rankdir=LR; node [shape=record];\n")
        for i, (name, mtype, shape) in enumerate(records):
            label = f"{name}: {mtype}\\n{shape}"
            fh.write(f'node{i} [label="{label}"];\n')
            if i > 0:
                fh.write(f"node{i-1} -> node{i};\n")
        fh.write("}\n")

    # Try render
    dot_bin = shutil.which("dot") or None
    if dot_bin:
        png = simple_path.with_suffix(".png")
        try:
            subprocess.run(
                [dot_bin, "-Tpng", str(simple_path), "-o", str(png)],
                check=True,
            )
            print(f"Rendered simplified graph: {png.name}")
        except Exception:
            print(f"Could not render simplified graph for {model_name}")
    else:
        print(f"Saved simplified dot to {simple_path}")


def main():
    models = ["SimpleNN", "EEG_CNN", "EEG_CNN2"]
    for name in models:
        if hasattr(NN_classes, name):
            cls = getattr(NN_classes, name)
            print(f"Visualizing {name}...")
            visualize_simple(cls, name)
        else:
            print(f"Model {name} not found in NN_classes")

    # Post-processing: rename files without extension that look like DOT
    for p in OUTDIR.iterdir():
        if p.is_file() and p.suffix == "":
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as fh:
                    head = fh.read(100)
                if "digraph" in head:
                    newp = p.with_suffix(".dot")
                    print(f"Renaming {p.name} -> {newp.name}")
                    p.rename(newp)
            except Exception:
                pass

    # Find dot binary (try PATH then common install locations)
    def find_dot_binary():
        dot = shutil.which("dot")
        if dot:
            return dot

        candidates = [
            r"C:\Program Files\Graphviz\bin\dot.exe",
            r"C:\Program Files (x86)\Graphviz2.38\bin\dot.exe",
            r"C:\Program Files\Graphviz2.44.1\bin\dot.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
        return None

    dot_bin = find_dot_binary()
    if dot_bin is None:
        print("Graphviz 'dot' not found. Install Graphviz or add it to PATH.")
        print("After install, run: dot -V to verify, then rerun this script.")
        print("Or render .dot files manually in PowerShell:")
        print(r"Get-ChildItem -Filter *.dot | ForEach-Object {")
        print(r"  dot -Tpng $_.FullName -o <output.png>")
        print(r"}")
        return

    # Render all .dot files in Images/
    for dotfile in OUTDIR.glob("*.dot"):
        png = dotfile.with_suffix(".png")
        svg = dotfile.with_suffix(".svg")
        try:
            subprocess.run(
                [dot_bin, "-Tpng", str(dotfile), "-o", str(png)],
                check=True,
            )
            subprocess.run(
                [dot_bin, "-Tsvg", str(dotfile), "-o", str(svg)],
                check=True,
            )
            print(f"Rendered {png.name} and {svg.name}")
        except subprocess.CalledProcessError:
            print(f"Failed to render {dotfile.name} with {dot_bin}")

    print("Done. Check the Images/ folder for .png/.svg files.")


if __name__ == "__main__":
    main()
