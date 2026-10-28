"""Select 24 real DIOR layouts and convert them to the existing demo TXT format."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

CATEGORIES = ("airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam",
              "Expressway-Service-area", "Expressway-toll-station", "golffield", "groundtrackfield",
              "harbor", "overpass", "ship", "stadium", "storagetank", "tenniscourt", "trainstation",
              "vehicle", "windmill")
POINTS = ("x_left_top", "y_left_top", "x_right_top", "y_right_top",
          "x_right_bottom", "y_right_bottom", "x_left_bottom", "y_left_bottom")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output_dir", default="outputs/exp1/layouts")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapping = {name.lower(): index for index, name in enumerate(CATEGORIES)}
    groups = {name: [] for name in ("adjacent", "dense", "single", "sparse")}
    manifest = []
    for index, path in enumerate(sorted(Path(args.annotations).glob("*.xml"))):
        annotation = ET.parse(path).getroot()
        objects = annotation.findall("object")
        if not 1 <= len(objects) <= 20:
            continue
        width, height = float(annotation.findtext("size/width")), float(annotation.findtext("size/height"))
        labels, boxes = [], []
        for obj in objects:
            labels.append(mapping[obj.findtext("name").lower()])
            box = obj.find("robndbox")
            boxes.append([float(box.findtext(key)) / (width if i % 2 == 0 else height)
                          for i, key in enumerate(POINTS)])
        centers = [(sum(box[::2])/4, sum(box[1::2])/4) for box in boxes]
        sizes = [(max(box[::2])-min(box[::2]), max(box[1::2])-min(box[1::2])) for box in boxes]
        pairs = [(i, j, math.dist(centers[i], centers[j]))
                 for i in range(len(boxes)) for j in range(i)]
        close_target = any(labels[i] == labels[j] and labels[i] in (13, 18)
                           and distance < 1.2 * max(*sizes[i], *sizes[j]) for i, j, distance in pairs)
        majority = Counter(labels).most_common(1)[0][1]
        group = None
        if 2 <= len(boxes) <= 5 and close_target:
            group = "adjacent"
        elif 6 <= len(boxes) <= 20 and majority / len(boxes) >= 0.7:
            group = "dense"
        elif len(boxes) == 1 and labels[0] in (0, 13, 15, 18) and sizes[0][0] * sizes[0][1] >= 0.01:
            group = "single"
        elif 2 <= len(boxes) <= 5 and min(distance for _, _, distance in pairs) > 0.25:
            group = "sparse"
        if group is not None and len(groups[group]) < 6:
            stem = f"{group}_{path.stem}"
            lines = [str(label) + " " + " ".join(f"{value:.8f}" for value in box)
                     for label, box in zip(labels, boxes)]
            (output/f"{stem}.txt").write_text("\n".join(lines) + "\n")
            groups[group].append(stem)
            manifest.append(dict(layout=stem, group=group, source=str(path), instances=len(boxes),
                                 categories=[CATEGORIES[label] for label in labels]))
            print(stem, manifest[-1]["categories"], flush=True)
        if all(len(items) == 6 for items in groups.values()):
            break
        if index % 500 == 0:
            print("Scanned", index + 1, {key: len(value) for key, value in groups.items()}, flush=True)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print({key: len(value) for key, value in groups.items()}, flush=True)
    assert len(manifest) == 24, "Expected six layouts per group"


if __name__ == "__main__":
    main()
