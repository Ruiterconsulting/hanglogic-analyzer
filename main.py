def detect_inner_contours(shape):
    """
    Detecteert ALLE binnencontouren (rond, sleuf, vierkant, complex)
    en negeert buitencontouren of randafrondingen.
    """
    contours = []
    contour_id = 0

    try:
        bbox = shape.BoundingBox()
        max_dim = max(bbox.xlen, bbox.ylen, bbox.zlen)

        for face_id, face in enumerate(shape.Faces()):
            wires = face.Wires()
            if len(wires) <= 1:
                continue  # geen binnencontouren

            outer_wire = wires[0]
            inner_wires = wires[1:]

            for wire in inner_wires:
                contour_id += 1
                points = []
                for edge in wire.Edges():
                    for vertex in edge.Vertices():
                        v = vertex.toTuple()
                        points.append([
                            round(float(v[0]), 3),
                            round(float(v[1]), 3),
                            round(float(v[2]), 3)
                        ])

                # Sla contouren over die maar 1 of 2 punten hebben
                if len(points) < 3:
                    continue

                # Gemiddelde positie = centrum
                cx = sum(p[0] for p in points) / len(points)
                cy = sum(p[1] for p in points) / len(points)
                cz = sum(p[2] for p in points) / len(points)

                # Grootte (schatting)
                dx = max(p[0] for p in points) - min(p[0] for p in points)
                dy = max(p[1] for p in points) - min(p[1] for p in points)
                dz = max(p[2] for p in points) - min(p[2] for p in points)
                diag = (dx**2 + dy**2 + dz**2) ** 0.5

                # Alleen contouren die klein genoeg zijn (max 50% van grootste dimensie)
                if diag > (max_dim * 0.5):
                    continue

                # Vormclassificatie
                edge_types = [e.geomType() for e in wire.Edges()]
                if all(t == "CIRCLE" for t in edge_types):
                    contour_type = "round"
                elif any(t == "ELLIPSE" for t in edge_types):
                    contour_type = "oval"
                elif edge_types.count("LINE") >= 4:
                    contour_type = "rectangular"
                else:
                    contour_type = "complex"

                contours.append({
                    "id": contour_id,
                    "face_id": face_id,
                    "type": contour_type,
                    "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
                    "points": points,
                    "edgeCount": len(wire.Edges()),
                    "size_estimate": round(diag, 3)
                })

        print(f"🧩 Detected {len(contours)} filtered inner contours.")
    except Exception as e:
        print("⚠️ Inner contour detection failed:", e)

    return contours
