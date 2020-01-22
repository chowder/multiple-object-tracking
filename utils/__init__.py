def save(tracks, filename):
    with open(filename, "w+") as f:
        for track in tracks:
            f.write(str(track))
            f.write("\n")
