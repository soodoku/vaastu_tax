import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


def test_analysis_plot_renders_with_pillow(tmp_path):
    fig, axis = plt.subplots()
    axis.scatter([1, 2, 3], [2, 4, 5])
    axis.set(xlabel="Area", ylabel="Price")
    output = tmp_path / "analysis.png"
    fig.savefig(output)
    plt.close(fig)
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width > 0 and image.height > 0
        assert image.convert("RGB").getextrema() != ((255, 255),) * 3
