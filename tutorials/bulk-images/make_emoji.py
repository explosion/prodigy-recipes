import pathlib
import pandas as pd
from sklearn.pipeline import make_pipeline 
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

from embetter.grab import ColumnGrabber
from embetter.vision import ImageLoader, ColorHistogramEncoder

# Pipeline using just colors
image_emb_pipeline = make_pipeline(
    ColumnGrabber("path"),
    ImageLoader(convert="RGB"),
    ColorHistogramEncoder(),
    UMAP(),
    MinMaxScaler()
)

# Load the image paths
img_paths = list(pathlib.Path("downloads", "twemoji").glob("*"))
dataf = pd.DataFrame({
    "path": [str(p) for p in img_paths][:1000]
})

# Apply the color histograms and umap
X = image_emb_pipeline.fit_transform(dataf)
dataf['x'] = X[:, 0]
dataf['y'] = X[:, 1]

# Save to disk
dataf.to_csv("twemoji.csv", index=False)
