import pathlib
import pandas as pd
from sklearn.pipeline import make_pipeline 
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

from embetter.grab import ColumnGrabber
from embetter.vision import ImageLoader, TimmEncoder

for model in ['xception', 'mobilenetv3_large_100']:
    image_emb_pipeline = make_pipeline(
      ColumnGrabber("path"),
      ImageLoader(convert="RGB"),
      TimmEncoder(model),
      UMAP(),
      MinMaxScaler()
    )
    
    # Make dataframe with image paths
    img_paths = list(pathlib.Path("downloads", "pets").glob("*"))
    dataf = pd.DataFrame({
      "path": [str(p) for p in img_paths]
    })
    
    # Make csv file with Umap'ed model layer 
    X = image_emb_pipeline.fit_transform(dataf)
    dataf['x'] = X[:, 0]
    dataf['y'] = X[:, 1]
    dataf.to_csv(f"pets-{model}.csv", index=False)
