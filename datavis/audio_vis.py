import pandas as pd
import plotly.graph_objects as go

SUPPORTED_FORMATS = ['html', 'png', 'webp', 'svg', 'pdf', 'eps']


def save_heatmap_with_datetime(df: pd.DataFrame, output_path: str, dformat: str = 'html') -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=df.T,
        x=df.index,
        y=df.columns.values,
        colorscale='Viridis'))

    if dformat == 'html':
        fig.write_html(output_path)
    elif dformat in SUPPORTED_FORMATS:
        fig.write_image(output_path, format=dformat)
    else:
        raise NotImplemented(f'Format {dformat} is not supported')