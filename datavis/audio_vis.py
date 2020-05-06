import pandas as pd
import plotly.graph_objects as go

SUPPORTED_FORMATS = ['html', 'png', 'webp', 'svg', 'pdf', 'eps']

sns_colorscale = [[0.0, '#3f7f93'],
 [0.071, '#5890a1'],
 [0.143, '#72a1b0'],
 [0.214, '#8cb3bf'],
 [0.286, '#a7c5cf'],
 [0.357, '#c0d6dd'],
 [0.429, '#dae8ec'],
 [0.5, '#f2f2f2'],
 [0.571, '#f7d7d9'],
 [0.643, '#f2bcc0'],
 [0.714, '#eda3a9'],
 [0.786, '#e8888f'],
 [0.857, '#e36e76'],
 [0.929, '#de535e'],
 [1.0, '#d93a46']]


def save_heatmap_with_datetime(df: pd.DataFrame, output_path: str, dformat: str = 'html'):
    fig = go.Figure(data=go.Heatmap(
        z=df.T,
        x=df.index,
        y=df.columns.values,
        colorscale='Viridis'))
    save_figure(fig, dformat, output_path)


def save_corr_matrix(df: pd.DataFrame, output_path: str, dformat: str = 'html'):
    corr = df.corr().values
    col_names = df.columns.values
    N = len(corr)
    corr = [[corr[i][j] if i > j else None for j in range(N)] for i in range(N)]
    text = [[f'corr({col_names[i]}, {col_names[j]}) = {corr[i][j]:.2f}' if i > j else '' for j in range(N)] for i in range(N)]
    heatmap = go.Heatmap(
        z=corr,
        x=col_names,
        y=col_names,
        xgap=1, ygap=1,
        colorscale=sns_colorscale,
        colorbar_thickness=20,
        colorbar_ticklen=3,
        hovertext=text,
        hoverinfo='text'
    )

    layout = go.Layout(
        title_text='Correlation matrix', title_x=0.5,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
     )
    fig = go.Figure(data=heatmap, layout=layout)
    save_figure(fig, dformat, output_path)


def save_figure(fig: go.Figure, dformat: str, output_path: str):
    if dformat == 'html':
        fig.write_html(output_path)
    elif dformat in SUPPORTED_FORMATS:
        fig.write_image(output_path, format=dformat)
    else:
        raise NotImplemented(f'Format {dformat} is not supported')